import os
import statistics as stats
import sys
import timeit
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adversarial_attack import AdversarialSupervisor, NodeSelector
from data_loader import DataLoader, Splitter
from evaluation.ModelSupervisor import ModelSupervisor
from greatx.attack.targeted import FGAttack, PGDAttack, RandomAttack, SGAttack
from static import *
from utility import ConfigHandler
from utility.util import logger


class AdversarialAssessment:
    """
    Orchestrates adversarial robustness assessment of adversarial methods by managing experimental
    configuration, data loading, and node selection for evaluation.

    Args:
        - model: Name of the GNN model to evaluate
        - dataset: Name of the dataset to use for evaluation
        - adversarial: Name of the adversarial attack method to apply
        - num_runs: Number of runs per split for averaging results (default: 3)
        - num_split: Number of data splits for cross-validation (default: 5)
        - base_seed: Base random seed for reproducibility (default: 720)
        - purification: Name of the graph purification method to apply before evaluation.
        If None, no purification is used (default: None)
        - use_tune_model: If True, uses tuned hyperparameter configuration; otherwise uses
        default configuration (default: True)
        - use_degree_node_selection: If True, selects target nodes based on both margin and
        degree (high/low degree + high/low margin + other); otherwise selects based on
        margin only (default: True)
        - device: Device to run evaluation on, e.g. 'cpu' or 'cuda' (default: 'cpu')
    """

    def __init__(
        self,
        model: str,
        dataset: str,
        adversarial: str,
        num_runs: int = 3,
        num_split: int = 5,
        base_seed: int = 720,
        purification: str = None,
        use_tune_model: bool = True,
        use_degree_node_selection: bool = True,
        device: str = "cpu",
    ) -> None:

        # Set up setting of the experiment
        self.model = model
        self.device = device
        self.dataset_name = dataset
        self.adversarial_name = adversarial
        self.num_runs = num_runs
        self.num_splits = num_split
        self.base_seed = base_seed
        self.use_tune_model = use_tune_model
        self.use_degree_node_selection = use_degree_node_selection
        self.purification_name = purification
        self.use_purification = False if purification is None else True

        # Set up resource for the evaluation
        self.data, _ = DataLoader.load(dataset)
        self.node_selector = (
            NodeSelector.select_nodes_margin_degree
            if use_degree_node_selection
            else NodeSelector.select_nodes_margin
        )
        self.node_category = (
            ["high_margin", "low_margin", "high_degree", "low_degree", "other"]
            if use_degree_node_selection
            else ["high_margin", "low_margin", "other"]
        )
        self.config_name = "best_config" if use_tune_model else "default"
        self.node_miss_classified_evasion = []
        self.node_miss_classified_poison = []
        self._log()

    def _log(self) -> None:
        """
        Log experiment setting
        """
        logger.info("==" * 30)
        logger.info("Adversarial Assessment setup")
        logger.info(f"Victim model: {self.model}")
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Adversarial Attack: {self.adversarial_name}")
        logger.info(f"Number of runs: {self.num_runs}")
        logger.info(f"Base seed: {self.base_seed}")
        logger.info(f"Use tune model: {self.use_tune_model}")
        logger.info(
            f"Use degree as extra criteria for selecting target nodes: {self.use_degree_node_selection}"
        )
        logger.info(f"Purification: {self.purification_name}")
        logger.info(f"Use Purification: {self.use_purification}")
        logger.info(f"Config setting: {self.config_name}")
        logger.info(f"Number of splits: {self.num_splits}")
        logger.info("==" * 30)

    def _get_stat(self, results_list: List) -> Tuple[int]:
        """
        The results are return in a tuple of (mean,std)
        """
        return (stats.mean(results_list), stats.stdev(results_list))

    def _load_attack(self, adversarial_name: str):
        """
        Returns the attack class corresponding to the specified adversarial attack name.

        Args:
            - adversarial_name: Name of the adversarial attack method to load

        Returns:
            - Uninstantiated attack class corresponding to the specified adversarial attack name

        Raises:
            - Exception: If the specified adversarial attack name is unsupported or undefined

        Supported attack methods:
            - sga: SGAttack
            - nettack: Nettack
            - fga: FGAttack
            - rnd: RandomAttack
            - pgdattack: PGDAttack
            - prbcdattack: PRBCDAttack
            - gottack: OrbitAttack
            - l1d_rnd_attack: RandomDegreeAttack
        """
        from adversarial_attack import (
            Nettack,
            OrbitAttack,
            PRBCDAttack,
            RandomDegreeAttack,
        )

        if adversarial_name == SGA_NAME:
            return SGAttack
        elif adversarial_name == NETTACK_NAME:
            return Nettack
        elif adversarial_name == FGA_NAME:
            return FGAttack
        elif adversarial_name == RND_NAME:
            return RandomAttack
        elif adversarial_name == PGDATTACK_NAME:
            return PGDAttack
        elif adversarial_name == PRBCD_NAME:
            return PRBCDAttack
        elif adversarial_name == GOTTACK:
            return OrbitAttack
        elif adversarial_name == L1D_RND_ATTACK:
            return RandomDegreeAttack
        else:
            raise Exception("Unsupport adversarial method")

    def _prepare_dict(self) -> Dict[str, List]:
        all_categories = self.node_category + ["total"]
        return_dict = {}
        for category in all_categories:
            return_dict[category] = []

        return return_dict

    def _load_config_split(self, data: Data) -> List[Dict]:
        """
        Loads and assembles model configurations for each data split, optionally incorporating
        split indices and purification settings.

        Args:
            - data: PyTorch Geometric Data object containing node features (x) and labels (y)

        Returns:
            - List of configuration dictionaries of length num_splits, each containing:
                - Model hyperparameters loaded from the config file
                - Split indices (idx_train, idx_val, idx_test) if use_tune_model is False
                - Purification configuration if use_purification is True

        Notes:
            - If use_tune_model is False, a default split config is loaded and data is split
            using the Splitter class; resulting fold indices are injected into each config
            - If use_purification is True, the purification config for the corresponding split
            is loaded and appended to each config under the 'purification' key
        """
        all_split_config = []
        if not self.use_tune_model:
            split_config = ConfigHandler.load_split_config(config_name="default")
            splitter = Splitter(**split_config)
            folds = splitter.split(data.x.numpy(), data.y.numpy())

        for split_idx in range(self.num_splits):
            config = ConfigHandler.load_model_config(
                self.model, self.config_name, self.dataset_name, split=split_idx
            )
            if not self.use_tune_model:
                config["split"] = {
                    IDX_TRAIN: folds[split_idx].idx_train,
                    IDX_VAL: folds[split_idx].idx_val,
                    IDX_TEST: folds[split_idx].idx_test,
                }

            if self.use_purification:
                purification_config = ConfigHandler.load_purification_config(
                    self.purification_name,
                    self.model,
                    "best_config",
                    self.dataset_name,
                    split=split_idx,
                )["model_config"]
                config["purification"] = purification_config

            all_split_config.append(config)
        return all_split_config

    def _get_surrogate_name(self, attack_method: str) -> str:
        """
        Returns the surrogate model name associated with the given attack method.

        Args:
            - attack_method: Name of the adversarial attack method

        Returns:
            - Name of the surrogate model as a string, or None if the attack method
            does not require a surrogate model:
                - SGA_NAME       -> 'SGC'
                - NETTACK_NAME   -> 'GCN_surrogate'
                - FGA_NAME       -> 'GCN_surrogate'
                - PGDATTACK_NAME -> 'GCN_surrogate'
                - PRBCD_NAME     -> 'GCN'
        """
        if attack_method == SGA_NAME:
            return "SGC"
        elif attack_method in [NETTACK_NAME, FGA_NAME, PGDATTACK_NAME]:
            return "GCN_surrogate"
        elif attack_method is PRBCD_NAME:
            return "GCN"
        return None

    def _get_gcn_surrogate(
        self,
        dataset: str,
        split_idx: int,
        seed: int,
        data: Data,
        idx_train: List,
        idx_val: List,
        idx_test: List,
    ) -> torch.nn.Module:
        """
        Loads, trains, and returns a GCN surrogate model for the given dataset split.

        Args:
            - dataset: Name of the dataset to load surrogate configuration for
            - split_idx: Index of the data split to load configuration from
            - seed: Random seed for reproducibility
            - data: PyTorch Geometric Data object used for training
            - idx_train: Indices of training nodes
            - idx_val: Indices of validation nodes
            - idx_test: Indices of test nodes

        Returns:
            - Trained GCN surrogate model as a torch.nn.Module
        """
        split_model_surrogate = ConfigHandler.load_model_config(
            "GCN", "best_config", dataset, split_idx
        )
        configs_surrogate = split_model_surrogate.get("model_config")
        surrogate = ModelSupervisor(data, self.device, seed=seed, **configs_surrogate)
        surrogate.train_model(idx_train, idx_val, idx_test)
        return surrogate.model

    def _update_particular_config(self, attacker, data):
        if self.adversarial_name == PRBCD_NAME:
            attacker.update_config(10000, 200)

    def _target_node_dict_to_tuple(self, target_node_dict):
        r"""
        conver dict of target nodes return from target nodes selector to tuple of target nodes
        """
        target_node_tuple = tuple()
        for category in target_node_dict:
            target_node_tuple = target_node_tuple + (target_node_dict[category],)

        return target_node_tuple

    def _update_dict_results(
        self,
        result_tuple: Tuple,
        num_rand: int,
        evasion_dict: Optional[Dict[str, List]] = None,
        poison_dict: Optional[Dict[str, List]] = None,
    ):
        time_total = 0
        evasion_total = 0
        poison_total = 0
        for idx, category in enumerate(self.node_category):
            result = result_tuple[idx]
            if evasion_dict is not None:
                if category == "other":
                    evasion_dict[category].append(result[EVASION] / num_rand)
                else:
                    evasion_dict[category].append(result[EVASION] / 10)
                evasion_total += result[EVASION]
            if poison_dict is not None:
                if category == "other":
                    poison_dict[category].append(result[POSION] / num_rand)
                else:
                    poison_dict[category].append(result[POSION] / 10)

                poison_total += result[POSION]

            time_total += result["time"]

        if evasion_dict is not None:
            evasion_dict["total"].append(evasion_total / 50)

        if poison_dict is not None:
            poison_dict["total"].append(poison_total / 50)
        return time_total, evasion_dict, poison_dict

    def evaluate(
        self,
        evasion: bool = True,
        poision: bool = True,
        budgets_list: Optional[List[int]] = [1, 2, 3, 4, 5],
    ) -> Tuple[pd.DataFrame]:
        """
        Evaluates the adversarial robustness of the model under evasion and/or poison attacks
        across a range of budget constraints.

        Args:
            - evasion: If True, performs evasion attack evaluation (default: True)
            - poision: If True, performs poison attack evaluation (default: True)
            - budgets_list: List of budget values defining the perturbation constraints
            for each attack (default: [1, 2, 3, 4, 5])

        Returns:
            - Tuple of DataFrames containing evaluation results for each budget,
            summarized across splits and runs for evasion and poison attacks respectively.
            Each cell is a tuple of (mean,std)
        """
        assert evasion or poision, "At least on of evasion and poision setting"
        num_rand = 10 if self.use_degree_node_selection else 30
        rowlist_evasion = [] if evasion else None
        rowlist_poision = [] if poision else None
        all_split_configs = self._load_config_split(self.data)

        for budget in budgets_list:
            evasion_dict = self._prepare_dict() if evasion else None
            poision_dict = self._prepare_dict() if poision else None
            time_list = []
            time_defense_list = []

            for split_idx in range(self.num_splits):
                split_model = all_split_configs[split_idx]
                split = split_model["split"]
                configs = split_model.get("model_config")
                purification_config = split_model.get("purification", None)
                idx_train, idx_val, idx_test = (
                    split["idx_train"],
                    split["idx_val"],
                    split["idx_test"],
                )

                for idx_run in range(self.num_runs):
                    seed = self.base_seed + idx_run

                    # Select target nodes
                    node_selector = ModelSupervisor(
                        self.data,
                        self.device,
                        seed=seed,
                        use_purification=self.use_purification,
                        purification_config=purification_config,
                        **configs,
                    )
                    start_defense = timeit.default_timer()
                    node_selector.train_model(idx_train, idx_val, idx_test)
                    time_defense_list.append(timeit.default_timer() - start_defense)
                    target_node_dict = self.node_selector(
                        node_selector, self.data, idx_test, seed=seed, num_rand=num_rand
                    )
                    target_node_tuple = self._target_node_dict_to_tuple(
                        target_node_dict
                    )

                    # Load & set up adversarial attack
                    attacker = self._load_attack(self.adversarial_name)(
                        self.data, device=self.device, seed=seed
                    )
                    self._update_particular_config(attacker, self.data)
                    surrogate_name = self._get_surrogate_name(self.adversarial_name)

                    if surrogate_name is not None:
                        split_model_surrogate = ConfigHandler.load_model_config(
                            surrogate_name, "best_config", self.dataset_name, split_idx
                        )
                        configs_surrogate = split_model_surrogate.get("model_config")
                        surrogate = ModelSupervisor(
                            self.data, self.device, seed=seed, **configs_surrogate
                        )
                        surrogate.train_model(idx_train, idx_val, idx_test)
                        attacker.setup_surrogate(surrogate.model)

                    # Perform adversarial attack
                    adversarial_supervisor = AdversarialSupervisor(
                        self.data,
                        attacker,
                        idx_train,
                        idx_val,
                        idx_test,
                        self.device,
                        use_purification=self.use_purification,
                        purification_config=purification_config,
                        **configs,
                    )
                    result_bucket = adversarial_supervisor.target_attack(
                        target_node_tuple,
                        budget,
                        evasion=True,
                        poision=True,
                        evasion_model=node_selector,
                        seed=seed,
                    )
                    adversarial_supervisor.save_result(
                        self.adversarial_name,
                        seed,
                        self.dataset_name,
                        self.model,
                        budget,
                        self.purification_name,
                    )
                    time, evasion_dict, poision_dict = self._update_dict_results(
                        result_bucket, num_rand, evasion_dict, poision_dict
                    )
                    time_list.append(time)

            # Update results for each budget
            if evasion:
                row_evasion = [budget]
                for key in evasion_dict:
                    row_evasion.append(self._get_stat(evasion_dict[key]))
                row_evasion.append(self._get_stat(time_list))
                row_evasion.append(self._get_stat(time_defense_list))

                rowlist_evasion.append(row_evasion)
            if poision:
                row_poison = [budget]
                for key in evasion_dict:
                    row_poison.append(self._get_stat(poision_dict[key]))
                row_poison.append(self._get_stat(time_list))
                row_poison.append(self._get_stat(time_defense_list))

                rowlist_poision.append(row_poison)

        if not os.path.exists(
            "{}{}/{}/".format(PATH_RESULT, self.model, self.dataset_name)
        ):
            os.makedirs("{}{}/{}/".format(PATH_RESULT, self.model, self.dataset_name))

        columns = (
            ["budget"]
            + self.node_category
            + ["total"]
            + ["time_attack"]
            + ["time_defense"]
        )

        df_poision = pd.DataFrame(rowlist_poision, columns=columns) if evasion else None
        df_evasion = pd.DataFrame(rowlist_evasion, columns=columns) if poision else None

        return df_evasion, df_poision
