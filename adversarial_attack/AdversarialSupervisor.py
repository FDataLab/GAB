import json
import os
import statistics as stats
import sys
import timeit

import torch
from torch_geometric.utils import degree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Tuple

from torch_geometric.data import Data

from evaluation.ModelSupervisor import ModelSupervisor
from greatx.attack.attacker import Attacker
from static import *
from utility.util import logger


class AdversarialSupervisor:
    r"""
    Given an adversarial method, perform target attack in either evasion or poison setting or both on victim model.

    Args:
        - adversarial : adversarial method
        - idx_train: list of nodes for training
        - idx_val: list of nodes for validation
        - idx_test: list of nodes for testing
        - device: cpu or 'cuda'
        - use_purification: True/False to indicate whether victim model has purification
        - purification_config: when use_purification is True, need to provide the configuration of purification
        - victim_configs: model configuration of victim models

    """

    def __init__(
        self,
        data: Data,
        adversarial: Attacker,
        idx_train: List,
        idx_val: List,
        idx_test: List,
        device: str = "cpu",
        use_purification: bool = False,
        purification_config: dict = None,
        **victim_configs,
    ) -> None:
        self.use_purification = use_purification
        self.purification_config = purification_config
        self.data = data
        self.adversarial = adversarial
        self.device = device
        self.victim_configs = victim_configs
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.degrees = (
            degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            .cpu()
            .numpy()
        )

        self.labels = []
        self.evasion_preds = []
        self.poision_preds = []
        self.gpu_mem_usage_list = []
        self.node_miss_classified_evasion = []
        self.node_miss_classified_poison = []

    def target_attack(
        self,
        bucket_target_nodes: Tuple[List[int], ...],
        budget: int,
        evasion: bool = False,
        poision: bool = True,
        evasion_model: ModelSupervisor = None,
        seed: int = 720,
        structure_attack: bool = True,
        feature_attack: bool = False,
        use_epsilon: bool = False,
    ) -> Tuple[List[int], ...]:
        """
        Perform target attack

        Args:
            - bucket_target_nodes: Tuple of lists of target node. Each list contains target nodes from the same category (high degree/low degree/high margin/etc.)
            - budget: budget constraint to perform attack
            - evasion: True/False to indicate whether to perform evasion attack
            - poision: True/False to indicate whether to perform poison attack
            - evasion_model: model to perform evasion attack on. To avoid re-train the evasion model from scratch
            - seed: random seed
            - structure_attack: True/False to indicate to perform structure attack
            - feature_attack: True/False to indicate to perform feature attack
            - use_epsilon: Instead of fix budget for every target nodes. The budget is defined according to the degree of target nodes
        """
        assert evasion or poision, "Either evasion or posion setting"
        self.gpu_mem_usage_list = []
        return_tuple = ()
        counter = 0

        for (
            target_nodes
        ) in (
            bucket_target_nodes
        ):  # For each list of target nodes that belong to the same category
            result = {"time": 0}  # total time
            if evasion:
                assert (
                    evasion_model is not None
                ), "Evasion setting requires re-train models"
                result[EVASION] = 0  # total number of success evasion attack
            if poision:
                result[POSION] = 0  # total number of success poison

            total = 0

            for node in target_nodes:  # for each target node
                truth_label = self.data.y[node].item()
                self.labels.append(truth_label)

                self.adversarial.reset()
                if use_epsilon:
                    node_budget = int(
                        round(budget * self.degrees[node])
                    )  # compute budget with respect to target node degree
                else:
                    node_budget = budget

                self.adversarial.set_max_perturbations(node_budget)
                start_attack = timeit.default_timer()

                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    self.adversarial.attack(
                        node,
                        num_budgets=node_budget,
                        structure_attack=structure_attack,
                        feature_attack=feature_attack,
                    )
                    gpu_mem_alloc = (
                        torch.cuda.max_memory_allocated() / 1000000
                        if torch.cuda.is_available()
                        else 0
                    )
                    self.gpu_mem_usage_list.append(gpu_mem_alloc)
                except Exception as e:
                    logger.error(f"Fail to perform attack due to:{e}")
                    duration = timeit.default_timer() - start_attack
                    total += duration
                    counter += 1
                    continue
                duration = timeit.default_timer() - start_attack
                total += duration

                if evasion:
                    # Get model prediction after attack for evasion scenario
                    pred_evasion = evasion_model.get_model_prediction(
                        self.adversarial.data()
                    )[node].item()
                    self.evasion_preds.append(pred_evasion)
                    if pred_evasion != truth_label:
                        result[EVASION] += 1
                        self.node_miss_classified_evasion.append(node)

                if poision:
                    # Get model prediction after attack for poison scenario
                    victim_model = ModelSupervisor(
                        self.adversarial.data(),
                        device=self.device,
                        seed=seed,
                        use_purification=self.use_purification,
                        purification_config=self.purification_config,
                        **self.victim_configs,
                    )

                    victim_model.train_model(
                        self.idx_train, self.idx_val, self.idx_test
                    )
                    pred_poision = victim_model.get_model_prediction(
                        self.adversarial.data()
                    )[node].item()
                    self.poision_preds.append(pred_poision)

                    if pred_poision != truth_label:
                        result[POSION] += 1
                        self.node_miss_classified_poison.append(node)

                counter += 1
                print(
                    f"\rINFO : finished {counter}/{len(target_nodes) * len(bucket_target_nodes)}",
                    end="",
                )

            result["time"] = total

            return_tuple += (result,)
        return return_tuple

    def save_result(
        self,
        adversarial_name: str,
        seed: int,
        dataset_name: str,
        model_name: str,
        budget: int,
        purification_name: str = None,
        config_setting: str = "best_config",
    ) -> None:
        """
        Saves prediction results to a JSON file organized by configuration, model, dataset, and budget.

        Args:
            - adversarial_name: Name of the adversarial attack method used
            - seed: Random seed used during the experiment
            - dataset_name: Name of the dataset the model was evaluated on
            - model_name: Name of the model used for predictions
            - budget: Budget constraint used during the attack
            - purification_name: Name of the purification method applied (if any). If None, no purification is used
            - config_setting: Configuration setting label used to organize saved results directory (default: "best_config")
        """
        if purification_name is None:
            saved_path = "{}{}/{}/{}/budget{}/".format(
                PATH_CACHED_PRED, config_setting, model_name, dataset_name, budget
            )
        else:
            saved_path = "{}{}/{}_{}/{}/budget{}/".format(
                PATH_CACHED_PRED,
                config_setting,
                model_name,
                purification_name,
                dataset_name,
                budget,
            )

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        saved_file = "{}{}_{}.json".format(saved_path, adversarial_name, seed)
        saved_dict = {
            "labels": self.labels,
            "evasion_preds": self.evasion_preds,
            "poision_preds": self.poision_preds,
        }
        with open(saved_file, "w") as json_file:
            json.dump(saved_dict, json_file, indent=4)

    def get_gpu_mem_usage(self) -> None:
        """
        Returns the mean and standard deviation of GPU memory usage across recorded measurements.

        Args:
            - None

        Returns:
            - Tuple of (mean, stdev) GPU memory usage from gpu_mem_usage_list

        Raises:
            - AssertionError: If gpu_mem_usage_list is empty (i.e., no attack methods have been run)
        """
        assert (
            len(self.gpu_mem_usage_list) > 0
        ), "Please run attack methods before getting GPU Memory Usage"
        return stats.mean(self.gpu_mem_usage_list), stats.stdev(self.gpu_mem_usage_list)

    def get_missclassified_nodes(self) -> None:
        """
        Returns the misclassified nodes from evasion and poison attacks.

        Args:
            - None

        Returns:
            - Tuple of (node_miss_classified_evasion, node_miss_classified_poison)
        """
        return self.node_miss_classified_evasion, self.node_miss_classified_poison
