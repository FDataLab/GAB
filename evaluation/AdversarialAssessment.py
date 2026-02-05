import sys
import os
import statistics as stats
import pandas as pd
import timeit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from data_loader import Splitter
from greatx.attack.targeted import SGAttack,RandomAttack,FGAttack,PGDAttack
from static import *
from utility import ConfigHandler
from evaluation.ModelSupervisor import ModelSupervisor
from adversarial_attack import NodeSelector
from adversarial_attack import AdversarialSupervisor



class AdversarialAssessment:
    def __init__(self,
                 model:str,
                 dataset : str,
                 adversarial : str,
                 num_runs = 3,
                 num_split= 5,
                 base_seed = 720,
                 purification : str = None,
                 use_tune_model :bool = True,
                 use_degree_node_selection =True,
                 device = "cpu"
        ) -> None:
        
        self.model = model
        self.device = device
        self.dataset_name = dataset
        self.data,_ =  DataLoader.load(dataset)
        self.adversarial_name = adversarial
        self.num_runs = num_runs
        self.num_splits = num_split
        self.base_seed = base_seed
        self.use_tune_model = use_tune_model
        self.use_degree_node_selection = use_degree_node_selection
        self.purification_name = purification
        self.use_purification = False if purification is None else True
        self.node_selector = NodeSelector.select_nodes_margin_degree if use_degree_node_selection else NodeSelector.select_nodes_margin
        self.node_category = ['high_margin','low_margin','high_degree','low_degree','other'] if use_degree_node_selection else ['high_margin','low_margin','other']
        self.config_name = "best_config" if use_tune_model else "default"
        self.node_miss_classified_evasion = []
        self.node_miss_classified_poison = []
        self._log()

    def _log(self) -> str:
        print("=="*30)
        print("INFO: Adversarial Assessment setup")
        print(f"INFO: Victim model: {self.model}")
        print(f"INFO: Dataset: {self.dataset_name}")
        print(f"INFO: Adversarial Attack: {self.adversarial_name}")
        print(f"INFO: Number of runs: {self.num_runs}")
        print(f"INFO: Base seed: {self.base_seed}")
        print(f"INFO: Use tune model: {self.use_tune_model}")
        print(f"INFO: Use degree as extra criteria for selecting target nodes: {self.use_degree_node_selection}")
        print(f"INFO: Purification: {self.purification_name}")
        print(f"INFO: Use Purification: {self.use_purification}")
        print(f"INFO: Config setting: {self.config_name}")
        print(f"INFO: Number of splits: {self.num_splits}")
        print("=="*30)
        
        
    def _get_stat(self,results_list: list):
        return (stats.mean(results_list),stats.stdev(results_list))
    
    def _load_attack(self,adversarial_name:str):
        from adversarial_attack import Nettack, RandomDegreeAttack, OrbitAttack, PRBCDAttack
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
        
    def _prepare_dict(self):
        all_categories = self.node_category + ['total']
        return_dict = {}
        for category in all_categories:
            return_dict[category] = []

        return return_dict
    
    def _load_config_split(self,data):
        all_split_config = []
        if not self.use_tune_model:
            split_config = ConfigHandler.load_split_config(config_name="default")
            splitter = Splitter(**split_config)
            folds = splitter.split(data.x.numpy(),data.y.numpy())
        
        for split_idx in range(self.num_splits):
            config = ConfigHandler.load_model_config(self.model,self.config_name,self.dataset_name,split=split_idx)
            if not self.use_tune_model:
                config['split'] ={
                    IDX_TRAIN: folds[split_idx].idx_train,
                    IDX_VAL:folds[split_idx].idx_val,
                    IDX_TEST:folds[split_idx].idx_test
                }

            if self.use_purification:
                purification_config = ConfigHandler.load_purification_config(self.purification_name,self.model,"best_config",self.dataset_name,split=split_idx)['model_config']
                config['purification'] = purification_config

            all_split_config.append(config)
        return all_split_config
    
    def _get_surrogate_name(self,attack_method):
        if attack_method == SGA_NAME:
            return 'SGC'
        elif attack_method in [NETTACK_NAME,FGA_NAME,PGDATTACK_NAME]:
            return 'GCN_surrogate'
        elif attack_method is PRBCD_NAME:
            return 'GCN'
        return None


    def _get_gcn_surrogate(self,dataset,split_idx,seed,data,idx_train,idx_val,idx_test):
        split_model_surrogate = ConfigHandler.load_model_config("GCN","best_config",dataset,split_idx)
        configs_surrogate = split_model_surrogate.get('model_config')
        surrogate = ModelSupervisor(data,self.device,seed=seed,**configs_surrogate)
        surrogate.train_model(idx_train,idx_val,idx_test)
        return surrogate.model

    def _update_particular_config(self,attacker,data):               
        if self.adversarial_name == PRBCD_NAME:
            attacker.update_config(10000,200)

    def _target_node_dict_to_tuple(self,target_node_dict):
        r"""
        conver dict of target nodes return from target nodes selector to tuple of target nodes
        """
        target_node_tuple = tuple()
        for category in target_node_dict:
            target_node_tuple = target_node_tuple + (target_node_dict[category],)
        
        return target_node_tuple
    
    def _update_dict_results(self,result_tuple,num_rand,evasion_dict = None, poison_dict = None):
        time_total = 0
        evasion_total = 0
        poison_total = 0
        for idx,category in enumerate(self.node_category):
            result = result_tuple[idx]
            if evasion_dict is not None:
                if category == "other":
                    evasion_dict[category].append(result[EVASION]/num_rand)
                else:
                    evasion_dict[category].append(result[EVASION]/10)
                evasion_total += result[EVASION]
            if poison_dict is not None:
                if category == "other":
                    poison_dict[category].append(result[POSION]/num_rand)
                else:
                    poison_dict[category].append(result[POSION]/10)

                poison_total += result[POSION]

            time_total += result['time']

        if evasion_dict is not None:
            evasion_dict['total'].append(evasion_total/50)

        if poison_dict is not None:
            poison_dict['total'].append(poison_total/50)
        return time_total,evasion_dict,poison_dict
    

   
    def evaluate(self,evasion = True, poision = True, budgets_list = [1,2,3,4,5]):
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
                split = split_model['split']
                configs = split_model.get('model_config')
                purification_config = split_model.get('purification',None)
                idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']
                
                for idx_run in range(self.num_runs):
                    seed = self.base_seed + idx_run

                    # Select target nodes
                    node_selector = ModelSupervisor(self.data,self.device,seed=seed,use_purification=self.use_purification,
                                                    purification_config=purification_config,**configs)
                    start_defense = timeit.default_timer()
                    node_selector.train_model(idx_train,idx_val,idx_test)
                    time_defense_list.append(timeit.default_timer() - start_defense)
                    target_node_dict= self.node_selector(node_selector,self.data,idx_test,seed=seed,num_rand=num_rand)
                    target_node_tuple = self._target_node_dict_to_tuple(target_node_dict)
                    
                    # Load & set up adversarial attack
                    attacker = self._load_attack(self.adversarial_name)(self.data, device=self.device,seed=seed)
                    self._update_particular_config(attacker,self.data)
                    surrogate_name = self._get_surrogate_name(self.adversarial_name)

                    if surrogate_name is not None:
                        split_model_surrogate = ConfigHandler.load_model_config(surrogate_name,"best_config",self.dataset_name,split_idx)
                        configs_surrogate = split_model_surrogate.get('model_config')
                        surrogate = ModelSupervisor(self.data,self.device,seed=seed,**configs_surrogate)
                        surrogate.train_model(idx_train,idx_val,idx_test)
                        attacker.setup_surrogate(surrogate.model)
                        
                    # Perform adversarial attack
                    adversarial_supervisor = AdversarialSupervisor(self.data,attacker,idx_train,idx_val,idx_test,self.device,
                                                                   use_purification=self.use_purification, purification_config= purification_config,**configs)
                    result_bucket = adversarial_supervisor.target_attack(target_node_tuple,budget,evasion=True,poision=True, evasion_model=node_selector,seed = seed)
                    adversarial_supervisor.save_result(self.adversarial_name,seed,self.dataset_name,self.model,budget,self.purification_name)
                    time,evasion_dict,poision_dict = self._update_dict_results(result_bucket,num_rand,evasion_dict,poision_dict)
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
        
        if not os.path.exists("{}{}/{}/".format(PATH_RESULT,self.model,self.dataset_name)):
            os.makedirs("{}{}/{}/".format(PATH_RESULT,self.model,self.dataset_name))

        columns = ['budget'] + self.node_category +['total'] + ['time_attack'] + ['time_defense']

        df_poision = pd.DataFrame(rowlist_poision,columns=columns) if evasion else None
        df_evasion = pd.DataFrame(rowlist_evasion,columns=columns) if poision else None

        return df_evasion,df_poision


