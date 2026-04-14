import sys
import os
import timeit
import statistics as stats
import timeit
import json
from torch_geometric.utils import degree
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.ModelSupervisor import ModelSupervisor
from torch_geometric.data import Data
from greatx.attack.attacker import Attacker
from typing import List, Tuple
from static import *

class AdversarialSupervisor:
    def __init__(self,data:Data,
                 adversarial:Attacker,
                 idx_train,
                 idx_val,
                 idx_test,
                 device = "cpu",
                 use_purification = False,
                 purification_config = None,
                 **victim_configs
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
        self.degrees = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long).cpu().numpy()

        self.labels = []
        self.evasion_preds = []
        self.poision_preds = []
        self.gpu_mem_usage_list = []
        self.node_miss_classified_evasion = []
        self.node_miss_classified_poison = []
                

    def target_attack(self,bucket_target_nodes:Tuple[List[int],...],budget,evasion = False, poision = True,
                      evasion_model:ModelSupervisor= None,seed = 720,structure_attack = True, feature_attack = False,use_epsilon = False):
        assert evasion or poision, "Either evasion or posion setting"
        self.gpu_mem_usage_list = []
        return_tuple = ()
        counter = 0     

        for target_nodes in bucket_target_nodes:
            result = {
                'time' : 0
            }
            if evasion:
                assert evasion_model is not None, "Evasion setting requires re-train models"
                result[EVASION] = 0
            if poision:
                result[POSION] = 0
            
            total = 0

            for node in target_nodes:
                truth_label = self.data.y[node].item()
                self.labels.append(truth_label)

                self.adversarial.reset()
                if use_epsilon:
                    node_budget = int(round(budget * self.degrees[node]))
                else:
                    node_budget = budget

                self.adversarial.set_max_perturbations(node_budget)
                start_attack = timeit.default_timer()
                    
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    self.adversarial.attack(node,num_budgets = node_budget,structure_attack = structure_attack, feature_attack = feature_attack)
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    self.gpu_mem_usage_list.append(gpu_mem_alloc)
                except Exception as e:
                    duration = timeit.default_timer() - start_attack
                    total += duration
                    counter += 1
                    continue
                duration = timeit.default_timer() - start_attack
                total += duration

                if evasion:
                    pred_evasion = evasion_model.get_model_prediction(self.adversarial.data())[node].item()
                    self.evasion_preds.append(pred_evasion)
                    if pred_evasion != truth_label:
                        result[EVASION] += 1
                        self.node_miss_classified_evasion.append(node)
                
                if poision:
                    victim_model = ModelSupervisor(self.adversarial.data(), device= self.device,seed = seed,use_purification=self.use_purification,purification_config=self.purification_config,**self.victim_configs)

                    victim_model.train_model(self.idx_train,self.idx_val,self.idx_test)
                    pred_poision = victim_model.get_model_prediction(self.adversarial.data())[node].item()
                    self.poision_preds.append(pred_poision)

                    if pred_poision != truth_label:
                        result[POSION] += 1
                        self.node_miss_classified_poison.append(node)
                
                counter += 1
                print(f'\rINFO : finished {counter}/{len(target_nodes) * len(bucket_target_nodes)}', end='')
                

            result['time'] = total

            return_tuple += (result,)
        return return_tuple

    def save_result(self,adversarial_name, seed, dataset_name,model_name,budget,purification_name = None,config_setting="best_config" ):
        if purification_name is None:
            saved_path = "{}{}/{}/{}/budget{}/".format(PATH_CACHED_PRED,config_setting,model_name,dataset_name,budget)
        else:
            saved_path = "{}{}/{}_{}/{}/budget{}/".format(PATH_CACHED_PRED,config_setting,model_name,purification_name,dataset_name,budget)

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        
        saved_file = "{}{}_{}.json".format(saved_path,adversarial_name,seed)
        saved_dict = {
            'labels':self.labels,
            'evasion_preds': self.evasion_preds,
            'poision_preds':self.poision_preds
        }
        with open(saved_file, 'w') as json_file:
            json.dump(saved_dict, json_file,indent=4)


    def get_gpu_mem_usage(self):
        assert len(self.gpu_mem_usage_list) > 0, "Please run attack methods before getting GPU Memory Usage"
        return stats.mean(self.gpu_mem_usage_list), stats.stdev(self.gpu_mem_usage_list)
    
    def get_missclassified_nodes(self):
        return self.node_miss_classified_evasion, self.node_miss_classified_poison



