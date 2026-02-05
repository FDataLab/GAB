import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import ModelGrid
from evaluation import ModelSupervisor
from utility.util import set_random
from utility import ConfigHandler
from static import *
from torch_geometric.data import Data
import torch
import timeit

def memory_stats():
    print("GPU:",torch.cuda.memory_allocated()/1024**2)
    print("GPU cached",torch.cuda.memory_cached()/1024**2)


class ModelSelector:
    def __init__(self,
                 config_dict:dict,
                 idx_train:np.ndarray,
                 idx_val :np.ndarray,
                 idx_test:np.ndarray,
                 data: Data,
                 device = 'cpu'
                 ) -> None:
        self.model_name = config_dict.get("model")
        self.all_aggrs = config_dict.get(AGGREGATION)
        self.config_dict = config_dict
        self.grid_generator = ModelGrid(self.config_dict)
        self.all_configs = self.grid_generator.init_grid()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.data = data
        self.results = [0]*len(self.all_configs)

        self.all_config_results_dict = {}
        for agrr in self.all_aggrs:
            self.all_config_results_dict[agrr] = [] 
        self.device = device

    def process_model_selection(self,seed = 720):
        counter = 1
        print("Finding the best configuration over {} configurataions".format(len(self.all_configs)))
        for index,config in enumerate(self.all_configs):
            aggre_type = config.get(AGGREGATION)
            
            config_perf = self._run_model_on_config(config,seed)   
            self.all_config_results_dict[aggre_type].append({"config_idx": index,"perfomance": config_perf})
            # print(f'\rINFO : finished {counter}/{len(self.all_configs)}', end='')
            counter +=1
        print("\n")

    def get_best_config(self):
        best_config_all_aggrs = {}
        for aggr in self.all_aggrs:
            best_config = max(self.all_config_results_dict[aggr],key=lambda x:x['perfomance'])
            best_config_all_aggrs[aggr] = best_config
        
        best_config = max(best_config_all_aggrs.values(),key=lambda x:x['perfomance'])
        return best_config_all_aggrs,best_config
    
    def _run_model_on_config(self,config,seed):
        set_random(seed)
        runner = ModelSupervisor(self.data,device=self.device,**config)
        # try:
            # To avoid bad configuration prevent model from converging
        start_train = timeit.default_timer()
        runner.train_model(idx_train=self.idx_train,idx_val=self.idx_val,idx_test= self.idx_test)
        val_acc = runner.get_prediction_accuracy(self.idx_val,self.data)
        runner.to_device('cpu')
        del runner.model
        del runner.optimizer
        del runner
        torch.cuda.empty_cache()
        # except Exception as e:
        #     return -1
        return val_acc
    
    def save_config(self,split_index,dataset_name):
        best_config_all_aggrs,best_config = self.get_best_config()
        for aggr in self.all_aggrs:
            aggr_best_config = {
                "split" :{
                    IDX_TRAIN : self.idx_train.tolist(),
                    IDX_VAL : self.idx_val.tolist(),
                    IDX_TEST : self.idx_test.tolist()
                },
                
                "model_config":self.all_configs[best_config_all_aggrs[aggr]['config_idx']],
                "performance":best_config_all_aggrs[aggr]['perfomance']
            }
            ConfigHandler.save_model_config(aggr_best_config,"{}_best_config".format(aggr),model_type=self.model_name,data=dataset_name,split=split_index)

        best_config_dict = {
            "split" :{
                    IDX_TRAIN : self.idx_train.tolist(),
                    IDX_VAL : self.idx_val.tolist(),
                    IDX_TEST : self.idx_test.tolist()
                },
                
                "model_config":self.all_configs[best_config['config_idx']],
                "performance":best_config['perfomance']
        }    
        ConfigHandler.save_model_config(best_config_dict,"best_config",model_type=self.model_name,data=dataset_name,split=split_index)

        


