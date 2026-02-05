import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import PurificationGrid
from evaluation import ModelSupervisor
from utility.util import set_random
from utility import ConfigHandler
from static import *
from torch_geometric.data import Data


class PurificationSelector:
    def __init__(self,
                 config_dict:dict,
                 idx_train:np.ndarray,
                 idx_val :np.ndarray,
                 idx_test:np.ndarray,
                 data: Data,
                 model_config,
                 device = 'cpu'
                 ) -> None:
        self.purification_name = config_dict.get(PURIFICATION)
        self.model_name = model_config.get(MODEL)
        self.config_dict = config_dict
        self.grid_generator = PurificationGrid(self.config_dict)
        self.all_configs = self.grid_generator.init_grid()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        # self.purification_name['idx_train'] = idx_train
        # self.purification_name['idx_val'] = idx_val
        self.data = data
        self.results = [0]*len(self.all_configs)
        self.model_config = model_config
        self.device = device

    def process_purification_selection(self,seed = 720):
        counter = 1
        print("Finding the best configuration over {} configurataions".format(len(self.all_configs)))
        for index,config in enumerate(self.all_configs):
            config_perf = self._run_model_on_config(self.model_config,config,seed)        
            self.results[index] = config_perf
            print(f'\rINFO : finished {counter}/{len(self.all_configs)}', end='')
            counter +=1
        print("\n")

    def get_best_config(self):
        return self.all_configs[np.argmax(self.results)],max(self.results)
    
    def _run_model_on_config(self,model_config,purification_config,seed):
        set_random(seed)
        
        runner = ModelSupervisor(data = self.data,device=self.device,purification_config=purification_config,use_purification=True,seed=seed,**model_config)
        runner.train_model(idx_train=self.idx_train,idx_val=self.idx_val,idx_test= self.idx_test)
        val_acc = runner.get_prediction_accuracy(self.idx_val,self.data)
        return val_acc
    
    def save_config(self,split_index,dataset_name):
        best_config,best_perf = self.get_best_config()
        
        best_config_dict = {
            "split" :{
                    IDX_TRAIN : self.idx_train,
                    IDX_VAL : self.idx_val,
                    IDX_TEST : self.idx_test
                },
            "model":self.model_name,    
            "model_config":best_config,
            "performance":best_perf
        }    
        ConfigHandler.save_purification_config(best_config_dict,"best_config",purification_type=self.purification_name,model_type=self.model_name,data=dataset_name,split=split_index)

        


