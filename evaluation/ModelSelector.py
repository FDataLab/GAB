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
from typing import Union, Dict, Tuple

def memory_stats():
    print("GPU:",torch.cuda.memory_allocated()/1024**2)
    print("GPU cached",torch.cuda.memory_cached()/1024**2)


class ModelSelector:
    """
    Manages hyperparameter selection for GNN models by evaluating all configurations
    generated from a grid search and tracking their performance.

    Args:
        - config_dict: Dictionary mapping hyperparameter names to lists of candidate values,
          used to generate all possible model configurations via ModelGrid
        - idx_train: Indices of training nodes
        - idx_val: Indices of validation nodes
        - idx_test: Indices of test nodes
        - data: PyTorch Geometric Data object used for model training and evaluation
        - device: Device to run model training and evaluation on (default: 'cpu')

    Notes:
        - All configurations are generated at initialization via ModelGrid.init_grid()
        - Results are tracked per aggregation method in all_config_results_dict,
          and per configuration index in results
    """
    def __init__(self,
        config_dict:dict,
        idx_train:np.ndarray,
        idx_val :np.ndarray,
        idx_test:np.ndarray,
        data: Data,
        device : str = 'cpu'
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

    def process_model_selection(self,seed : int = 720) -> None:
        """
        Iterates over all generated configurations, evaluates each one, and records
        performance results grouped by aggregation type.

        Args:
            - seed: Random seed for reproducibility across configuration evaluations (default: 720)

        Notes:
            - Results for each configuration are stored in all_config_results_dict
            under the corresponding aggregation type key, along with the config index
            - Progress is printed to stdout showing the total number of configurations to evaluate
        """
        counter = 1
        print("Finding the best configuration over {} configurataions".format(len(self.all_configs)))
        for index,config in enumerate(self.all_configs):
            aggre_type = config.get(AGGREGATION)
            
            config_perf = self._run_model_on_config(config,seed)   
            self.all_config_results_dict[aggre_type].append({"config_idx": index,"perfomance": config_perf})
            counter +=1
        print("\n")

    def get_best_config(self) -> Tuple[Dict[str,Union[int, str, float]]]:
        """
        Identifies the best hyperparameter configuration across all message aggregation types
        and returns both per-aggregation and overall best configurations.

        Returns:
            - Tuple of:
                - best_config_all_aggrs: Dictionary mapping each aggregation type to its
                best performing configuration (config_idx and performance score)
                - best_config: The single best configuration across all aggregation types
                based on highest performance score
        """
        best_config_all_aggrs = {}
        for aggr in self.all_aggrs:
            best_config = max(self.all_config_results_dict[aggr],key=lambda x:x['perfomance'])
            best_config_all_aggrs[aggr] = best_config
        
        best_config = max(best_config_all_aggrs.values(),key=lambda x:x['perfomance'])
        return best_config_all_aggrs,best_config
    
    def _run_model_on_config(self,config : Dict[str,Union[int, str, float]],seed : int):
        set_random(seed)
        runner = ModelSupervisor(self.data,device=self.device,**config)
        runner.train_model(idx_train=self.idx_train,idx_val=self.idx_val,idx_test= self.idx_test)
        val_acc = runner.get_prediction_accuracy(self.idx_val,self.data)
        runner.to_device('cpu')
        del runner.model
        del runner.optimizer
        del runner
        torch.cuda.empty_cache()
        return val_acc
    
    def save_config(self,split_index: int,dataset_name: str) -> None:
        """
        Saves the best hyperparameter configurations to disk for each aggregation type
        and for the overall best configuration.

        Args:
            - split_index: Index of the data split associated with the configurations being saved
            - dataset_name: Name of the dataset associated with the configurations being saved

        Notes:
            - For each aggregation type, saves the best configuration under
            '{aggr}_best_config' via ConfigHandler
            - The overall best configuration across all aggregation types is saved
            separately under 'best_config' via ConfigHandler
            - Each saved configuration includes the train/val/test split indices,
            model hyperparameters, and validation performance score
        """
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

        


