from copy import deepcopy 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from static import *

class Grid:
    def __init__(self,config_dict:dict) -> None:
        self.config_dict = config_dict

        self._assert_config_dict()
        self.all_configs = []

    def init_grid(self):
        for key,values in self.config_dict.items():
            if key == self._get_single_grid_identifier():
                values = [values]
            assert len(values) > 0, "Need to define at least one value for hyper-parameter:{}".format(key)
            self.all_configs = self._update_config(self.all_configs,values,key)
        return self.all_configs

    def _update_config(self,
                    all_configs:list,
                    possible_values: list,
                    hyperparameter: str):
        new_configs_list = []
        if len(all_configs) == 0:
            for possible_value in possible_values:
                new_config = {hyperparameter:possible_value}
                new_configs_list.append(new_config)
        else:
            old_configs_list = deepcopy(all_configs)
            for possible_value in possible_values:
                for old_config in old_configs_list:
                    new_config = deepcopy(old_config)
                    new_config[hyperparameter] = possible_value
                    new_configs_list.append(new_config)

        return new_configs_list
        
    def _get_single_grid_identifier(self):
        pass

    def _assert_config_dict(self):
        assert len(self.config_dict) > 0, "Configs dictionary need at least one key-value pair"