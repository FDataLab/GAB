from copy import deepcopy 
import os 
import sys
from typing import Dict, List, Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from static import *

class Grid:
    """
    Generates all possible hyperparameter configurations from a configuration dictionary
    using grid search expansion.

    Args:
        - config_dict: Dictionary mapping hyperparameter names to lists of possible values.
          Must contain at least one key-value pair.

    Methods:
        - init_grid: Expands config_dict into a flat list of all possible hyperparameter
          combinations and returns it
        - _update_config: Incrementally builds the list of configurations by combining
          existing configs with each possible value of a new hyperparameter
        - _get_single_grid_identifier: Abstract-like method to be overridden by subclasses;
          returns the key identifier for hyperparameters that should not be expanded as a list
        - _assert_config_dict: Validates that config_dict contains at least one key-value pair

    Raises:
        - AssertionError: If config_dict is empty or any hyperparameter has an empty list of values
    """
    def __init__(self,config_dict:dict) -> None:
        self.config_dict = config_dict

        self._assert_config_dict()
        self.all_configs = []

    def init_grid(self) -> List[Dict[str,Union[int, str, float]]]:
        """
        Expands the configuration dictionary into a list of all possible hyperparameter combinations.

        Returns:
            - List of dictionaries mapping hyperparameter names to their assigned values
            (int, str, or float), where each dictionary represents a unique combination
            across all keys in config_dict

        Raises:
            - AssertionError: If any hyperparameter has an empty list of candidate values

        Notes:
            - Keys matching the single grid identifier are treated as scalar values
            and wrapped in a list before expansion to prevent iteration over their contents
        """
        for key,values in self.config_dict.items():
            if key == self._get_single_grid_identifier():
                values = [values]
            assert len(values) > 0, "Need to define at least one value for hyper-parameter:{}".format(key)
            self.all_configs = self._update_config(self.all_configs,values,key)
        return self.all_configs

    def _update_config(
        self,
        all_configs:list,
        possible_values: list,
        hyperparameter: str
    ) -> List[Dict[str,Union[int, str, float]]]:
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


class ModelGrid(Grid):
    """
    Extends Grid to generate all possible GNN-based model hyperparameter configurations,
    including optimizer and optional scheduler settings.

    Args:
        - config_dict: Dictionary mapping hyperparameter names to lists of possible values.
          Must include MODEL, NUM_LAYERS, LEARNING_RATE, HIDDEN_UNITS, EPOCHS,
          AGGREGATION, and OPTIMIZER_STATE as required keys.

    Methods:
        - init_grid: Extends parent grid expansion to separately handle optimizer and
          scheduler configurations, returning the full list of model configurations

    Raises:
        - AssertionError: If any required hyperparameter key is missing or has no candidate values
        - Exception: If the specified optimizer or scheduler is unsupported
    """
    def __init__(self,config_dict:dict) -> None:
        super().__init__(config_dict)

    def _get_single_grid_identifier(self):
        return MODEL

    def _assert_config_dict(self):
        super()._assert_config_dict()
        assert MODEL in self.config_dict, "Model is not defined"

        assert_values = [NUM_LAYERS,LEARNING_RATE,HIDDEN_UNITS,EPOCHS,AGGREGATION,OPTIMIZER_STATE]
        for key in assert_values:
            assert key in self.config_dict and len(self.config_dict[key]) > 0, "{} must be at least 1 possible value".format(key)

    def init_grid(self) -> List[Dict[str,Union[int, str, float]]]:
        """
        Expands the model configuration dictionary into all possible hyperparameter combinations,
        including optimizer and optional scheduler configurations.

        Returns:
            - List of dictionaries where each dictionary represents a unique combination
            of model hyperparameters, optimizer settings, and scheduler settings (if provided)

        Notes:
            - OPTIMIZER_STATE and SCHEDULER_STATE are extracted from config_dict before
            parent grid expansion to handle them separately via dedicated update methods
            - Scheduler configurations are only applied if SCHEDULER_STATE is present in config_dict
        """
        optimizer_state = self.config_dict.pop(OPTIMIZER_STATE)
        scheduler_state = None
        if SCHEDULER_STATE in self.config_dict:
            scheduler_state = self.config_dict.pop(SCHEDULER_STATE)

        super().init_grid()
        self._update_config_optimizer(optimizer_state)
        if not scheduler_state is None:
            self._update_configs_scheduler(scheduler_state)
        return self.all_configs
    
    def _update_config_optimizer(self,optimizer_state):
        if optimizer_state.get(OPTIMIZER) == ADAM:
            optimizer = [ADAM]
            self.all_configs = self._update_config(self.all_configs,optimizer,OPTIMIZER)
            opti_state_configs = []
            for key in optimizer_state:
                if key != OPTIMIZER:
                    schedule_state_configs = self._update_config(schedule_state_configs,optimizer_state.get(key),key)
            if len(opti_state_configs) > 0:
                self.all_configs = self._update_config(self.all_configs,opti_state_configs,OPTIMIZER_STATE)

        else:
            raise Exception("{} is not supported".format(optimizer_state.get(OPTIMIZER)))

        
    def _update_configs_scheduler(self,scheduler_state):
        if scheduler_state.get(SCHEDULER) == STEP_LR:
            scheduler_class = [STEP_LR]
            self.all_configs = self._update_config(self.all_configs,scheduler_class,SCHEDULER)
            schedule_state_configs = []
            for key in scheduler_state:
                if key != SCHEDULER:
                    schedule_state_configs = self._update_config(schedule_state_configs,scheduler_state.get(key),key)
            
            if len(schedule_state_configs) >0 :
                self.all_configs = self._update_config(self.all_configs,schedule_state_configs,SCHEDULER_STATE)
        else:
            raise Exception("Scheduler {} is not supported".format(scheduler_state.get(SCHEDULER)))
        

class PurificationGrid(Grid):
    """
    Extends Grid to generate all possible purification hyperparameter configurations.

    Args:
        - config_dict: Dictionary mapping purification hyperparameter names to lists
          of possible values. Must contain at least one key-value pair.
    """
    def __init__(self,config_dict:dict) -> None:
        super().__init__(config_dict)
    
    def _get_single_grid_identifier(self):
        return PURIFICATION

    def _assert_config_dict(self):
        super()._assert_config_dict()