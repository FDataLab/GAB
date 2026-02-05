from copy import deepcopy 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation import Grid


from static import *

class ModelGrid(Grid):
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

    def init_grid(self):
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




    
