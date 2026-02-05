from typing import Optional
from torch_geometric.data import Data

import torch

"""
Provides the interface for models training supervisor
=======
July 08 2024
"""

class IModelSupervisor:

    def train_model(self,idx_train, idx_val,idx_test,log_interval = 10):
        pass

    def get_model_output(self,data:Data):
        pass
    
    def get_model_prediction(self,data:Data):
        pass

    def to_device(device):
        pass
    
    def get_prediction_accuracy(self,mask,data:Data,prediction:Optional[torch.tensor] = None):
        pass

    


