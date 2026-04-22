from typing import List, Optional

import torch
from torch_geometric.data import Data

"""
Provides the interface for models training supervisor
=======
July 08 2024
"""


class IModelSupervisor:
    """
    Interface defining the contract for managing GNN model training, inference, and evaluation.
    Subclasses must implement all methods to provide concrete model supervision logic.

    Methods:
        - train_model: Trains the model using the provided train/val/test index splits,
          logging progress at the specified interval
        - get_model_output: Returns the raw output logits of the model for the given data
        - get_model_prediction: Returns the predicted class labels for the given data
        - to_device: Moves the model and data to the specified device (e.g., 'cpu', 'cuda')
        - get_prediction_accuracy: Computes prediction accuracy over the specified mask,
          optionally using precomputed predictions
    """

    def train_model(
        self, idx_train: List, idx_val: List, idx_test: List, log_interval: int = 10
    ):
        pass

    def get_model_output(self, data: Data):
        pass

    def get_model_prediction(self, data: Data):
        pass

    def to_device(device: str):
        pass

    def get_prediction_accuracy(
        self, mask, data: Data, prediction: Optional[torch.tensor] = None
    ):
        pass
