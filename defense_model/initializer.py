import torch.nn as nn
from typing import Optional

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.util import logger

def init_xavier_uniform(model):
    for m in model.modules():
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

def init_xavier_normal(model):
    for m in model.modules():
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

def init_kaiming_uniform(model):
    for m in model.modules():
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

def init_kaiming_normal(model):
    for m in model.modules():
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

def init_orthogonal(model):
    for m in model.modules():
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.orthogonal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

WEIGHT_INIT_REGISTRY = {
    "xavier_uniform":  init_xavier_uniform,
    "xavier_normal":   init_xavier_normal,
    "kaiming_uniform": init_kaiming_uniform,
    "kaiming_normal":  init_kaiming_normal,
    "orthogonal":      init_orthogonal,
}

def apply_weight_init(model, init_name: Optional[str]):
    """Apply a named init strategy to model. No-op if init_name is None or 'default'."""
    if init_name is None or init_name == "default":
        return
    init_fn = WEIGHT_INIT_REGISTRY.get(init_name)
    if init_fn is None:
        raise ValueError(
            f"Unknown weight init '{init_name}'. "
            f"Choose from: {list(WEIGHT_INIT_REGISTRY.keys())}"
        )
    init_fn(model)
    logger.info(f"Applied weight initialization: {init_name}")