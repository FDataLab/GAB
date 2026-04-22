import os
import sys
import timeit
import warnings

import numpy as np
import torch

# Hide all warnings
warnings.filterwarnings("ignore")

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import DataLoader, PyG_WIKI_NETWORK_DATASET, Splitter
from evaluation import ModelGrid, ModelSelector
from utility import ConfigHandler
from utility.config import args

if __name__ == "__main__":
    hyper_grid_config = ConfigHandler.load_hyper_grid(
        model_type=args.model, grid_name="hyper_grid"
    )
    grid_generator = ModelGrid(hyper_grid_config)
    all_configs = grid_generator.init_grid()
    data, _ = DataLoader.load(args.dataset)
    split_config = ConfigHandler.load_split_config(config_name="default")

    splitter = Splitter(**split_config)
    folds = splitter.split(data.x.numpy(), data.y.numpy())
    for split_idx in range(0, 5):
        fold = folds[split_idx]
        idx_train, idx_val, idx_test = fold.idx_train, fold.idx_val, fold.idx_test
        if args.dataset in PyG_WIKI_NETWORK_DATASET:
            mask_train, mask_val, mask_test = (
                data.train_mask[:, split_idx],
                data.val_mask[:, split_idx],
                data.test_mask[:, split_idx],
            )
            idx_train, idx_val, idx_test = (
                torch.nonzero(mask_train, as_tuple=True)[0],
                torch.nonzero(mask_val, as_tuple=True)[0],
                torch.nonzero(mask_test, as_tuple=True)[0],
            )
        hyper_grid_config = ConfigHandler.load_hyper_grid(
            model_type=args.model, grid_name="hyper_grid"
        )
        modelSelector = ModelSelector(
            config_dict=hyper_grid_config,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            data=data,
            device=args.device,
        )
        start = timeit.default_timer()
        modelSelector.process_model_selection()
        end = timeit.default_timer()
        modelSelector.save_config(split_idx, args.dataset)
        print("Total time:{}".format(end - start))
        print(modelSelector.get_best_config())
