import os
import sys
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets.heterophilous_graph_dataset import (
    HeterophilousGraphDataset,
)

warnings.filterwarnings("ignore")

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from greatx.datasets import GraphDataset
from greatx.utils import split_nodes
from static import *
from data_loader.GraphLandLoader import PyGDataset  # noqa: E402  (GraphLand's own PyGDataset class)

DATASETS_greatX = {
    "citeseer",
    "citeseer_full",
    "cora",
    "cora_ml",
    "cora_full",
    "amazon_cs",
    "amazon_photo",
    "coauthor_cs",
    "coauthor_phy",
    "polblogs",
    "karate_club",
    "pubmed",
    "flickr",
    "blogcatalog",
    "dblp",
    "acm",
    "uai",
    "pdn",
}

PyG_HETEROPHILOUS_DATASET = {
    "Roman-empire",
    "Amazon-ratings",
    "Minesweeper",
    "Tolokers",
    "Questions",
}

PyG_WIKI_NETWORK_DATASET = {"chameleon", "crocodile", "squirrel"}

PyG_GRAPHLAND_DATASET = {
    "hm-categories",
    "pokec-regions",
    "web-topics",
    "tolokers-2",
    "city-reviews",
    "artnet-exp",
    "web-fraud",
    "hm-prices",
    "avazu-ctr",
    "city-roads-M",
    "city-roads-L",
    "twitch-views",
    "artnet-views",
    "web-traffic",
}

GRAPHLAND_NO_TEMPORAL_SPLIT = {
    "city-reviews",
    "city-roads-M",
    "city-roads-L",
    "web-traffic",
}


class DataLoader:
    """
    Loads graph datasets and returns them in PyTorch Geometric (PyG) format.

    Supports datasets from four collections:
        - Homophily dataset: citeseer, citeseer_full, cora, cora_ml, cora_full, amazon_cs,
          amazon_photo, coauthor_cs, coauthor_phy, polblogs, karate_club, pubmed, flickr,
          blogcatalog, dblp, acm, uai, pdn
        - Heterophily dataset: Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions
        - Wiki network: chameleon, crocodile, squirrel
        - GraphLand: hm-categories, pokec-regions, web-topics, tolokers-2, city-reviews,
          artnet-exp, web-fraud, hm-prices, avazu-ctr, city-roads-M, city-roads-L,
          twitch-views, artnet-views, web-traffic
    """

    @abstractmethod
    def load(dataset: str, graphland_split: str = "RL") -> Data:
        """
        Load dataset.

        Args:
            dataset: Name of the dataset to load.
            graphland_split: Split type for GraphLand datasets only
                ("RL", "RH", "TH", or "THI"). Ignored for all other
                dataset collections.
                - "RL"/"RH"/"TH" (transductive): returns (data, default_split)
                  where default_split is a (train_mask, val_mask, test_mask)
                  tuple, consistent with the rest of this loader.
                - "THI" (inductive): returns 3 graph snapshots instead of
                  one. Because this breaks the (data, default_split) return
                  contract, THI results are returned as a dict:
                  {"train": (train_data, None), "val": (val_data, None),
                   "test": (test_data, None)}. Access masks directly via
                  train_data.train_mask, val_data.val_mask, etc.
        """
        name = dataset
        if dataset in DATASETS_greatX:
            if not os.path.exists(PATH_DATASET):
                os.makedirs(PATH_DATASET)

            dataset = GraphDataset(
                root=PATH_DATASET,
                name=dataset,
                transform=T.LargestConnectedComponents(),
            )
            data = dataset[0]
            split = split_nodes(data.y, random_state=720)
            default_split = {
                IDX_TRAIN: split.train_nodes,
                IDX_VAL: split.val_nodes,
                IDX_TEST: split.test_nodes,
            }
        elif dataset in [
            "ogbn-products",
            "ogbn-proteins",
            "ogbn-arxiv",
            "ogbn-papers100M",
            "ogbn-mag",
        ]:
            data = PygNodePropPredDataset(
                name=dataset, root=PATH_DATASET, transform=T.ToSparseTensor()
            )[0]
            adj_t = data.adj_t.to_symmetric()
            ogb_dataset = PygNodePropPredDataset(
                name=dataset, root=PATH_DATASET, transform=T.ToUndirected()
            )
            default_split = ogb_dataset.get_idx_split()
            data = ogb_dataset[0]
            data.adj_t = adj_t
            data.y = data.y.squeeze()

        elif dataset in PyG_HETEROPHILOUS_DATASET:
            data = HeterophilousGraphDataset(name=dataset, root=PATH_DATASET)
            default_split = None

        elif dataset in PyG_WIKI_NETWORK_DATASET:
            gcn_geo_pre = False if dataset == "crocodile" else True
            data = WikipediaNetwork(
                name=dataset, root=PATH_DATASET, geom_gcn_preprocess=gcn_geo_pre
            )[0]
            default_split = (
                data.train_mask[:, 0],
                data.val_mask[:, 0],
                data.test_mask[:, 0],
            )

        elif dataset in PyG_GRAPHLAND_DATASET:
            if graphland_split in ("TH", "THI") and dataset in GRAPHLAND_NO_TEMPORAL_SPLIT:
                raise ValueError(
                    f"'{graphland_split}' split is not available for dataset "
                    f"'{dataset}'. Use 'RL' or 'RH' instead."
                )

            gl_dataset = PyGDataset(name=dataset, split=graphland_split)

            if graphland_split == "THI":
                train_data, val_data, test_data = gl_dataset
                train_data.name = name
                val_data.name = name
                test_data.name = name
                return (
                    {
                        "train": train_data,
                        "val": val_data,
                        "test": test_data,
                    },
                    None,
                )

            data = gl_dataset[0]
            default_split = (
                data.train_mask,
                data.val_mask,
                data.test_mask,
            )

        else:
            raise Exception("Unsupport datasets")

        data.name = name
        return data, default_split


if __name__ == "__main__":

    gl_dataset, gl_split = DataLoader.load("hm-categories", graphland_split="RH")
    print(gl_dataset)