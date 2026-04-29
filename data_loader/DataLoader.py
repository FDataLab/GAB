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


class DataLoader:
    """
    Loads graph datasets and returns them in PyTorch Geometric (PyG) format.

    Supports datasets from three collections:
        - Homophily dataset: citeseer, citeseer_full, cora, cora_ml, cora_full, amazon_cs,
          amazon_photo, coauthor_cs, coauthor_phy, polblogs, karate_club, pubmed, flickr,
          blogcatalog, dblp, acm, uai, pdn
        - Heterophily dataset: Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions
        - Wiki network: chameleon, crocodile, squirrel
    """

    @abstractmethod
    def load(dataset: str) -> Data:
        """
        Load dataset.
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

        else:
            raise Exception("Unsupport datasets")

        data.name = name
        return data, default_split


if __name__ == "__main__":
    ogb_dataset, _ = DataLoader.load("ogbn-arxiv")
    print(ogb_dataset)
