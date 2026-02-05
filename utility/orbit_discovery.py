import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
import numpy as np
import networkx as nx
from orbit_table_generator import OrbitTableGenerator, generate_orbit_tables_from_count
import orca
from torch_geometric.utils import to_networkx

if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation import ModelSupervisor
    from data_loader import DataLoader
    from utility import ConfigHandler
    from static import *

    args.dataset = "squirrel"
    
    data,split = DataLoader.load(args.dataset)

    edge_index = data.edge_index
    G = nx.Graph()

    # Add edges to the graph
    uni_edge = set()
    # print(edge_index.shape[1])
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if not (src,dst) in uni_edge and src != dst:
            G.add_edge(src, dst)
            uni_edge.add((src,dst))
            uni_edge.add((dst,src))
    nx.write_edgelist(G, "graph.txt", data=False)

       

    print("Generating graph")
    # G = to_networkx(data, to_undirected=True)
    # G = to_networkx(data)
    print("Done generating graph")


    orbit_counts = orca.orbit_counts("node", 5, G)
    print("Done counting orbit")
    orbit_df = generate_orbit_tables_from_count(orbit_counts,sorted(list(G.nodes)))
    orbit_df.to_csv("squirrel_orbit_df.csv")
