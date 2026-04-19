import numpy as np
from deeprobust.graph.utils import *
import os 
import sys
import networkx as nx
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation import IModelSupervisor
from utility.util import set_random
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data




class NodeSelector:
    """
    Target node selector.
    """
    @staticmethod
    def _filter_node_minimum_degree(
        margin_dict : Dict,
        min_degree: int,
        data : Data
    ) -> Dict:
        """
        Filters a margin dictionary to retain only nodes with degree greater than a minimum threshold.

        Args:
            - margin_dict: Dictionary mapping node indices to their margin values
            - min_degree: Minimum degree threshold; nodes with degree <= this value are excluded
            - data: PyTorch Geometric Data object containing edge_index for graph structure

        Returns:
            - Filtered dictionary containing only nodes whose degree exceeds min_degree
        """
        adj = to_scipy_sparse_matrix(data.edge_index)
        degree_values = np.array(adj.sum(axis=1)).flatten().tolist()
        valid_node = [node for node,degree in enumerate(degree_values) if degree >min_degree ]
        valid_margin_dict = {}
        for node in margin_dict.keys():
            if node in valid_node:
                valid_margin_dict[node] = margin_dict[node]
        return valid_margin_dict
    
    @staticmethod
    def select_nodes_margin(
        target_runner:IModelSupervisor,
        data:Data,
        idx_test : List,
        seed : int = 720,
        num_rand : int = 20,
        min_degree : int = 0
    ) -> Dict[str,List]:
        """
        Selecting nodes as reported in Nettack paper.

        Args:
            - target_runner : Victim model tariner
            - data: dataset to attack
            - idx_test: list of nodes for testing
            - seed: random seed
            - num_rand: number of random target nodes
            - min_degree: minimum degree

        Return: List of target nodes for each category:
            (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
            (ii) the 10 nodes with lowest margin (but still correctly classified) and
            (iii) 20 more nodes randomly
        """

        set_random(seed)
        target_runner.model.eval()
        output = target_runner.get_model_output(data)
        
        margin_dict = {}
        for idx in idx_test:
            margin = classification_margin(output[idx], data.y[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        
        if min_degree > 0:
            margin_dict = NodeSelector._filter_node_minimum_degree(margin_dict,min_degree,data)
        sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
        assert len(sorted_margins) >= 20+ num_rand, "Not enough nodes"

        high = [x for x, y in sorted_margins[: 10]]
        low = [x for x, y in sorted_margins[-10: ]]
        other = [x for x, y in sorted_margins[10: -10]]
        other = np.random.choice(other, num_rand, replace=False).tolist()

        
        return {
            "high_margin":high,
            "low_margin":low,
            "other":other
        }
    
    @staticmethod
    def select_nodes_margin_degree(
        target_runner:IModelSupervisor,
        data:Data,
        idx_test : List,
        seed : int = 720,
        num_rand : int = 10,
        min_degree : int = 0
    ) -> Dict[str,List]:
        """
        Enhance target node selector from Nettack with node degree.

        Args:
            - target_runner : Victim model tariner
            - data: dataset to attack
            - idx_test: list of nodes for testing
            - seed: random seed
            - num_rand: number of random target nodes
            - min_degree: minimum degree

        Return: List of target nodes for each category:
            (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
            (ii) the 10 nodes with lowest margin (but still correctly classified) and
            (iii) 10 nodes with highest degree
            (iv) 10 nodes with lowest degree
            (v) 10 nodes with random
        """
        set_random(seed)
        target_runner.model.eval()
        output = target_runner.get_model_output(data)
        
        margin_dict = {}
        for idx in idx_test:
            margin = classification_margin(output[idx], data.y[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        if min_degree > 0:
            margin_dict = NodeSelector._filter_node_minimum_degree(margin_dict,min_degree,data)
        sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
        high = [x for x, y in sorted_margins[: 10]]
        low = [x for x, y in sorted_margins[-10: ]]
        other = [x for x, y in sorted_margins[10: -10]]
        adj = to_scipy_sparse_matrix(data.edge_index)
        degree_values = NodeSelector._calculate_degrees(adj,other)
        highest_degree,lowest_degree = NodeSelector._select_node_value_margin(degree_values)
        remaining = list((set(other) - set(highest_degree)) - set(lowest_degree))
       
        other = np.random.choice(remaining, num_rand, replace=False).tolist()
        return {
            "high_margin":high,
            "low_margin":low,
            "highest_degree":highest_degree,
            "lowest_degree":lowest_degree,
            "other":other
        }
    
    @staticmethod
    def _select_node_value_margin(tuples_values : List[tuple],top_n : int = 10):
        tuples_values.sort(key=lambda x: x[1])
        highest = [node for node,_ in tuples_values[-top_n:]]
        lowest = [node for node,_ in tuples_values[:top_n]]
        return highest,lowest
    
    @staticmethod
    def _calculate_degrees (adj_matrix,node_list : List):
        # For undirected graphs
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Sum of rows (or columns, as it's symmetric)
        degrees_tuple = [(node,degrees[node]) for node in node_list]
        return degrees_tuple

    
    
        