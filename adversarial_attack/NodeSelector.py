import numpy as np
from deeprobust.graph.utils import *
import os 
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation import IModelSupervisor
from utility.util import set_random
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data




class NodeSelector:
    @staticmethod
    def _filter_node_minimum_degree(margin_dict,min_degree,data):
        adj = to_scipy_sparse_matrix(data.edge_index)
        degree_values = np.array(adj.sum(axis=1)).flatten().tolist()
        valid_node = [node for node,degree in enumerate(degree_values) if degree >min_degree ]
        valid_margin_dict = {}
        for node in margin_dict.keys():
            if node in valid_node:
                valid_margin_dict[node] = margin_dict[node]
        return valid_margin_dict
    
    @staticmethod
    def select_nodes_margin(target_runner:IModelSupervisor,data:Data,idx_test,seed = 720,num_rand= 20,min_degree = 0):
        '''
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 20 more nodes randomly
        '''

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
    def select_nodes_margin_degree(target_runner:IModelSupervisor,data:Data,idx_test,seed = 720,num_rand= 10,min_degree = 0):
        '''
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 10 nodes with highest degree
        (iv) 10 nodes with lowest degree
        (v) 10 nodes with random
        '''
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
        # other = np.random.choice(other, 20, replace=False).tolist()
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
    def select_node_margin_centrality(target_runner:IModelSupervisor):
        '''
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 10 nodes with highest centrality
        (iv) 10 nodes with lowest centrality
        '''

        target_runner.model.eval()
        output = target_runner.get_model_output()
        
        margin_dict = {}
        for idx in target_runner.idx_test:
            margin = classification_margin(output[idx], target_runner.labels_torch[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
        high = [x for x, y in sorted_margins[: 10]]
        low = [x for x, y in sorted_margins[-10: ]]
        other = [x for x, y in sorted_margins[10: -10]]
        # other = np.random.choice(other, 20, replace=False).tolist()
        central_values = NodeSelector._calculate_centrality(target_runner.adj,other)
        highest_central,lowest_central = NodeSelector._select_node_value_margin(central_values)

        return high,low, highest_central,lowest_central
    
    
    
    @staticmethod
    def _select_node_value_margin(tuples_values,top_n = 10):
        tuples_values.sort(key=lambda x: x[1])
        highest = [node for node,_ in tuples_values[-top_n:]]
        lowest = [node for node,_ in tuples_values[:top_n]]
        return highest,lowest
    
    @staticmethod
    def _calculate_degrees (adj_matrix,node_list):
        # For undirected graphs
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # Sum of rows (or columns, as it's symmetric)
        degrees_tuple = [(node,degrees[node]) for node in node_list]
        return degrees_tuple
    
    @staticmethod
    def _calculate_centrality (adj_matrix,node_list):
       row_indices,col_indices = adj_matrix.nonzero()
       edgelists = list(zip(row_indices,col_indices))
       G = nx.Graph(edgelists)
       centrality_node = nx.degree_centrality(G)
       centrality_values = [(node,centrality_node[node]) for node in node_list]
       return centrality_values
    
    @staticmethod
    def select_node_degree(target_runner:IModelSupervisor,top_n = 40):
        target_runner.model.eval()
        output = target_runner.get_model_output()
        correct_nodes = []
        for idx in target_runner.idx_test:
            margin = classification_margin(output[idx], target_runner.labels_torch[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            correct_nodes.append(idx)
        degree_values = NodeSelector._calculate_degrees(target_runner.adj,correct_nodes)
        highest_degree,lowest_degree = NodeSelector._select_node_value_margin(degree_values,top_n=top_n)
        
        return highest_degree,lowest_degree
    
    @staticmethod
    def select_node_centrality(target_runner:IModelSupervisor,top_n = 40):
        target_runner.model.eval()
        output = target_runner.get_model_output()
        correct_nodes = []
        for idx in target_runner.idx_test:
            margin = classification_margin(output[idx], target_runner.labels_torch[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            correct_nodes.append(idx)
        degree_values = NodeSelector._calculate_centrality(target_runner.adj,correct_nodes)
        highest_degree,lowest_degree = NodeSelector._select_node_value_margin(degree_values,top_n=top_n)
        
        return highest_degree,lowest_degree
        