import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from static import *


def normalize_adjacency_matrix(adj_matrix):
    # Symmetric normalization: A' = D^(-1/2) * A * D^(-1/2)
    rowsum = np.array(adj_matrix.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = coo_matrix(np.diag(d_inv_sqrt))
    
    return d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)

def sparse_adjacency_to_edge_list(sparse_adj_matrix):
    sparse_adj_matrix = normalize_adjacency_matrix(sparse_adj_matrix)
     # Find the indices of the non-zero elements in the matrix
    row_indices, col_indices = sparse_adj_matrix.nonzero()
    
    # Pair the indices to form the edge list
    edge_list = set(zip(row_indices, col_indices))

    edge_list =np.array(list(edge_list)).T
    
    return edge_list

def load_opt(opt_name):
    if opt_name == ADAM:
        from torch.optim import Adam
        return Adam
    else:
        raise Exception("{} is not supported".format(opt_name))
    
def load_scheduler(scheduler_name):
    if scheduler_name == STEP_LR:
        from torch.optim.lr_scheduler import StepLR
        return StepLR
    else:
       raise Exception("Scheduler {} is not supported".format(scheduler_name)) 


