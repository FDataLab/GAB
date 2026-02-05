import numpy as np
from scipy.sparse import csr_matrix

def sparse_adjacency_to_edge_list(sparse_adj_matrix):
    """
    Converts a sparse adjacency matrix in CSR format to NumPy edge list.

    Parameters:
    sparse adjacency matrix (csr_matrix): A 2D NumPy array with shape (number of nodes, number of nodes),
                            where each column represents an edge.

    Returns:
    2D array: edge list in shape (2,number of edges).
    """

     # Find the indices of the non-zero elements in the matrix
    row_indices, col_indices = sparse_adj_matrix.nonzero()
    
    # Pair the indices to form the edge list
    edge_list = list(zip(row_indices, col_indices))

    edge_list =np.array(edge_list).T
    
    return edge_list

def edge_list_to_sparse_adjacency(edge_list: np.ndarray) -> csr_matrix:
    """
    Converts a NumPy edge list to a sparse adjacency matrix in CSR format.

    Parameters:
    edge_list (np.ndarray): A 2D NumPy array with shape (2, number of edges),
                            where each column represents an edge.

    Returns:
    csr_matrix: A sparse adjacency matrix in CSR format.
    """
    if edge_list.shape[0] != 2:
        raise ValueError("Edge list must have shape (2, number of edges)")

    # Extract row indices and column indices from the edge list
    row_indices, col_indices = edge_list

    # Create a data array of ones with the same length as the number of edges
    data = np.ones(edge_list.shape[1])

    # Determine the size of the adjacency matrix (assuming zero-based indexing)
    num_nodes = max(row_indices.max(), col_indices.max()) + 1

    # Create the sparse adjacency matrix in CSR format
    sparse_adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

    return sparse_adj_matrix

def edge_list_to_sysmetric_sparse_adjacency(edge_list: np.ndarray) -> csr_matrix:
    """
    Converts a NumPy edge list to a sparse adjacency matrix in CSR format.

    Parameters:
    edge_list (np.ndarray): A 2D NumPy array with shape (2, number of edges),
                            where each column represents an edge.

    Returns:
    csr_matrix: A sparse adjacency matrix in CSR format.
    """
    if edge_list.shape[0] != 2:
        raise ValueError("Edge list must have shape (2, number of edges)")

    row = edge_list[:, 0]
    col = edge_list[:, 1]
    data = np.ones(edge_list.shape[0])

    # Create symmetric matrix
    adj_matrix = csr_matrix((data, (row, col)), shape=(edge_list.max() + 1, edge_list.max() + 1))
    adj_matrix = adj_matrix + adj_matrix.T

    # Ensure no double counting of self-loops
    adj_matrix.setdiag(adj_matrix.diagonal() / 2)

    return adj_matrix