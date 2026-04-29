from copy import copy

import numpy as np
import scipy.sparse as sp
import torch
from numba import njit
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


class JaccardPurification(BaseTransform):
    r"""
    Jaccard version from DeepRobust
    """

    def __init__(self, binary_feature=False, threshold=0.01):
        self.binary_feature = binary_feature
        self.threshold = threshold

    def __call__(self, data: Data, inplace: bool = False) -> Data:
        if not inplace:
            data = copy(data)

        adj_mtx = csr_matrix(to_dense_adj(data.edge_index).squeeze(0).cpu().numpy())
        features = csr_matrix(data.x.cpu().numpy())
        adj_mtx = self.drop_dissimilar_edges(features=features, adj=adj_mtx)
        adj_mtx = adj_mtx.tocoo()
        rows = adj_mtx.row  # Source nodes
        cols = adj_mtx.col  # Destination nodes

        # Step 3: Create edge list tensor in PyTorch
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

        data.edge_index = edge_index
        return data

    def drop_dissimilar_edges(self, features, adj, metric="similarity"):
        """Drop dissimilar edges.(Faster version using numba)"""
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)

        adj_triu = sp.triu(adj, format="csr")

        if sp.issparse(features):
            features = features.todense().A  # make it easier for njit processing

        if metric == "distance":
            removed_cnt = dropedge_dis(
                adj_triu.data,
                adj_triu.indptr,
                adj_triu.indices,
                features,
                threshold=self.threshold,
            )
        else:
            if self.binary_feature:
                removed_cnt = dropedge_jaccard(
                    adj_triu.data,
                    adj_triu.indptr,
                    adj_triu.indices,
                    features,
                    threshold=self.threshold,
                )
            else:
                removed_cnt = dropedge_cosine(
                    adj_triu.data,
                    adj_triu.indptr,
                    adj_triu.indices,
                    features,
                    threshold=self.threshold,
                )
        modified_adj = adj_triu + adj_triu.transpose()
        return modified_adj


@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a * b)
            J = (
                intersection
                * 1.0
                / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)
            )

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA) - 1):
        for i in range(iA[row], iA[row + 1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (
                np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8
            )

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt
