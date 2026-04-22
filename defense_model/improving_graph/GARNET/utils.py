import random

import hnswlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.sparse import csr_matrix, diags, identity
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from torch_sparse import SparseTensor


def construct_adj(neighs):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:, 0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1, 1), k, axis=1)).reshape(
        -1,
    )
    col = neighs[:, 1:].reshape(
        -1,
    )
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    adj.data[:] = 1

    return adj


def normal_adj(adj):
    adj = SparseTensor.from_scipy(adj)
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float("inf")] = 0
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)

    return DAD.to_scipy(layout="csr")


def embedding_normalize(embedding, norm):
    if norm == "unit_vector":
        return normalize(embedding, axis=1)
    elif norm == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(embedding)
    elif norm == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(embedding)
    else:
        return embedding


def adj2laplacian(A):
    norm_adj = normal_adj(A)
    L = identity(norm_adj.shape[0]).multiply(1 + 1e-6) - norm_adj

    return L


def hnsw(features, k=10, ef=100, M=48):
    num_samples, dim = features.shape

    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, _ = p.knn_query(features, k + 1)
    adj = construct_adj(neighs)

    return adj
