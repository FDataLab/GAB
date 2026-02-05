import logging
import math
import os
import socket
from typing import Callable, Optional, Tuple

import numba
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.cpp_extension import load
import torch_scatter
import torch_sparse
try:
    try:
        import kernels as custom_cuda_kernels
        if not hasattr(custom_cuda_kernels, 'topk'):
            raise ImportError()
    except ImportError:
        cache_dir = os.path.join('.', 'extension', socket.gethostname(), torch.__version__)
        os.makedirs(cache_dir, exist_ok=True)
        custom_cuda_kernels = load(name="kernels",
                                   sources=["kernels/csrc/custom.cpp", "kernels/csrc/custom_kernel.cu"],
                                   extra_cuda_cflags=['-lcusparse', '-l', 'cusparse'],
                                   build_directory=cache_dir)
except:  # noqa: E722
    logging.warn('Cuda kernels could not loaded -> no CUDA support!')


def soft_median(
    A: torch_sparse.SparseTensor,
    x: torch.Tensor,
    p=2,
    temperature=1.0,
    eps=1e-12,
    **kwargs
) -> torch.Tensor:
    """Soft Weighted Median.

    Parameters
    ----------
    A : torch_sparse.SparseTensor,
        Sparse [batch_size, n] tensor of the weighted/normalized adjacency matrix.
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    p : int, optional
        Norm for distance calculation
    temperature : float, optional
        Controlling the steepness of the softmax, by default 1.0.
    eps : float, optional
        Precision for softmax calculation.

    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    n, d = x.size()
    batch_size = A.size(0)

    row_index, col_index, edge_weights = A.coo()
    edge_index = torch.stack([row_index, col_index], dim=0)

    weight_sums = torch_scatter.scatter_add(edge_weights, row_index)

    with torch.no_grad():
        median_idx = custom_cuda_kernels.dimmedian_idx(x, edge_index, edge_weights, A.nnz(), batch_size)
        median_col_idx = torch.arange(d, device=x.device).view(1, -1).expand(batch_size, d)
    x_median = x[median_idx, median_col_idx]

    distances = torch.norm(x_median[row_index] - x[col_index], dim=1, p=p) / pow(d, 1 / p)

    soft_weights = torch_scatter.composite.scatter_softmax(-distances / temperature, row_index, dim=-1, eps=eps)
    weighted_values = soft_weights * edge_weights
    row_sum_weighted_values = torch_scatter.scatter_add(weighted_values, row_index)
    final_adj_weights = weighted_values / row_sum_weighted_values[row_index] * weight_sums[row_index]

    new_embeddings = torch_sparse.spmm(edge_index, final_adj_weights, batch_size, n, x)

    return new_embeddings

