"""
Script implementation of the GCORN model.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

from .matrix_ortho import *


def normalize_tensor_adj(adj):
    device = adj.device
    adj = sp.coo_matrix(adj.cpu())
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    coo = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)


class ConvClass(nn.Module):
    """
    This is an adaptation of the original implementation of the GCN to take into
    account the orthogonal projection (as explained in the paper).
    ---
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation: The activation to be used.
        iteration_val (int) : The number of projection iteration
        order_val (int) : The order of the projection
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        activation,
        beta_val=0.5,
        iteration_val=25,
        order_val=2,
    ):
        super(ConvClass, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim))

        self.activation = activation

        self.reset_parameters()

        self.beta_val = beta_val
        self.iters = iteration_val
        self.order = order_val

    def forward(self, x, adj):
        scaling = scale_values(self.weight.data).to(x.device)
        ortho_w = orthonormalize_weights(
            self.weight.t() / scaling,
            beta=self.beta_val,
            iters=self.iters,
            order=self.order,
        ).t()

        x = F.linear(x, ortho_w)

        return self.activation(torch.mm(adj, x))

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lin = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.lin(x)
        return x


class GCORN(torch.nn.Module):
    """
    Class implementation of an adapted 2-Layers GCN (GCORN) as explained in the
    paper. The model consists of two GCORN propagation with a final MLP layer
    as a ReadOut.
    ---
    in_channels (int) : The input dimension
    hidden_channels (int) : The hidden dimension of the embeddings to be used
    out_channels (int) : The output dimension (the number of classes to predict)
    """

    def __init__(
        self, in_channels, hidden_channels, nlayers, out_channels, drop_out=0.5
    ):
        super().__init__()

        self.activation = nn.ReLU()
        self.drop_out = drop_out
        self.convs = []
        self.convs.append(
            ConvClass(in_channels, hidden_channels, activation=self.activation)
        )

        for _ in range(1, nlayers):
            self.convs.append(
                ConvClass(hidden_channels, hidden_channels, activation=self.activation)
            )

        self.lin = MLPClassifier(hidden_channels, out_channels, self.activation)

    def to(self, device):
        self.lin.to(device)
        for conv in self.convs:
            conv.to(device)
        return self

    def forward(self, x, edge_index, edge_weight=None):

        adj_true = to_dense_adj(edge_index)[0, :, :]

        norm_adj = normalize_tensor_adj(adj_true)
        for conv in self.convs:
            x = conv(x, norm_adj)
            x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.lin(x)

        return F.log_softmax(x, dim=1)
