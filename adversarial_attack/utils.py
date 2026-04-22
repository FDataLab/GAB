import os
import sys

import networkx as nx
import numpy as np
from deeprobust.graph.utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculate_centrality(adj_matrix, direct=False):
    row_indices, col_indices = adj_matrix.nonzero()
    edgelists = list(zip(row_indices, col_indices))
    G = nx.Graph(edgelists)
    centrality_node = nx.degree_centrality(G)
    centrality_values = [(key, value) for key, value in centrality_node.items()]
    return centrality_values


def select_nodes_margin_centrality(adj_matrix, top_n=10):
    centrality_values = calculate_centrality(adj_matrix)
    centrality_values.sort(key=lambda x: x[1])
    highest = [node for node, _ in centrality_values[-top_n:]]
    lowest = [node for node, _ in centrality_values[:top_n]]
    return highest + lowest


def select_nodes_margin(target_runner, data, idx_test, label):
    """
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    """

    target_runner.model.eval()
    output = target_runner.predict(data)

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], label[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[:10]]
    low = [x for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10:-10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


def select_nodes_margin_v1(target_runner):
    """
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    """

    target_runner.model.eval()
    output = target_runner.get_model_output()

    margin_dict = {}
    for idx in target_runner.idx_test:
        margin = classification_margin(output[idx], target_runner.labels_torch[idx])
        if margin < 0:  # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[:10]]
    low = [x for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10:-10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other
