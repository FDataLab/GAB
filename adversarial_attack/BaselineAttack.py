import copy
import os
import random
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from greatx.attack.targeted import RandomAttack, TargetedAttacker
from utility.util import set_random

"""
    Simple heuristic baseline attacks

"""


class RandomDegreeAttack(RandomAttack):
    """
    Implementation of L1D-RND.
    This will either remove an existing edge or add a new edge.
    """

    def __init__(self, data, device="cpu", seed=None, name=None, r=0.5, **kwargs):
        super().__init__(data, device, seed, name, **kwargs)
        self.deg = (
            degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            .cpu()
            .numpy()
        )
        self.r = r

    def get_added_edge(self, influence_nodes):
        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(self.nodes_set - set(neighbors) - set([self.target, u]))

        attacker_nodes = random.sample(
            attacker_nodes, int(len(attacker_nodes) * self.r)
        )
        best_node = None
        best_degree = -1
        for node in attacker_nodes:
            if self.deg[node] > best_degree:
                best_degree = self.deg[node]
                best_node = node
        v = best_node

        if self.is_legal_edge(u, v) and not (v is None):
            return (u, v)
        return None

    def compute_norm(self, node):
        neighbors = self.adjacency_matrix[node].indices.tolist()
        neighbors.append(node)
        return torch.norm(self.ori_data.x[neighbors, :]).item()

    def get_removed_edge(self, influence_nodes):
        u = random.choice(influence_nodes)
        neighbors = self.adjacency_matrix[u].indices.tolist()
        attacker_nodes = list(set(neighbors) - set([self.target, u]))
        threshold = int(len(attacker_nodes) * self.r)
        if threshold > 5:
            attacker_nodes = random.sample(attacker_nodes, threshold)

        if len(attacker_nodes) == 0:
            return None

        best_node = None
        best_norm = float("-inf")
        for node in attacker_nodes:
            node_norm = self.compute_norm(node)
            if node_norm > best_norm:
                best_norm = node_norm
                best_node = node

        v = best_node

        if self.is_singleton_edge(u, v) and not (v is None):
            return None

        if self.is_legal_edge(u, v):
            return (u, v)
        else:
            return None


class RandomOnlyAddAttack(RandomDegreeAttack):
    """
    Implementation of L1D-RND.
    This will only add a new edge.
    """

    def attack(
        self,
        target,
        *,
        num_budgets=None,
        threshold=1,
        direct_attack=True,
        structure_attack=True,
        feature_attack=False,
        disable=False
    ):

        TargetedAttacker.attack(
            self=self,
            target=target,
            target_label=None,
            num_budgets=num_budgets,
            direct_attack=direct_attack,
            structure_attack=structure_attack,
            feature_attack=feature_attack,
        )

        if direct_attack:
            influence_nodes = [target]
        else:
            influence_nodes = self.adjacency_matrix[target].indices.tolist()

        num_chosen = 0

        with tqdm(
            total=self.num_budgets, desc="Peturbing graph...", disable=disable
        ) as pbar:
            while num_chosen < self.num_budgets:
                # randomly choose to add or remove edges
                if random.random() <= threshold:
                    delta = 1
                    print("add edge")
                    edge = self.get_added_edge(influence_nodes)
                else:
                    delta = -1
                    edge = self.get_removed_edge(influence_nodes)

                if edge is not None:
                    u, v = edge
                    if delta > 0:
                        self.add_edge(u, v, num_chosen)
                    else:
                        self.remove_edge(u, v, num_chosen)

                    num_chosen += 1
                    # print(num_chosen)
                    pbar.update(1)

        return self


class RandomOnlyRemoveAttack(RandomDegreeAttack):
    """
    Implementation of L1D-RND.
    This will only remove an existing new edge.
    """

    def attack(
        self,
        target,
        *,
        num_budgets=None,
        threshold=1,
        direct_attack=True,
        structure_attack=True,
        feature_attack=False,
        disable=False
    ):

        TargetedAttacker.attack(
            self=self,
            target=target,
            target_label=None,
            num_budgets=num_budgets,
            direct_attack=direct_attack,
            structure_attack=structure_attack,
            feature_attack=feature_attack,
        )

        if direct_attack:
            influence_nodes = [target]
        else:
            influence_nodes = self.adjacency_matrix[target].indices.tolist()

        num_chosen = 0

        with tqdm(
            total=self.num_budgets, desc="Peturbing graph...", disable=disable
        ) as pbar:
            while num_chosen < self.num_budgets:
                # randomly choose to add or remove edges
                if random.random() <= threshold:
                    delta = 1
                    edge = self.get_added_edge(influence_nodes)
                else:
                    delta = -1
                    edge = self.get_removed_edge(influence_nodes)

                if edge is not None:
                    u, v = edge
                    if delta > 0:
                        self.add_edge(u, v, num_chosen)
                    else:
                        self.remove_edge(u, v, num_chosen)

                    num_chosen += 1
                    pbar.update(1)

        return self


class DegreeAttack:
    def __init__(self, data: Data, device="cpu", seed=720):
        set_random(seed)
        self.ori_data = data
        self.device = device
        self.deg = (
            degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            .cpu()
            .numpy()
        )

    def set_max_perturbations(self, budget):
        self.n_perturbations = budget

    def reset(self):
        self.modified_edgelist = None

    def attack(
        self, target, num_budgets=None, structure_attack=True, feature_attack=False
    ):
        self.n_perturbations = num_budgets
        data = self.ori_data
        if structure_attack:
            deg = self.deg
            candidate_nodes = np.argsort(deg)[::-1]
            modified_edgelist = copy.deepcopy(data.edge_index).cpu()
            for i in range(self.n_perturbations):
                new_edge = torch.tensor([[target], [candidate_nodes[i]]])
                modified_edgelist = torch.cat((modified_edgelist, new_edge), dim=1)
            self.modified_edgelist = modified_edgelist

    def data(self):
        assert (
            self.modified_edgelist is not None
        ), "attack() method must be called before calling data()"
        modified_data = copy.deepcopy(self.ori_data)
        modified_data.edge_index = self.modified_edgelist
        return modified_data


class NormAttack:
    def __init__(self, data: Data, device="cpu", seed=720):
        set_random(seed)
        self.ori_data = data
        self.device = device

    def set_max_perturbations(self, budget):
        self.n_perturbations = budget

    @property
    def p(self):
        raise Exception("p for torch.norm is not yet defined")

    def reset(self):
        self.modified_edgelist = None

    def data(self):
        assert (
            self.modified_edgelist is not None
        ), "attack() method must be called before calling data()"
        modified_data = copy.deepcopy(self.ori_data)
        modified_data.edge_index = self.modified_edgelist
        return modified_data

    def attack(
        self, target, num_budgets=None, structure_attack=True, feature_attack=False
    ):
        self.n_perturbations = num_budgets
        data = self.ori_data
        x = data.x
        norm = torch.norm(x, p=self.p, dim=1).cpu().numpy()
        candidate_nodes = np.argsort(norm)[::-1]
        modified_edgelist = copy.deepcopy(data.edge_index).cpu()
        for i in range(self.n_perturbations):
            new_edge = torch.tensor([[target], [candidate_nodes[i]]])
            modified_edgelist = torch.cat((modified_edgelist, new_edge), dim=1)
        self.modified_edgelist = modified_edgelist


class L1NormAttack(NormAttack):
    @property
    def p(self):
        return 1


class L2NormAttack(NormAttack):
    @property
    def p(self):
        return 2


class FrobeniusNormAttack(NormAttack):
    @property
    def p(self):
        return "fro"


class NuclearNormAttack(NormAttack):
    @property
    def p(self):
        return "nuc"
