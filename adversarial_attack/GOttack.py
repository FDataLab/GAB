
"""
Author: Anonymous Authors
Credits:
            - The OrbitAttack is implemented with reference to Nettack's source code,
            a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
            by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
            - Gottack is built on top of DeepRobust - A PyTorch Library for Adversarial
            Attacks and Defenses developed by Yaxin Li, Wei Jin, Han Xu and Jiliang Tang
"""
import warnings
import torch
from deeprobust.graph import utils
import numpy as np
import scipy.sparse as sp
from numba import jit
from greatx.nn.models.surrogate import Surrogate
from utility.util import set_random
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from copy import copy
import pandas as pd

class OrbitAttack:
    def __init__(self, data:Data , device='cpu',seed = 720,orbit_type = 1518):
        """
        Input:
            model: surrogate model
            orbit_dict: pandas dataframe of orbit type of all nodes in the network
            attack_type: orbit type of candidate nodes
            orbit_type: 3 available options: Orbit_type_I,Orbit_type_II and two_Orbit_type
            Gottack uses the same setting as default function's parameters
        """
        set_random(seed)
        assert isinstance(data, Data)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_weight is None

        self.ori_data = data
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.features = data.x
        self.adj :sp.csr_matrix = to_dense_adj(
            data.edge_index, max_num_nodes=data.num_nodes)[0]
        self.labels = data.y

        self.cooc_constraint = None

        self.nnodes = data.num_nodes
        self.device = device

        self.modified_adj = None
        self.modified_features = None
        orbit_dict = pd.read_csv(f"data/orbit/{data.name}_orbit_table.csv")
        self.matching_index = orbit_dict.index[orbit_dict["two_Orbit_type"] == orbit_type].tolist()

    

    def setup_surrogate(self,model):
        self.surrogate = model
        Surrogate.setup_surrogate(self, surrogate=self.surrogate, freeze=True)
        W = None
        for para in self.surrogate.parameters():
            if para.ndim == 1:
                warnings.warn("The surrogate model has `bias` term, "
                              "which is ignored and the model itself "
                              f"may not be a perfect choice for.")
                continue
            if W is None:
                W = para
            else:
                W = para @ W

        assert W is not None
        self.W = W.t().cpu().numpy()
        self.nclass = self.W.shape[-1]

    def freeze_surrogate(self) -> "Surrogate":
        """Freezie the parameters of the surrogate model.

        Returns
        -------
        Surrogate
            the class itself
        """
        for para in self.surrogate.parameters():
            para.requires_grad_(False)
        return self

    def set_max_perturbations(self,budget):
        self.n_perturbations = budget

        self.modified_adj = None
        self.modified_features = None
        self.cooc_constraint = None
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []

    

    def filter_potential_singletons(self, modified_adj):
        """Computes a mask for entries potentially leading to singleton nodes, i.e.
        one of the two nodes corresponding to the entry have degree 1 and there
        is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(self.nnodes, 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def get_linearized_weight(self):
        surrogate = self.surrogate
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()

    def attack(self, target,num_budgets=None,attack_structure =True,attack_features=False, direct=True, verbose=False, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : torch.Tensor or scipy.sparse.csr_matrix
            Origina (unperturbed) node feature matrix. Note that
            torch.Tensor will be automatically transformed into
            scipy.sparse.csr_matrix
        ori_adj : torch.Tensor or scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix. Note that
            torch.Tensor will be automatically transformed into
            scipy.sparse.csr_matrix
        labels :
            node labels
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers:
            number of influencer nodes when performing indirect attack.
            (setting `direct` to False). When `direct` is True, it would be ignored.
        verbose : bool
            whether to show verbose logs
        """

        if self.nnodes is None:
            self.nnodes = self.adj.shape[0]

        self.target_node = target
        self.n_perturbations = num_budgets
        self.attack_features = attack_features
        self.attack_structure = attack_structure

        if type(self.adj) is torch.Tensor:
            self.ori_adj = utils.to_scipy(self.adj).tolil()
            self.modified_adj = utils.to_scipy(self.adj).tolil()
            self.ori_features = utils.to_scipy(self.features).tolil()
            self.modified_features = utils.to_scipy(self.features).tolil()
        else:
            self.ori_adj = self.adj.tolil()
            self.modified_adj = self.adj.tolil()
            self.ori_features = self.features.tolil()
            self.modified_features = self.features.tolil()


        assert self.n_perturbations > 0, "need at least one perturbation"

        self.adj_norm = utils.normalize_adj(self.modified_adj)

        logits = (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W )[target]

        self.label_u = self.labels[target]
        label_target_onehot = np.eye(int(self.nclass))[self.labels[target]]
        best_wrong_class = (logits - 1000*label_target_onehot).argmax()
        surrogate_losses = [logits[self.labels[target]] - logits[best_wrong_class]]

        if verbose:
            print("##### Starting attack #####")
            print("##### Attack only using structure perturbations #####")
            print("##### Performing {} perturbations #####".format(num_budgets))


        self.potential_edges = (np.array([[target, value] for value in self.matching_index])).astype("int32")


        for _ in range(num_budgets):
            if verbose:
                print("##### ...{}/{} perturbations ... #####".format(_+1, num_budgets))


            filtered_edges_final =  self.potential_edges

            # Compute new entries in A_hat_square_uv
            a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final, target)
            # Compute the struct scores for each potential edge
            struct_scores = self.struct_score(a_hat_uv_new, self.modified_features @ self.W)

            best_edge_ix = struct_scores.argmin()
            best_edge_score = struct_scores.min()
            best_edge = filtered_edges_final[best_edge_ix]


            # perform edge perturbation
            self.modified_adj[tuple(best_edge)] = self.modified_adj[tuple(best_edge[::-1])] = 1 - self.modified_adj[tuple(best_edge)]
            self.adj_norm = utils.normalize_adj(self.modified_adj)

            self.structure_perturbations.append(tuple(best_edge))
            self.feature_perturbations.append(())
            surrogate_losses.append(best_edge_score)

        # return self.modified_adj, self.modified_features

    def get_attacker_nodes(self, n=5, add_additional_nodes = False):
        """Determine the influencer nodes to attack node i based on
        the weights W and the attributes X.
        """
        assert n < self.nnodes-1, "number of influencers cannot be >= number of nodes in the graph!"
        neighbors = self.ori_adj[self.target_node].nonzero()[1]
        assert self.target_node not in neighbors

        potential_edges = np.column_stack((np.tile(self.target_node, len(neighbors)),neighbors)).astype("int32")

        # The new A_hat_square_uv values that we would get if we removed the edge from u to each of the neighbors, respectively
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges, self.target_node)

        # XW = self.compute_XW()
        XW = self.modified_features @ self.W

        # compute the struct scores for all neighbors
        struct_scores = self.struct_score(a_hat_uv, XW)
        if len(neighbors) >= n:  # do we have enough neighbors for the number of desired influencers?
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:

            influencer_nodes = neighbors
            if add_additional_nodes:  # Add additional influencers by connecting them to u first.
                # Compute the set of possible additional influencers, i.e. all nodes except the ones
                # that are already connected to u.
                poss_add_infl = np.setdiff1d(np.setdiff1d(np.arange(self.nnodes),neighbors), self.target_node)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n-len(neighbors)
                possible_edges = np.column_stack((np.tile(self.target_node, n_possible_additional), poss_add_infl))

                # Compute the struct_scores for all possible additional influencers, and choose the one
                # with the best struct score.
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges, self.target_node)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers::]]

                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def compute_logits(self):
        return (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[self.target_node]

    def strongest_wrong_class(self, logits):
        label_u_onehot = np.eye(self.nclass)[self.label_u]
        return (logits - 1000*label_u_onehot).argmax()



    def gradient_wrt_x(self, label):
        # return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, label].T)
        return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, label].reshape(1, -1))

    def reset(self):
        """Reset Nettack
        """
        self.modified_adj = None
        self.modified_features = None
        self.cooc_constraint = None
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []


    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.

        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:,self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores

    def compute_new_a_hat_uv(self, potential_edges, target_node):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """

        edges = np.array(self.modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.modified_adj.sum(0).A1 + 1

        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees,
                                         potential_edges.astype(np.int32), target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape=[len(potential_edges), self.nnodes])

        return a_hat_uv

    def data(
        self,
        edge_ratio: float = 1.0,
        feat_ratio: float = 1.0,
        coalesce: bool = True,
        symmetric: bool = True,
    ) -> Data:
        """Get the attacked graph denoted by
        PyG-like data instance. Note that this method
        uses LRU cache for efficiency, the computation is
        only excuted at the first call if the input parameters
        were the same.

        Parameters
        ----------
        edge_ratio : float, optional
            the fraction of edge perturbations, i.e.,
            how many perturbed edges are used to
            construct the perturbed graph.
            by default 1.0
        feat_ratio : float, optional
            the fraction of feature perturbations, i.e.,
            how many perturbed features are used to
            construct the perturbed graph.
            by default 1.0
        coalesce : bool, optional
            whether to coalesce the output edges.
        symmetric : bool, optional
            whether the output graph is symmetric, by default True

        Example
        -------
        >>> # Get the perturbed graph, including
        >>> # edge flips and feature flips
        >>> attacker.data()

        >>> # Get the perturbed graph, with
        >>> # specifying edge_ratio
        >>> attacker.data(edge_ratio=0.5)

        >>> # Get the perturbed graph, with
        >>> # specifying feat_ratio
        >>> attacker.data(feat_ratio=0.5)

        Returns
        -------
        Data
            the attacked graph denoted by PyG-like data instance
        """

        data = copy(self.ori_data)

        coo_matrix = self.modified_adj.tocoo()
        # Extract row and column indices (edges)
        row = torch.tensor(coo_matrix.row, dtype=torch.long)
        col = torch.tensor(coo_matrix.col, dtype=torch.long)

        # Stack the indices to create the edge list
        edge_list = torch.stack([row, col], dim=0)
        data.edge_index = edge_list

        if self.attack_features:
            data.x = torch.tensor(self.modified_features,dtype=torch.float)

        return data
@jit(nopython=True)
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


@jit(nopython=True)
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    """
    N = degs.shape[0]

    twohop_u= twohop_ixs[twohop_ixs[:, 0] == u, 1]


    #result_array = potential_edges[:, 1]

    #twohop_u = np.array([i for i in range(0, 2110)])
    #np.union1d(result_array, twohop_u_1)

    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values

def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.
    """


    degs = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

def compute_alpha(n, S_d, d_min):
    """
    Approximate the alpha of a power law distribution.

    """

    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.
    """

    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

    return new_S_d, new_n


def compute_log_likelihood(n, alpha, S_d, d_min):
    """
    Compute log likelihood of the powerlaw fit.

    """

    return n * np.log(alpha) + n * alpha * np.log(d_min) - (alpha + 1) * S_d

def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff
