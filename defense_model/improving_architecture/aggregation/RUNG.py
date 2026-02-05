import torch
from torch import nn
from torch_geometric.utils import add_self_loops, degree, to_dense_adj


def add_loops(A):
    n = A.shape[-1]
    return A + torch.eye(n, device=A.device)

def pairwise_squared_euclidean(X, Y):
    '''
    Adapted from [are_gnn_robust](https://github.com/LoadingByte/are-gnn-defenses-robust)

    $$
    Z_{ij} = \sum_k (F_{ik} - F_{jk})^2 \
        = \sum_k F_{ik}^2 + F_{jk}^2 - 2  F_{ik}  F_{jk}, 
    $$
    where $\sum_k F_{ik}  F_{jk} = (F F^\top)_{ij}$
    The matmul is already implemented efficiently in torch
    '''

    squared_X_feat_norms = (X * X).sum(dim=-1)  # sxfn_i = <X_i|X_i>
    squared_Z_feat_norms = (Y * Y).sum(dim=-1)  # szfn_i = <Z_i|Z_i>
    pairwise_feat_dot_prods = X @ Y.transpose(-2, -1)  # pfdp_ij = <X_i|Z_j> # clever...
    return (-2 * pairwise_feat_dot_prods + squared_X_feat_norms[:, None] + squared_Z_feat_norms[None, :]).clamp_min(0)

def sym_norm(A):
    Dsq = A.sum(-1).sqrt()
    return A / Dsq / Dsq.unsqueeze(-1)


class MLP(nn.Module):

    def __init__(
            self,
            n_feat,
            n_class,
            hidden_dims,
            bias: bool = True,
            dropout: float = 0.5
    ):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias)
            for in_dim, out_dim in zip([n_feat] + hidden_dims, hidden_dims + [n_class])
        ])
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, X):
        for linear in self.linears[:-1]:
            X = self.dropout(nn.functional.relu(linear(X)))
        X = self.linears[-1](X)
        return X

def get_mcp_att_func(gamma, ep=0.01, soft=False, beta=None, **kwargs):
    # x / gamma - x ^ 2 / gamma ^ 2 / 2
    def att(w):
        w += ep
        z = w.sqrt() # convert w to l_21 to match mcp & scad formulation. check with continuity of attention func.

        if torch.where(z < ep)[0].shape != (0,):
            raise ValueError('w should be smaller than ep')

        high_idx = torch.where(z > gamma)
        z[z <= gamma] = 1 / (2 * (z[z <= gamma])) - 1 / (2 * gamma)
        z[high_idx] = 0
        return z
    
    if soft:
        assert beta is not None
    def soft_att(w):
        w += ep
        z = w.sqrt()
        # softmax(1 / 2z - 1 / 2gamma, 0)
        x = 1 / (2 * z) - 1 / (2 * gamma)
        weight = torch.exp(beta * x)
        
        assert (weight == weight).all(), 'nan in soft mcp'
        
        
        # # make sure x is positive xe^bx / (1 + e^bx) = (bx+1)(1+e^bx)-xbe^bx = 0
        # # bx + e^bx + 1 = 0
        # # bx + 1 + bx + b^2 x^2 / 2 ... + 1 = 0
        # # x ~= (-2b +- \sqrt{4b^2 - 4b^2}) / b^2 = -2 / b
        # bias = 2 / beta

        # x = (x + bias) * weight / (1 + weight)
        
        # Well that was stupid...
        x = torch.log(1 + weight) / beta
        return x
    
    return att if not soft else soft_att
    

class RUNG(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            out_dim: int, 
            hidden_dims, 
            gamma: float, 
            lam_hat: float, 
            quasi_newton=True, 
            eta=None, 
            prop_step=10, 
            dropout=0.5, 
    ):
        super().__init__()
        # MLP Settings (decoupled architecture: F = RUNG(MLP(A, F0)))
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # Graph Smoothing Settings
        # objective: \sumedge |fi - fj| + \sumnode \lambda |fi - fi0|
        self.lam_hat = lam_hat 
        # variable substitution: lam_hat = 1 / (1 + lam), s.t. lam_hat
        # is bounded in [0, 1]
        self.lam = 1 / lam_hat - 1 
        self.quasi_newton = quasi_newton
        self.prop_layer_num = prop_step
        self.w = get_mcp_att_func(gamma) # W = d_{y^2} \rho(y)
        self.eta = eta

        # Verify Parameter Validity
        assert 0 <= lam_hat <= 1, 'lam_hat should be in [0, 1]!'
        if quasi_newton:
            assert eta is None, 'no need to specify stepsize in QN-IRLS'
        else:
            assert 0 < eta, 'must use nonzero stepsize'
    

    
    def forward(self,F, edge_index,edge_weight):
        A = to_dense_adj(edge_index)[0, :,:]
        # decoupled architecture: F = RUNG(MLP(A, F0))
        F0 = self.mlp(F)

        # add self loop to graph to avoid zero degree
        A = add_loops(A)
        # record degree matrix
        D = A.sum(-1)
        D_sq = D.sqrt().unsqueeze(-1)
        # normalize A
        A_tilde = sym_norm(A)
        
        # record F0 for skip connection (teleportation in APPNP)
        F = F0

        for layer_number in range(self.prop_layer_num):
            # Z_{ij} = |f_i - f_j|_2^2
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            # W_{ij} = d_{y^2} \rho(y), y = |f_i - f_j|_2
            W = self.w(Z.sqrt())
            # diag terms in W set to zero: see Remark 2 in paper
            W[torch.arange(W.shape[0]), torch.arange(W.shape[0])] = 0
            # check W
            
            #if not (W == W).all():
            #        raise Exception('Nan occurs in W! Check rho and F.')
            W[torch.isnan(W)]=1
            
            if self.quasi_newton: # Quasi-Newton IRLS
                # approx Hessian
                Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
                # Unbiased Robust Aggregation: guaranteed convergence!
                F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
            
            else: # IRLS
                diag_q = torch.diag((W * A).sum(-1)) / D    
                # gradient of H_hat
                grad_smoothing = 2 * (diag_q - W * A_tilde) @ F
                grad_reg = 2 * (self.lam * F - self.lam) * F0
                F = F - self.eta * (grad_smoothing + grad_reg)

        return F
