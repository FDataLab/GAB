
import torch
from torch import Tensor

from torch.nn.parameter import Parameter
from torch_geometric.nn.models import MLP
import os
import sys
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class GRAND(torch.nn.Module):
    def __init__(self,  in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        drop_node_rate :Optional[float]=0.5,
        order :Optional[int]=5,
        temp :Optional[float]=0.5,
        lam :Optional[float]=1.0,
        sample :Optional[int]=4,
        **kwargs,) -> None:

        super().__init__()


        #set up MLP
        self.mlp = MLP(in_channels=in_channels,hidden_channels=hidden_channels,out_channels= out_channels,num_layers=num_layers,dropout= dropout,
                       act= act, act_first=act_first,act_kwargs=act_kwargs,norm=norm,norm_kwargs=norm_kwargs,**kwargs)
        
        #set up hyper-paramerters for GRAND's algorithm
        self.scaler = StandardScaler()

        self.drop_node_rate = drop_node_rate
        self.lam = lam
        self.order = order
        self.temp = temp
        self.sample = sample

        self.train_mode = None

        
    
    def to(self,device="cpu"):
        self.mlp.to(device)
        return self
        

    def reset_parameters(self):
        self.mlp.reset_parameters()


    def parameters(self, recurse: bool = True):
        return self.mlp.parameters()

    def forward(
        self,
        x: Tensor,
        edge_index,
        idx_train = None,
        labels = None,
    ) -> Tensor:
        X = x
        K = self.sample
        A = to_dense_adj(edge_index).squeeze(0)

        if self.train_mode:
            X_list = []
            
            for k in range(K):
                X_list.append(self._rand_prop(X, training=self.train_mode,A=A))

            output_list = []
            for k in range(K):
                output_list.append(torch.log_softmax(self.mlp(X_list[k]), dim=-1))

            
            loss_train = 0.
            for k in range(K):
                loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
            
                
            loss_train = loss_train/K
            
            loss_consis = self._consis_loss(output_list)

            loss_train = loss_train + loss_consis 
            return loss_train
        else:
            X = x
            X = self._rand_prop(X, training=self.train_mode,A=A)
            output = self.mlp(X)
            output = torch.log_softmax(output, dim=-1)
            return output

    def train(self):
        self.train_mode = True
        self.mlp.train()

    def eval(self):
        self.train_mode = False
        self.mlp.eval()

    def _propagate(self,feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
        x = feature
        y = feature
        for i in range(order):
            x = torch.spmm(A, x).detach_()
            #print(y.add_(x))
            y.add_(x)
            
        return y.div_(order+1.0).detach_()
    

    def _rand_prop(self,features, training,A):
        n = features.shape[0]
        drop_rate = self.drop_node_rate
        drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
        
        if training:
                
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)

            features = masks.cuda() * features
                
        else:
                
            features = features * (1. - drop_rate)
        features = self._propagate(features, A, self.order)    
        return features
    
    def _consis_loss(self,logps):
        ps = [torch.exp(p) for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        #p2 = torch.exp(logp2)
        
        sharp_p = (torch.pow(avg_p, 1./self.temp) / torch.sum(torch.pow(avg_p, 1./self.temp), dim=1, keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))
        loss = loss/len(ps)
        return self.lam * loss