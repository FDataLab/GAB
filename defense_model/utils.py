from torch_geometric.nn.models.basic_gnn import GraphSAGE,PNA,GAT,EdgeCNN,GIN
import sys
import os
import warnings
from torch_geometric.nn import aggr as torch_aggr
from greatx.nn.models import SGC,RobustGCN,GNNGUARD,ElasticGNN
from greatx.defense.purification import SVDPurification
import torch_geometric.transforms as T


# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from defense_model.improving_training.GRAND.grand import GRAND
from defense_model.improving_training import Noisy_GCN

from defense_model.improving_graph.GARNET.garnet import GarnetPurification
from defense_model.improving_graph import JaccardPurification
from defense_model.improving_architecture.aggregation.soft_median import SoftMedianAggregation
from defense_model.improving_architecture.conv.GCORN import GCORN
from defense_model.improving_architecture.aggregation.RUNG import RUNG

from utility.util import logger
from static import *


def load_model(in_channel,out_channel,degree_dist,**kwargs):

    model_name = kwargs.get(MODEL)
    nhid = kwargs.get(HIDDEN_UNITS,16)
    nlayers = kwargs.get(NUM_LAYERS,2)
    drop_out = kwargs.get(DROPOUT,0)
    aggr = kwargs.get(AGGREGATION,"sum")

    if aggr == "median":
        aggr = torch_aggr.MedianAggregation()
    if aggr == "softmedian":
        aggr = SoftMedianAggregation()
    
    assert model_name is not None, "Please define model"
    assert in_channel is not None, "Please defne in_channel"
    assert out_channel is not None, "Please define out_channel"
    
    if model_name == "GCN":
        from torch_geometric.nn.models.basic_gnn import GCN
        model = GCN(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == 'GIN':
          model = GIN(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == 'GSAGE':
         model = GraphSAGE(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == "PNA":
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        model = PNA(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggregators=aggregators, 
                    scalers=scalers,
                    deg= degree_dist)
    elif model_name == "GAT":
        model = GAT(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == "EdgeCNN":
        model = EdgeCNN(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == "SGC":
         model = SGC(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out)
    
    elif model_name == "GCN_surrogate":
        from greatx.nn.models import GCN
        model = GCN(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out)
    
    elif model_name == "RobustGCN":
        gamma = kwargs.get(GAMMA_ATTENTION,1)
        model = RobustGCN(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out,gamma = gamma)

    elif model_name == "GNNGuard":
        model = GNNGUARD(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out)
    
    elif model_name == "ElasticGNN":
        prop_step_k = kwargs.get(PROPA_STEP_K)
        lambda1 = kwargs.get(LAMBDA1)
        lambda2 = kwargs.get(LAMBDA2)
        model = ElasticGNN(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],
                           dropout = drop_out,K=prop_step_k,lambda1= lambda1,lambda2=lambda2)

    elif model_name == "GRAND":
        drop_node_rate = kwargs.get(DROP_NODE_RATE)
        temp = kwargs.get(TEMP)
        lambda_param = kwargs.get(LAMBDA)
        sample = kwargs.get(SAMPLE)

        model = GRAND(in_channel,nhid,nlayers,out_channel,
                      drop_out,drop_node_rate=drop_node_rate,temp=temp,lam=lambda_param,sample=sample)
        
    elif model_name == "GCORN":
        model = GCORN(in_channel,nhid,nlayers,out_channel,drop_out)

    elif model_name == "RUNG":
        gamma = kwargs.get('gamma')
        lambda_hat = kwargs.get('lambda_hat')
        model = RUNG(in_channel,
                     out_channel,
                     [nhid]*nlayers,
                     gamma, 
                     lambda_hat,
                     dropout = drop_out
                     )
    elif model_name == "NoisyGNN":
        model = Noisy_GCN(nfeat = in_channel, nhid = nhid , nclass = out_channel,dropout=drop_out)

    else:
        raise Exception('Unsupport or undefined model{}'.format(model_name))
    logger.info('Using models {} '.format(model_name))
    return model

def load_graph_purification(device = None,**kwargs):
    purification_name = kwargs.get(PURIFICATION)
    assert purification_name is not None

    if purification_name == "Jaccard":
        threshold = kwargs.get(THRESHOLD)
        purification = JaccardPurification(threshold)
    elif purification_name == "SVD":
        threshold = kwargs.get(THRESHOLD)
        top_k = kwargs.get(TOP_K)
        purification = SVDPurification(K=top_k,threshold=threshold)

    elif purification_name == "GARNET":
        r = kwargs.get(TOP_SINGULAR_R)
        k = kwargs.get(TOP_K)
        use_feature = bool(kwargs.get("use_feature"))
        weighted_knn = bool(kwargs.get("weighted_knn"))
        adj_norm = bool(kwargs.get('adj_norm'))

        gamma = kwargs.get(GAMMA)
        purification = GarnetPurification(r,k,gamma,use_feature=use_feature,
                                          weighted_knn=weighted_knn,adj_norm=adj_norm)
        
    else:
        raise Exception('Unsupport or undefined model{}'.format(purification_name))
    
    return purification


