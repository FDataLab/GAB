
import sys
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
import timeit
from torch_geometric.data import Data
from typing import Union, Dict, Optional, List

    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from defense_model.utils import load_model,load_graph_purification
from utility.config import args
from evaluation.utils import load_opt,load_scheduler
from utility.util import set_random, logger
from torch_geometric.utils import degree

from static import *



class ModelSupervisor:
    """
    Manages the full lifecycle of a GNN model including initialization, training,
    and inference, with optional graph purification preprocessing.

    Args:
        - data: PyTorch Geometric Data object containing graph structure and node features
        - device: Device to run the model on, e.g. 'cpu' or 'cuda' (default: 'cpu')
        - seed: Random seed for reproducibility (default: 720)
        - train_early_stopping: If True, applies early stopping during training (default: True)
        - use_purification: If True, applies graph purification to the data before
          training (default: False)
        - purification_config: Dictionary of purification hyperparameters. Required if
          use_purification is True (default: None)
        - **model_args: Additional keyword arguments for model and optimizer configuration,
          including EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MODEL,
          OPTIMIZER, OPTIMIZER_STATE, SCHEDULER, and SCHEDULER_STATE

    Notes:
        - Degree distribution is computed from the graph and passed to the model loader
          for degree-aware models such as PNA
        - If purification is applied and the resulting edge_index is None, it is
          reconstructed from the sparse adjacency matrix adj_t
        - Optimizer and scheduler are initialized based on model_args configuration
    """
    def __init__(
        self,
        data:Data,
        device : str = "cpu",
        seed : int = 720,
        train_early_stopping : bool = True,
        use_purification : bool = False,
        purification_config : Optional[Dict[str,Union[int, str, float]]] = None,
        **model_args
    ) -> None:
        set_random(seed)
        deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        degree_dist = torch.bincount(deg)
        
        self.kwagrs = model_args
        self.data = data

        #Config from args
        self.device = device
        self.train_early_stopping = train_early_stopping

        #Configs from kwargs for hyper-parameters
        num_features = data.x.size(-1)
        num_classes = data.y.max().item() + 1
        self.epoch = model_args.get(EPOCHS)
        self.patience = model_args.get(PATIENCE)
        self.lr = model_args.get(LEARNING_RATE)
        self.weight_decay = float(model_args.get(WEIGHT_DECAY,0))
        self.batch_size = model_args.get(BATCH_SIZE)
        self.model_name = model_args.get(MODEL)
        self.model = load_model(num_features,num_classes,degree_dist,**model_args)
        opti_state = model_args.get(OPTIMIZER_STATE)
        
        if opti_state is not None:
            self.optimizer =  load_opt(model_args.get(OPTIMIZER))(self.model.parameters(),lr= self.lr,weight_decay=self.weight_decay,**opti_state)
        else:
            self.optimizer =  load_opt(model_args.get(OPTIMIZER))(self.model.parameters(),lr= self.lr,weight_decay=self.weight_decay)

        self.scheduler = model_args.get(SCHEDULER)
        if self.scheduler is not None:
            scheduler_state = model_args.get(SCHEDULER_STATE)
            self.scheduler = load_scheduler(self.scheduler)(self.optimizer,**scheduler_state)
        
        # Purification
        self.use_purification = use_purification
        self.purification = None
        self.purification_config = purification_config
        if use_purification:
            assert purification_config is not None, "In order to use purification, please define purification config"
            self.purification = load_graph_purification(device=self.device,**purification_config)
            data = self.purification(data = data, inplace= False)
            if data.edge_index is None:
                data.edge_index = data.adj_t.nonzero(as_tuple=False).t()
        
        #Data
        self.number_nodes = data.num_nodes
        self.edge_index_torch = data.edge_index
        self.edge_attr = data.edge_attr
        self.features_torch = data.x
        self.labels_torch = data.y
        
        
        self.to_device(self.device)
        self.output = None
        self.prediction = None


    def to_device(self,device : str = "cpu") -> None:
        """
        Moves the model and all associated graph tensors to the specified device.

        Args:
            - device: Target device to move tensors to, e.g. 'cpu' or 'cuda' (default: 'cpu')
        """
        self.model = self.model.to(device)
        self.edge_index_torch = self.edge_index_torch.to(device)
        self.features_torch = self.features_torch.to(device)
        self.labels_torch = self.labels_torch.to(device)
        if not self.edge_attr is None:
            self.edge_attr = self.edge_attr.to(device)

    
        
    def _train(self,idx_train : List[int]) -> float:
        """
        1 epoch of training.
        """
        self.model.train()
        self.optimizer.zero_grad()
        if self.model_name != "GRAND":
            if self.batch_size is not None:
                out = self.model( self.features_torch, self.edge_index_torch,edge_weight = self.edge_attr ,batch_size = int(self.batch_size) )
            else:
                out = self.model( self.features_torch, self.edge_index_torch,edge_weight = self.edge_attr)
            loss = F.cross_entropy(out[idx_train], self.labels_torch[idx_train])
        else:
            loss = self.model(self.features_torch, self.edge_index_torch,idx_train,self.labels_torch)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss)
    

    def train_model(
        self,
        idx_train : List,
        idx_val : List,
        idx_test : List,
        log_interval : int = 10
    ):
        """
        Trains the GNN model using the provided train/validation/test index splits.

        Args:
            - idx_train: Indices of training nodes
            - idx_val: Indices of validation nodes
            - idx_test: Indices of test nodes
            - log_interval: Number of epochs between progress log outputs (default: 10)
        """
        logger.info(">>Start traing {} on {} for node classification".format(args.model,args.dataset))
        logger.info("Number of nodes:{}".format(self.number_nodes))
        logger.info("Training node:{} , Validation node: {} , Test node: {}".format(len(idx_train),len(idx_val),len(idx_test)))
        self.model.train()

        best_val_acc = best_test_acc = 0
        best_val_loss = 100
        best_weight = None
        times = []
        patience = self.patience
        for epoch in range(1, self.epoch + 1):
            start_epoch = timeit.default_timer()
            loss = self._train(idx_train)

            #Validation and test
            epoch_output = self.get_model_output(self.data,force_no_use_purification=True) #force: to avoid keep using purification
            epoch_prediction = epoch_output.argmax(dim=-1)
            val_loss = F.cross_entropy(epoch_output[idx_val],self.labels_torch[idx_val])
            train_acc = self.get_prediction_accuracy(idx_train,self.data,epoch_prediction)
            test_acc =  self.get_prediction_accuracy(idx_test,self.data,epoch_prediction)
            val_acc =  self.get_prediction_accuracy(idx_val,self.data,epoch_prediction)


            if val_loss  < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

                best_weight = deepcopy(self.model.state_dict())
                self.output = epoch_output
                self.prediction = epoch_prediction
                patience = self.patience #Reset patience

            elif self.train_early_stopping and val_loss >= best_val_loss:
                patience -= 1
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

                best_weight = deepcopy(self.model.state_dict())
                self.output = epoch_output
                self.prediction = epoch_prediction
                patience = self.patience #Reset patience
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            delta_epoch = timeit.default_timer() - start_epoch
            times.append(delta_epoch)
            if epoch == 1 or epoch % log_interval == 0:
                logger.info("=="*30)
                logger.info("Epoch:{}, Loss:{:.4f}, Time:{:.3f}, GPU:{:.1f}MiB".format(epoch,loss,delta_epoch,gpu_mem_alloc))
                logger.info("Train: {:.3f}, Validation accuracy:{:.3f}, Test: {:.3f}".format(train_acc,val_acc,test_acc))

            if self.train_early_stopping and patience <=0 :
                logger.info(">> Early stopping")
                break
        
        self.model.load_state_dict(best_weight)
        logger.info(">> Done Training")
        total_time = sum(times)
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info("Best val:{:.3f}, Best test: {:.3f}".format(best_val_acc,best_test_acc))

    
    def get_model_output(self,data:Data,force_no_use_purification : bool = False) -> torch.Tensor:
        """
        Computes and returns the raw output logits of the model for the given graph data.

        Args:
            - data: PyTorch Geometric Data object containing graph structure and node features
            - force_no_use_purification: If True, skips purification even if use_purification
            is enabled (default: False)

        Returns:
            - Raw output logits as a torch.Tensor
        """
        self.model.eval()
        if self.use_purification and not force_no_use_purification:
            assert self.purification is not None, "In order to use purification, please define purification config when initialize the supervisor"
            purification = load_graph_purification(device=self.device,**self.purification_config)
            data = purification(data = data, inplace= False)
            if data.edge_index is None:
                data.edge_index = data.adj_t.nonzero(as_tuple=False).t()
        features_torch = data.x.to(self.device)
        edge_index_torch = data.edge_index.to(self.device)
        edge_attr = data.edge_attr
        if not edge_attr is None:
            edge_attr = edge_attr.to(self.device)
        with torch.set_grad_enabled(False):
            output = self.model( features_torch, edge_index_torch,edge_attr)
        return output

    def get_model_prediction(self,data:Data) -> torch.Tensor:
        """
        Returns the predicted class labels for each node in the given graph data.

        Args:
            - data: PyTorch Geometric Data object containing graph structure and node features

        Returns:
            - Predicted class indices as a torch.Tensor of shape (num_nodes,)
        """
        return self.get_model_output(data).argmax(dim=-1)

    def get_prediction_accuracy(self,mask,data:Data,prediction : bool = None) -> float:
        """
        Computes the prediction accuracy over a specified subset of nodes.

        Args:
            - mask: Indices or boolean mask identifying the subset of nodes to evaluate
            - data: PyTorch Geometric Data object containing graph structure and node features
            - prediction: Precomputed prediction tensor. If None, predictions are computed
            from the model (default: None)

        Returns:
            - Accuracy as a float representing the fraction of correctly predicted nodes
            within the specified mask
        """
        if prediction is None:
            prediction = self.get_model_prediction(data)
        return int((prediction[mask] == self.labels_torch[mask]).sum()) / len(mask)
    
    def load_model_state(self,model_cached_address : str) -> None:
        """
        Loads model weights from a cached state dictionary into the current model.

        Args:
            - model_cached_address: State dictionary containing the saved model weights
        """
        self.model.load_state_dict(model_cached_address)

    def save_model_state(self,model_cached_address: str) -> None:
        """
        Saves the current model weights to disk at the specified file path.

        Args:
            - model_cached_address: File path where the model state dictionary will be saved
        """
        torch.save(self.model.state_dict(),model_cached_address)
        
