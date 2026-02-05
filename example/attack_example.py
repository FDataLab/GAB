
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adversarial_attack import Nettack
from evaluation import ModelSupervisor
from data_loader import DataLoader
from utility import ConfigHandler
from static import *
from adversarial_attack import SurrogateLoader
from adversarial_attack import NodeSelector
from adversarial_attack import AdversarialSupervisor


# @TODO: This example needs to be updated.

if __name__=="__main__":
    from utility.config import args

    args.dataset = 'cora'
    args.seed = 722
    data,split = DataLoader.load(args.dataset)

    split_0_model = ConfigHandler.load_model_config("GCN","best_config","cora",0)
    split = split_0_model['split']
    best_config = split_0_model.get('model_config')
    print(best_config)
    idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_train']


    surrogate_supervisor = ModelSupervisor(data,args.device,**best_config)
    surrogate_supervisor.train_model(idx_train,idx_val,idx_test)
    target_bucket= NodeSelector.select_nodes_margin_degree(surrogate_supervisor,data,idx_test,seed=args.seed)

    adversarial_surrogate = ConfigHandler.load_model_config(data,idx_train,idx_val,device= args.device,seed=args.seed,model='GCN')
    nettack = Nettack(data,args.device,args.seed)
    nettack.setup_surrogate(adversarial_surrogate)

    adv_supervisor =  AdversarialSupervisor(data,nettack,idx_train,idx_val,idx_test,args.device,**best_config)
    high_result,low_result, highest_degree_result,lowest_degree_result,other_result = adv_supervisor.target_attack(target_bucket,5,evasion=True,poision=True, evasion_model=surrogate_supervisor,seed = 720)
    evasion_count = high_result[EVASION] + high_result[EVASION] + low_result[EVASION] + highest_degree_result[EVASION] + lowest_degree_result[EVASION] + other_result[EVASION]
    poision_count = high_result[POSION] + high_result[EVASION] + low_result[POSION] + highest_degree_result[POSION] + lowest_degree_result[POSION] + other_result[POSION]
    print(evasion_count,poision_count)

   
    