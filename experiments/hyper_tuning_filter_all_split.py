import sys
import os
import warnings
import numpy as np
import timeit
# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.config import args
from evaluation import PurificationSelector
from data_loader import DataLoader
from utility import ConfigHandler

if __name__ == "__main__":
    hyper_grid_config = ConfigHandler.load_hyper_puri_grid(purification_type=args.purification,grid_name="hyper_grid")
    
    data,_ = DataLoader.load(args.dataset)
    
    for split_idx in range(0,5):
        if args.purification == "GTrans":
            hyper_grid_config['dataset'] = [args.dataset]
            hyper_grid_config['split_indx'] = [split_idx]
        split_model = ConfigHandler.load_model_config(args.model,"best_config",args.dataset,split_idx)
        split = split_model['split']
        model_config = split_model.get('model_config')
        idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']
        puriSelector = PurificationSelector(
            config_dict= hyper_grid_config,
            idx_train=idx_train,
            idx_val= idx_val,
            idx_test= idx_test,
            data=data,
            device=args.device,
            model_config=model_config
        )
        start = timeit.default_timer()
        puriSelector.process_purification_selection()
        end = timeit.default_timer()
        puriSelector.save_config(split_idx,args.dataset)
        print("Total time:{}".format(end -start))
        print(puriSelector.get_best_config())
    
     

    
   
    


