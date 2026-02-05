import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from static import *

class ConfigHandler:
    @staticmethod
    def load_model_config(model_type="GCN",config_name = "default",dataset = None,split = None):
        if dataset is None or config_name == "default":
            file_path = '{}{}/{}.yaml'.format(PATH_CONFIG_MODEL,model_type,config_name)
        else:
            file_path = '{}{}/{}/split_{}/{}.yaml'.format(PATH_CONFIG_MODEL,model_type,dataset,split,config_name)
        with open(file_path,'r') as file:
            data = yaml.safe_load(file)
        return data
    
    @staticmethod
    def save_model_config(config,config_name,model_type = "GCN",data="cora",split=None):
        file_path = '{}{}/{}/split_{}/'.format(PATH_CONFIG_MODEL,model_type,data,split)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open('{}{}.yaml'.format(file_path,config_name),'w') as file:
            yaml.dump(config,file)
        print("INFO: save model config successfully at {}{}.yaml".format(file_path,config_name))
    
    @staticmethod
    def load_purification_config(purification_type="GARNET",model_type= "GCN",config_name = "default",dataset = None,split = None):
        if dataset is None:
            file_path = '{}{}/{}.yaml'.format(PATH_CONFIG_PURIFICATION,purification_type,config_name)
        else:
            file_path = '{}{}/{}/{}/split_{}/{}.yaml'.format(PATH_CONFIG_PURIFICATION,purification_type,model_type,dataset,split,config_name)
        with open(file_path,'r') as file:
            data = yaml.safe_load(file)
        return data
    
    @staticmethod
    def save_purification_config(config,config_name,purification_type="GARNET",model_type="GCN",data="cora",split=None):
        file_path = '{}{}/{}/{}/split_{}/'.format(PATH_CONFIG_PURIFICATION,purification_type,model_type,data,split)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open('{}{}.yaml'.format(file_path,config_name),'w') as file:
            yaml.dump(config,file)
        print("INFO: save model config successfully at {}{}.yaml".format(file_path,config_name))

    @staticmethod
    def load_split_config(config_name = "default"):
        file_path = '{}{}.yaml'.format(PATH_CONFIG_SPLIT,config_name)
        with open(file_path,'r') as file:
            data = yaml.safe_load(file)
        return data
    
    @staticmethod
    def load_hyper_grid(model_type = "GCN",grid_name = "hyper_grid"):
        file_path = '{}{}/{}.yaml'.format(PATH_CONFIG_MODEL,model_type,grid_name)
        with open(file_path,'r') as file:
            grid = yaml.safe_load(file)
        return grid
    
    @staticmethod
    def load_hyper_puri_grid(purification_type = "GARNET_GCN",grid_name = "hyper_grid"):
        file_path = '{}{}/{}.yaml'.format(PATH_CONFIG_PURIFICATION,purification_type,grid_name)
        with open(file_path,'r') as file:
            grid = yaml.safe_load(file)
        return grid
    
if __name__ == "__main__":
    print(ConfigHandler.load_model_config())
    print(ConfigHandler.load_split_config())
    print(ConfigHandler.load_grid())