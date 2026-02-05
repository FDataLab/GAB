import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation import Grid

from static import *

class PurificationGrid(Grid):
    def __init__(self,config_dict:dict) -> None:
        super().__init__(config_dict)
    
    def _get_single_grid_identifier(self):
        return PURIFICATION

    def _assert_config_dict(self):
        super()._assert_config_dict()

if __name__ == "__main__":
    configs = {
        "purification":"garnet",
        "k" :[1,23]
    }
    puri_grid = PurificationGrid(configs).init_grid()
    print(puri_grid)
    