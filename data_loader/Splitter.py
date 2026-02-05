from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Fold:
    def __init__(self,idx_train,idx_val,idx_test) -> None:
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

class ISplitter:
    def split(self,X,y):
        pass

class Splitter(ISplitter):
    def __init__(self,
                 stratified:bool,
                 n_split:int,
                 train_ratio:float,
                 val_ratio:float,
                 test_ratio:float,
                 seed:int
                 ) -> None:
        assert train_ratio + val_ratio + test_ratio <= 1, "Sum up ration is greater than 1: train ratio({}), validation ratio({}) and test ratio({})".format(str(train_ratio),
                                                                                                                                                            str(val_ratio),
                                                                                                                                                            str(test_ratio))
        self.stratified = stratified
        self.n_split = n_split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.outer_splitter = None
        self.inner_splitter = None
        self._init_splitter()

    def _init_splitter(self):
        if self.stratified:
            self.outer_splitter = StratifiedShuffleSplit(self.n_split,test_size= self.test_ratio,random_state=self.seed)
            self.inner_splitter = StratifiedShuffleSplit(1,test_size=self.val_ratio/(self.train_ratio + self.val_ratio),random_state=self.seed)
        else:
            self.outer_splitter = ShuffleSplit(self.n_split,test_size= self.test_ratio,random_state=self.seed)
            self.inner_splitter = ShuffleSplit(1,test_size=self.val_ratio,random_state=self.seed)

    def split(self,X:np.ndarray,
              y: np.ndarray = None) :
        if self.stratified:
            assert y is not None, "Labels y are missing for stratified split."
            assert X.shape[0] == len(y)
        outer_idx =np.array(range(X.shape[0]))

        folds = []
        for i, (train_index_outer, test_index) in enumerate(self.outer_splitter.split(outer_idx, y)):
            fold_test_index = outer_idx[test_index]
            

            inner_idx = outer_idx[train_index_outer]
            inner_y = y[train_index_outer]
            for j, (train_index_inner, val_index) in enumerate(self.inner_splitter.split(inner_idx, inner_y)):
                fold_train_index = inner_idx[train_index_inner]
                fold_val_index = inner_idx[val_index]
            
            folds.append(Fold(idx_train=fold_train_index,idx_val=fold_val_index,idx_test=fold_test_index))
            
           
        assert len(set(fold_train_index.tolist() + fold_val_index.tolist() + fold_test_index.tolist())) == X.shape[0]
        assert len(folds) == self.n_split
        return folds
        



if __name__ == '__main__':
    X = np.random.rand(10, 1433)
    y = np.random.randint(1, size=10)
    split = Splitter(True,2,0.2,0.2,0.6,0)
    split.split(X,y)
