from sklearn.model_selection import (
    StratifiedShuffleSplit,
    ShuffleSplit,
)
import numpy as np
import sys
import os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Fold:
    """
    Stores train/validation/test index splits for a single fold.

    Args:
        - idx_train: Indices of training samples
        - idx_val: Indices of validation samples
        - idx_test: Indices of test samples
    """
    def __init__(self,idx_train,idx_val,idx_test) -> None:
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

class ISplitter:
    """
    Interface for dataset splitting strategies.
    Subclasses must implement the split method to define custom train/val/test splitting logic.

    Methods:
        - split(X, y): Splits data X and labels y into Fold objects
    """
    def split(self,X,y):
        pass

class Splitter(ISplitter):
    """
    Concrete implementation of ISplitter that splits data into train/validation/test folds
    using either stratified or standard shuffle splitting.

    Args:
        - stratified: If True, uses stratified splitting to preserve class distribution across folds
        - n_split: Number of folds to generate
        - train_ratio: Proportion of data to use for training
        - val_ratio: Proportion of data to use for validation
        - test_ratio: Proportion of data to use for testing
        - seed: Random seed for reproducibility

    Raises:
        - AssertionError: If train_ratio + val_ratio + test_ratio exceeds 1.0

    Methods:
        - _init_splitter: Initializes outer and inner splitters based on stratified flag
        - split(X, y): Splits data into n_split Fold objects each containing train/val/test indices

    Notes:
        - Outer splitter separates test set; inner splitter further divides remaining data into train/val
        - For stratified splits, labels y must be provided and match the length of X
    """
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

    def _init_splitter(self) -> None:
        if self.stratified:
            self.outer_splitter = StratifiedShuffleSplit(self.n_split,test_size= self.test_ratio,random_state=self.seed)
            self.inner_splitter = StratifiedShuffleSplit(1,test_size=self.val_ratio/(self.train_ratio + self.val_ratio),random_state=self.seed)
        else:
            self.outer_splitter = ShuffleSplit(self.n_split,test_size= self.test_ratio,random_state=self.seed)
            self.inner_splitter = ShuffleSplit(1,test_size=self.val_ratio,random_state=self.seed)

    def split(
        self,
        X:np.ndarray,
        y: np.ndarray = None
    ) -> List [Fold]:
        """
        Splits data into a list of Fold objects each containing train/validation/test indices.

        Args:
            - X: Input data array of shape (n_samples, n_features)
            - y: Label array of shape (n_samples,). Required for stratified splitting, ignored otherwise

        Returns:
            - List of Fold objects of length n_split, each containing idx_train, idx_val, and idx_test indices

        Raises:
            - AssertionError: If stratified=True and y is None or length of y does not match X
            - AssertionError: If the union of train/val/test indices does not cover all samples in X
            - AssertionError: If the number of generated folds does not equal n_split
        """
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
        
