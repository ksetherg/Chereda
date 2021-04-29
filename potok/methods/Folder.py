from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..core import DataUnit, DataLayer

class Folder:
    def __init__(self, n_folds=5, seed=None):
        self.n_folds = n_folds
        self.seed = seed
        self.folds = None

    def generate_folds(self, x: DataUnit, y: DataUnit):
        indx = x.index['train']
        folder = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        folds = {f'Fold_{i}': DataUnit(*idxs) for i, idxs in enumerate(folder.split(indx))}
        self.folds = folds
        self.indx = x.indxs

    def get_folds(self, xy: DataUnit) -> DataLayer:
        '''can be parallelized'''
        folds = [xy.get_by_index(indx) for name, indx in self.folds.items()] #get_by_index how does it work with index = None?
        """insert test data"""
        return DataLayer(*folds)

    def combine_folds(self, datas: DataLayer):
        return unfolded

    # @unpack_data(mode='all')
    # def unfold(self, folds_dict: dict, Xy):
    #     folds = list(folds_dict.values())
    #     unfolded = folds[0].__class__.combine(folds)
    #     unfolded = unfolded.reindex(Xy.index)
    #     return unfolded