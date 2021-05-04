from typing import List
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
        folds = DataLayer(*[DataUnit(train, valid) for train, valid in folder.split(indx)])
        self.folds = folds

    def get_folds(self, xy: DataUnit) -> DataLayer:
        assert xy['valid'] is None, 'Currnetly Double Validation is not allowed.'
        valid_xy =  DataUnit(train=xy['train'], valid=xy['train'])
        folds = [valid_xy.get_by_index(indx) for indx in self.folds] #can be parallelized
        folds = [fold.copy(**{'test': xy['test']}) for fold in folds] #can be parallelized
        return DataLayer(*folds)

    def combine_folds(self, datas: DataLayer) -> DataUnit:
        combined = DataUnit.combine(datas)
        return combined
