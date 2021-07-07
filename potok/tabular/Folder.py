from typing import List
from sklearn.model_selection import KFold

from ..core import Operator, DataDict, DataLayer


class Folder(Operator):
    def __init__(self, n_folds: int = 5, seed: int = 4242, **kwargs):
        super().__init__(**kwargs)
        self.n_folds = n_folds
        self.seed = seed
        self.folds = None

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        indx = x['train'].index
        folder = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        # folds = DataLayer(*[DataUnit(train, valid) for train, valid in folder.split(indx)])
        folds = [DataDict(train, valid) for train, valid in folder.split(indx)]
        self.folds = folds
        return None

    def x_forward(self, x: DataDict) -> DataLayer:
        x2 = self.get_folds(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataLayer:
        y2 = self.get_folds(y)
        return y2

    def y_backward(self, y_frwd: DataLayer) -> DataDict:
        y = DataDict.combine(y_frwd)
        return y

    def get_folds(self, xy: DataDict) -> DataLayer:
        assert xy['valid'] is None, 'Currnetly Double Validation is not allowed.'
        if xy['train'] is not None:
            valid_xy =  DataDict(train=xy['train'], valid=xy['train'])
            folds = [valid_xy.get_by_index(indx) for indx in self.folds]
            folds = [fold.copy(**{'test': xy['test']}) for fold in folds]
        else:
            folds = [xy]*self.n_folds
        return DataLayer(*folds)
        # return folds
