from typing import List
from sklearn.model_selection import KFold

from ..core import Operator, DataDict


class Folder(Operator):
    def __init__(self, n_folds: int = 5, seed: int = 4242, **kwargs):
        super().__init__(**kwargs)
        self.n_folds = n_folds
        self.seed = seed
        self.folds = None

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        assert x['train'] is not None, 'Train required.'
        index = x['train'].index
        folder = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        folds = DataDict(**{f'Fold_{i}': DataDict(train=train, valid=valid)
                            for i, (train, valid) in enumerate(folder.split(index))})
        self.folds = folds
        return None

    def x_forward(self, x: DataDict) -> DataDict:
        x2 = self.get_folds(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        y2 = self.get_folds(y)
        return y2

    def y_backward(self, y_frwd: DataDict) -> DataDict:
        y = DataDict.combine(y_frwd)
        return y

    def get_folds(self, xy: DataDict) -> DataDict:
        units = xy.units
        if 'train' in units:
            units = [unit + f'_{i}' if unit == 'valid' else unit for i, unit in enumerate(units)]
            units.remove('train')
            valid_xy = DataDict(train=xy['train'], valid=xy['train'])
            folds = {k: valid_xy.get_by_index(v) for k, v in self.folds}
            [fold.__setstate__({unit: xy[unit] for unit in units}) for k, fold in folds.items()]
        else:
            folds = {f'Fold_{i}': xy for i in range(self.n_folds)}
        return DataDict(**folds)

