from typing import Tuple
from ..core import Operator, DataDict


class Validation(Operator):
    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
        self.index = None

    def x_forward(self, x: DataDict) -> DataDict:
        self.index = x.index
        x2 = self.folder.x_forward(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        y2 = self.folder.y_forward(y)
        return y2

    def y_backward(self, y_frwd: DataDict) -> DataDict:
        y_bck = self.folder.y_backward(y_frwd)
        y = DataDict(train=y_bck['valid'])
        units = y_bck.units
        units.remove('valid')
        y.__setstate__({unit: y_bck[unit] for unit in units})
        y = y.reindex(self.index)
        return y
    
    def _fit_(self, x: DataDict, y: DataDict) -> None:
        self.folder._fit_(x, y)
        return None
