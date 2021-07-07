from typing import Tuple
from ..core import Operator, DataLayer, DataDict


class Validation(Operator):
    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
        self.index = None

    def x_forward(self, x: DataDict) -> DataLayer:
        self.index = x.index
        x2 = self.folder.x_forward(x)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataLayer:
        y2 = self.folder.y_forward(y)
        return y2

    def y_backward(self, y_frwd: DataLayer) -> DataDict:
        y = self.folder.y_backward(y_frwd)
        y = DataDict(train=y['valid'], test=y['test'])
        y = y.reindex(self.index)
        return y
    
    def _fit_(self, x: DataDict, y: DataDict) -> None:
        self.folder._fit_(x, y)
        return None
