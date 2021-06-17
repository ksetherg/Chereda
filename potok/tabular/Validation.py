from typing import Tuple
from ..core import Operator, DataLayer, DataUnit


class Validation(Operator):
    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder
        self.index = None

    def x_forward(self, x: DataUnit) -> DataLayer:
        x2 = self.folder.x_forward(x)
        return x2

    def y_forward(self, y: DataUnit, x: DataUnit = None, x_frwd: DataUnit = None) -> DataLayer:
        y2 = self.folder.y_forward(y)
        return y2

    def y_backward(self, y_frwd: DataLayer) -> DataUnit:
        y = self.folder.y_backward(y_frwd)
        y = DataUnit(train=y['valid'], test=y['test'])
        # y = y.reindex(self.index)
        return y
    
    def _fit_(self, x: DataUnit, y: DataUnit) -> None:
        print('Fitting with Validation')
        self.index = y.index
        self.folder._fit_(x, y)
        return None
