import copy
from typing import List, Iterator, Tuple
from .Data import Data, DataUnit
from .ApplyToDataUnit import ApplyToDataUnit


class Node:
    def __init__(self, **kwargs):
        if bool(kwargs) and ('name' in kwargs):
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__
        
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        return
    
    def __str__(self) -> str:
        return self.name

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        return x, y

    def predict_forward(self, x : DataUnit) -> DataUnit:
        return x
    
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        return y_frwd

    @property
    def copy(self) -> 'Node':
        return copy.copy(self)


class Operator(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @ApplyToDataUnit()
    def x_forward(self, x: DataUnit) -> Data:
        return x

    @ApplyToDataUnit()
    def y_forward(self, y: DataUnit, x: DataUnit = None, x_frwd: DataUnit = None) -> DataUnit:
        return y

    @ApplyToDataUnit()
    def y_backward(self, y_frwd: DataUnit) -> DataUnit:
        return y_frwd

    def _fit_(self, x: DataUnit, y: DataUnit) -> None:
        return None

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        self._fit_(x, y)
        x_frwd = self.x_forward(x)
        y_frwd = self.y_forward(y, x, x_frwd)
        return x_frwd, y_frwd

    def predict_forward(self, x: DataUnit) -> DataUnit:
        x_frwd = self.x_forward(x)
        return x_frwd

    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.y_backward(y_frwd)
        return y