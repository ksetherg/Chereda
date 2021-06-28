import copy
import dill
from pathlib import Path

from typing import List, Iterator, Tuple
from .Data import Data, DataUnit
from .ApplyToDataUnit import ApplyToDataUnit

class Serializable:

    def _state_(self, state: dict):
        return state

    def _save_(self, prefix: Path = None) -> None:
        pass

    def _load_(self, prefix: Path = None) -> None:
        pass

    def save(self, prefix: Path = None) -> None:
        prefix.mkdir(parents=True, exist_ok=True)
        self._save_(prefix)
        file_name = prefix / (self.name + '.dill')
        with open(file_name, "wb") as dill_file:
            dill.dump(self, dill_file)

    def load(self, prefix: Path = None) -> None:
        file_name = prefix + self.name + '.dill'
        with open(file_name, "rb") as dill_file:
            instance = dill.load(dill_file)
            self.__setstate__(instance.__dict__)
        self._load_(prefix)
        # return instance
    
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state = self._state_(state)
        return state
    
    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        return


class Node(Serializable):
    def __init__(self, **kwargs):
        if bool(kwargs) and ('name' in kwargs):
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__
        
    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        return x, y

    def predict_forward(self, x : DataUnit) -> DataUnit:
        return x
    
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        return y_frwd

    @property
    def copy(self) -> 'Node':
        return copy.copy(self)

    def __str__(self) -> str:
        return self.name


class Operator(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def x_forward(self, x: DataUnit) -> Data:
        return x

    def y_forward(self, y: DataUnit, x: DataUnit = None, x_frwd: DataUnit = None) -> DataUnit:
        return y

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


class Regression(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _predict_(self, x: DataUnit) -> Data:
        return x

    def _fit_(self, x: DataUnit, y: DataUnit) -> None:
        return None

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        self._fit_(x, y)
        y_frwd = self._predict_(x)
        return x, y_frwd

    def predict_forward(self, x: DataUnit) -> DataUnit:
        y_frwd = self._predict_(x)
        return y_frwd
