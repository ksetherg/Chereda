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
    
    def _fit_(self, x: Data, y: Data) -> (Data, Data):
        return x, y
        
    def fit(self, x: DataUnit, y: DataUnit) -> (DataUnit, DataUnit):
        return ApplyToDataUnit(self._fit_)(x, y)
        
    def _predict_forward_(self, x: Data) -> Data:
        return x
    
    def predict_forward(self, x : DataUnit) -> DataUnit:
        return ApplyToDataUnit(self._predict_forward_)(x)
            
    def _predict_backward_(self, y: Data) -> Data:
        return y
    
    def predict_backward(self, y: DataUnit) -> DataUnit:
        return ApplyToDataUnit(self._predict_backward_)(y)

    @property
    def copy(self) -> 'Node':
        return copy.copy(self)