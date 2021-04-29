from dataclasses import dataclass
from typing import Any, List
import copy

from .Data import Data
from .DataLayer import DataLayer


@dataclass #(frozen=True) not pickable
class DataUnit:    
    train: Data = None
    valid: Data = None
    test: Data = None
    
    def __post_init__(self):
        types = [type(v) for k, v in self]
        assert len(types) >= 1, 'Currently empty DataUnits are not allowed.'
        assert types.count(types[0]) == len(types), 'All types must be the same.'
        
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
    
    def __reduce__(self):
        return self.__class__, copy.copy(tuple(self.__dict__.values()))
        
    def __iter__(self):
        new_dict = {k: v for k, v in self.__dict__.items() if v is not None}
        return iter(new_dict.items())
    
    def __getitem__(self, key: str):
        return self.__dict__[key]
    
    def __setitem__(self, key: str, value: Data):
        assert key in ['train', 'valid', 'test'], f'Key Error: {key}.'
        self.update({key: value})

    def to_dict(self):
        return self.__dict__.copy()
    
    def update(self, state: dict):
        self.__dict__.update(state)
        return 
        
    @property
    def units(self):
        units = [k for k, v in self]
        return units
    
    @property
    def X(self):
        X = {k: v.X for k, v in self}
        return DataUnit(**X)
    
    @property
    def Y(self):
        Y = {k: v.Y for k, v in self}
        return DataUnit(**Y)
    
    @property
    def index(self):
        indx = {k: v.index for k, v in self}
        return DataUnit(**indx)

    def get_by_index(self, index: 'DataUnit'):
        assert self.units == index.units, 'Units must match.'
        res = {k1: v1.get_by_index(v2) for k1, v1, k2, v2 in zip(self, index)}
        return DataUnit(**res)

    def reindex(self, index: 'DataUnit'):
        assert self.units == index.units, 'Units must match.'
        res = {k1: v1.reindex(v2) for k1, v1, k2, v2 in zip(self, index)}
        return DataUnit(**res)

    @classmethod
    def combine(cls, datas: List['DataUnit']):
        layer = DataLayer(*datas)
        args = {}
        for unit in layer.units:
            unit_data = layer[unit].to_list()
            args[unit] = unit_data[0].combine(unit_data)
        raise DataUnit(*args)
    
    def copy(self, **kwargs):
        new_data = copy.copy(self)
        new_data.__dict__.update(kwargs)
        return new_data