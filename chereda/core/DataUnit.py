from dataclasses import dataclass
from typing import Any
import copy


@dataclass #(frozen=True) not pickable
class DataUnit:    
    train: Any = None
    valid: Any = None
    test: Any = None
    
    def __post_init__(self):
        types = []
        for k, v in self:
            if v is not None:
                types.append(type(v))
        assert types.count(types[0]) == len(types), 'All types must be the same.'
        
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __reduce__(self):
        return self.__class__, copy.copy(tuple(self.__dict__.values()))
        
    def __iter__(self):
        return iter(self.__dict__.items())
    
    def __getitem__(self, key: str):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        assert key in ['train', 'valid', 'test'], f'Key Error: {key}'
        self.update({key: value})
    
    def update(self, state: dict):
        self.__dict__.update(state)
        return
        
    @property
    def units(self):
        units = []
        for k, v in self:
            if v is not None:
                units.append(k)
        return units
        
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
    @property
    def X(self):
        X = {}
        for k, v in self:
            if v is not None:
                X[k] = v.X
        return DataUnit.from_dict(X)
    
    @property
    def Y(self):
        Y = {}
        for k, v in self:
            if v is not None:
                Y[k] = v.Y
        return DataUnit.from_dict(Y)
    
