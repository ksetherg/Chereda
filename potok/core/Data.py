from dataclasses import dataclass
from typing import List, Iterator, Tuple
import copy
# import ray


@dataclass
class Data:
    def __str__(self) -> str:
        return self.__class__.__name__

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        return 
    
    def __reduce__(self):
        return self.__class__, copy.copy(tuple(self.__dict__.values()))
        
    @property
    def X(self) -> 'Data':
        raise Exception('Not implemented.')

    @property
    def Y(self) -> 'Data':
        raise Exception('Not implemented.')

    @property
    def index(self):
        raise Exception('Not implemented.')

    def get_by_index(self, index) -> 'Data':
        raise Exception('Not implemented')

    def reindex(self, index) -> 'Data':
        raise Exception('Not implemented')

    @classmethod
    def combine(cls, datas: List['Data']) -> 'Data':
        raise Exception('Not implemented')
    
    def copy(self, **kwargs) -> 'Data':
        new_data = copy.copy(self)
        new_data.__dict__.update(kwargs)
        return new_data


@dataclass
class DataUnit(Data):
    train: Data = None
    valid: Data = None
    test: Data = None
    
    def __post_init__(self):
        types = [type(v) for k, v in self]
        assert len(types) >= 1, 'Currently empty DataUnit are not allowed.'
        assert types.count(types[0]) == len(types), 'All unit types must be the same.'
        
    def __iter__(self) -> Iterator:
        new_dict = {k: v for k, v in self.__dict__.items() if v is not None}
        return iter(new_dict.items())
    
    def __getitem__(self, key: str) -> Data:
        return getattr(self, key, None)
    
    def __setitem__(self, key: str, value: Data):
        assert key in ['train', 'valid', 'test'], f'Key Error: {key}.'
        self.update({key: value})
        return

    def to_dict(self) -> dict:
        return self.copy()
    
    def update(self, state: dict):
        self.__dict__.update(state)
        return 
        
    @property
    def units(self) -> list:
        units = [k for k, v in self]
        return units
    
    @property
    def X(self) -> 'DataUnit':
        X = {k: v.X for k, v in self}
        return DataUnit(**X)
    
    @property
    def Y(self) -> 'DataUnit':
        Y = {k: v.Y for k, v in self}
        return DataUnit(**Y)
    
    @property
    def index(self) -> 'DataUnit':
        indx = {k: v.index for k, v in self}
        return DataUnit(**indx)

    def get_by_index(self, index: 'DataUnit') -> 'DataUnit':
        assert self.units == index.units, 'Units must match.'
        # f = ray.remote(lambda x, y: x.get_by_index(y))
        # res = ray.get([f.remote(v1, v2) for (k1, v1), (k2, v2) in zip(self, index)])
        # res = {k: v for k, v in zip(self.units, res)}
        res = {k1: v1.get_by_index(v2) for (k1, v1), (k2, v2) in zip(self, index)}
        return DataUnit(**res)

    def reindex(self, index: 'DataUnit') -> 'DataUnit':
        assert self.units == index.units, 'Units must match.'
        # f = ray.remote(lambda x, y: x.reindex(y))
        # res = ray.get([f.remote(v1, v2) for (k1, v1), (k2, v2) in zip(self, index)])
        # res = {k: v for k, v in zip(self.units, res)}
        res = {k1: v1.reindex(v2) for (k1, v1), (k2, v2) in zip(self, index)}
        return DataUnit(**res)

    @classmethod
    def combine(cls, datas: 'DataLayer') -> 'DataUnit':
        units = datas.units
        data_cls = datas.args[0][units[0]].__class__

        # f = ray.remote(lambda x: data_cls.combine(x))
        # res = ray.get([f.remote(datas[unit].to_list()) for unit in units])
        # res = {k: v for k, v in zip(units, res)}

        f = lambda x: data_cls.combine(x)
        res = {unit: f(datas[unit].to_list()) for unit in units}
        return DataUnit(**res)
    

@dataclass(init=False)
class DataLayer(Data):
    args: List[Data]

    def __init__(self, *args):
        self.args = args
        self.validate_types(args)
    
    def validate_types(self, args: List[Data]):
        assert len(args) > 0, 'Currently empty DataLayer are not allowed.'
        notna = [arg is not None for arg in args]
        assert all(notna), 'DataLayer contains None.'
        types = [type(arg) for arg in args]
        assert types.count(types[0]) == len(types), 'All types must be the same.'
        
    def __reduce__(self):
        return self.__class__, copy.copy(tuple(*self.__dict__.values()))
        
    def __iter__(self) -> Iterator:
        return iter(self.to_list())
    
    def __len__(self) -> int:
        return len(self.to_list())
    
    def __getitem__(self, key: str) -> List[Data]:
        res = [getattr(arg, key, None) for arg in self]
        return DataLayer(*res)
    
    def __setitem__(self, key, value):
        raise Exception('Not implemented.')
            
    def to_list(self) -> list:
        return list(*self.__dict__.copy().values())

    def update(self, data: list):
        new = self.__dict__.update({'args': tuple(data)})
        return new

    @property
    def units(self) -> list:
        assert all([hasattr(arg, 'units') for arg in self]), 'Data object has no property units.'
        units = [arg.units for arg in self]
        intersection = list(set.intersection(*map(set, units)))
        assert len(intersection) >= 1, 'Units intersection is empty.'
        return intersection

    @property
    def X(self) -> 'DataLayer':
        X = [arg.X for arg in self]
        return DataLayer(*X)
    
    @property
    def Y(self) -> 'DataLayer':
        Y = [arg.Y for arg in self]
        return DataLayer(*Y)
    
    @property
    def index(self) -> 'DataLayer':
        indx = [v.index for v in self]
        return DataLayer(*indx)

    def get_by_index(self, index: 'DataLayer') -> 'DataLayer':
        assert len(self) == len(index), 'DataLayers must be same shape.'
        # f = ray.remote(lambda x, y: x.get_by_index(y))
        # res = ray.get([f.remote(v1, v2) for v1, v2 in zip(self, index)])
        res = [v1.get_by_index(v2) for v1, v2 in zip(self, index)]
        return DataLayer(*res)

    def reindex(self, index: 'DataLayer') -> 'DataLayer':
        assert len(self) == len(index), 'DataLayers must be same shape.'
        # f = ray.remote(lambda x, y: x.reindex(y))
        # res = ray.get([f.remote(v1, v2) for v1, v2 in zip(self, index)])
        res = [v1.reindex(v2) for v1, v2 in zip(self, index)]
        return DataLayer(*res)

    @classmethod
    def combine(cls, datas: List['DataLayer']):
        raise Exception('Not implemented.')
    