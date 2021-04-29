from dataclasses import dataclass
import copy

from .DataUnit import DataUnit


@dataclass(init=False)
class DataLayer:
    args: List[DataUnit]

    def __init__(self, *args):
        self.args = args
        self.validate_types(args)
    
    def validate_types(self, args):
        notna = [arg is not None for arg in args]
        assert all(notna), 'DataLayer contains None.'
        types = [type(arg) for arg in args]
        assert types.count(types[0]) == len(types), 'All types must be the same.'
        
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __reduce__(self):
        return self.__class__, copy.copy(tuple(*self.__dict__.values()))
        
    def __iter__(self):
        return iter(self.to_list())
    
    def __len__(self):
        return len(self.to_list())
    
    def __getitem__(self, key: str):
        res = [DataUnit(**{key: arg[key]}) for arg in self]
        return DataLayer(*res)
            
    def to_list(self):
        return list(*self.__dict__.copy().values())

    def update(self, data: list):
        new = self.__dict__.update({'args': tuple(data)})
        return

    @property
    def units(self):
        units = [arg.units for arg in self]
        intersection = list(set.intersection(*map(set, units)))
        assert len(intersection) >= 1, 'Units intersection is empty.'
        return intersection

    @property
    def X(self):
        X = [arg.X for arg in self]
        return DataLayer(*X)
    
    @property
    def Y(self):
        Y = [arg.Y for arg in self]
        return DataLayer(*Y)
    
    @property
    def index(self):
        indx = [v.index for v in self]
        return DataLayer(*indx)

    def get_by_index(self, index: 'DataLayer'):
        assert len(self) == len(index), 'DataLayers must be same shape!'
        res = [v1.get_by_index(v2) for v1, v2 in zip(self, index)]
        return DataLayer(*res)

    def reindex(self, index: 'DataLayer'):
        assert len(self) == len(index), 'DataLayers must be same shape!'
        res = [v1.reindex(v2) for v1, v2 in zip(self, index)]
        return DataLayer(*res)

    @classmethod
    def combine(cls, datas: List['DataLayer']):
        raise Exception('Not implemented')
    
    def copy(self, **kwargs):
        new_data = copy.copy(self)
        new_data.__dict__.update(kwargs)
        return new_data