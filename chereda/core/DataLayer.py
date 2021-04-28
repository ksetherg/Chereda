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
        types = []
        for arg in args:
            if arg is not None:
                types.append(type(arg))
        assert types.count(types[0]) == len(types), 'All types must be the same.'
        
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __reduce__(self):
        return self.__class__, copy.copy(tuple(*self.__dict__.values()))
        
    def __iter__(self):
        return iter(*self.__dict__.values())
    
    def __len__(self):
        return len(list(*self.__dict__.values()))
    
    def __getitem__(self, idx: int):
        return list(*self.__dict__.values())[idx]
    
    def __setitem__(self, idx, value):
        assert idx < len(self), 'Index out of range.'
        data = list(*self.__dict__.values())
        data[idx] = value
        self.update(data)

    def update(self, data: list):
        new = self.__dict__.update({'args': tuple(data)})
        return

    @property
    def units(self):
        units = []
        for arg in self:
            if arg is not None:
                units.append(arg.units)
        intersection = list(set.intersection(*map(set, units)))
        return intersection

    @property
    def X(self):
        X = []
        for arg in self:
            if arg is not None:
                X.append(arg.X)
        return DataLayer(*X)
    
    @property
    def Y(self):
        Y = []
        for arg in self:
            if arg is not None:
                Y.append(arg.Y)
        return DataLayer(*Y)