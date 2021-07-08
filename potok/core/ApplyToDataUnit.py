from functools import partial
from itertools import starmap
import wrapt
import ray

from .Data import DataDict


class ApplyToDataDict:
    def __init__(self, mode='all', backend='map'):
        # self.wrapped = wrapped
        self.mode = mode
        self.backend = backend
      
    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        return self.apply(wrapped, instance, *args, **kwargs)
      
    # def __call__(self, *args, **kwargs):
    #     return self.apply(self.wrapped, self.wrapped.__self__, *args, **kwargs)
    
    def apply(self, wrapped, instance, *args, **kwargs):
        all_units = []
        for arg in args:
            all_units.append(arg.units)
        assert all_units.count(all_units[0]) == len(all_units)
        units = all_units[0]
            
        if ('train' in units) and (self.mode != 'all'):
            units.remove('train')

        args2 = [[arg[unit] for arg in args] for unit in units]
        kwargs2 = {unit: {k: v[unit] for k, v in kwargs.items()} for unit in units}

        if self.backend == 'ray':
            res = self.apply_with_ray(wrapped, instance, *args2, **kwargs2)
        elif self.backend == 'map':
            res = self.apply_with_map(wrapped, instance, *args2, **kwargs2)
        else:
            raise Exception()
        
        """Must return separately"""
        if isinstance(res[0], tuple):
            result = []
            for i in range(len(res[0])):
                arg = list(map(lambda x: x[i], res))
                result.append(DataDict(**dict(zip(units, arg))))
            result = tuple(result)
        else:
            result = DataDict(**dict(zip(units, res)))
        return result

    @staticmethod
    def apply_with_ray(wrapped, instance, *args, **kwargs):
        state = ray.put(instance)
        res = [ray.remote(wrapped.__func__).remote(state, *arg, **kwarg) for arg, kwarg in zip(args, kwargs.values())]
        return ray.get(res)

    @staticmethod
    def apply_with_map(wrapped, instance, *args, **kwargs):
        return [wrapped(*arg, **kwarg) for arg, kwarg in zip(args, kwargs.values())]
