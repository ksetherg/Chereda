from functools import partial
from itertools import starmap
import ray

from .Data import DataUnit


class ApplyToDataUnit:
    def __init__(self, wrapped, mode='all', backend='ray'):
        self.wrapped = wrapped
        self.mode = mode
        self.backend = backend
      
    #     @wrapt.decorator """Not picklable"""
#     def __call__(self, wrapped, instance, args, kwargs):
#         print('call')
#         return self.apply(wrapped, instance, *args, **kwargs)
      
    def __call__(self, *args, **kwargs):
        return self.apply(self.wrapped, self.wrapped.__self__, *args, **kwargs)
    
    def apply(self, wrapped, instance, *args, **kwargs):
        all_units = []
        for arg in args:
            all_units.append(arg.units)
        assert all_units.count(all_units[0]) == len(all_units)
        units = all_units[0]
            
        if ('train' in units) and (self.mode != 'all'):
            units.remove('train')

        args2 = [[arg[unit] for arg in args] for unit in units]
        
        if self.backend == 'ray':
            res = self.apply_with_ray(wrapped, instance, *args2, **kwargs)
        elif self.backend == 'map':
            res = self.apply_with_map(wrapped, instance, *args2, **kwargs)
        
        """Must return separately"""
        if isinstance(res[0], tuple):
            result = []
            for i in range(len(res[0])):
                arg = list(map(lambda x: x[i], res))
                result.append(DataUnit(**dict(zip(units, arg))))
            result = tuple(result)
        else:
            result = DataUnit(**dict(zip(units, res)))
        return result
    
    def apply_with_ray(self, wrapped, instance, *args, **kwargs):
        state = ray.put(instance)
        res = [ray.remote(wrapped.__func__).remote(state, *arg, **kwargs) for arg in args]
        return ray.get(res)
    
    def apply_with_map(self, wrapped, instance, *args, **kwargs):
        return list(starmap(partial(wrapped, **kwargs), args))