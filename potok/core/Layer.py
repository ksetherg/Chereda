import ray

from .DataLayer import DataLayer
from .Node import Node

class Layer(Node):
    def __init__(self, *nodes, **kwargs):
        super().__init__(**kwargs)
        layer = []
        for node in nodes:
            if isinstance(node, Node):
                layer.append(node)
            else:
                raise Exception('Unknown node type!')
                
        self.layer = layer
        self.shapes = None
        
    def fit(self, x, y):
        """Consider args as DataLayer"""
        assert len(self.layer) == len(x) == len(y), 'Layer and data shapes must be same'
        actors = [ray.remote(node.__class__).remote(**node.__dict__) for node in self.layer]
        res = [node.fit.remote(*args) for node, args in zip(actors, zip(x, y))]
        """Update states"""
        states = ray.get([node.__getstate__.remote() for node in actors])
        [node.__setstate__(state) for node, state in zip(self.layer, states)]
        
        res = ray.get(res)
        X = DataLayer(*self.flatten_forward(map(lambda x: x[0], res)))
        Y = DataLayer(*self.flatten_forward(map(lambda x: x[1], res)))
        return X, Y
    
    def predict_forward(self, x):
        assert len(self.layer) == len(x), 'Layer and data shapes must be same'        
        res = [ray.remote(node.predict_forward.__func__).remote(node, xx) for node, xx in zip(self.layer, x)]
        result = ray.get(res)
        result1d = DataLayer(*self.flatten_forward(result))
        return result1d
    
    def predict_backward(self, y):
        y2 = self.flatten_backward(y)
        assert len(self.layer) == len(y2), 'Layer and data shapes must be same'
        res = [ray.remote(node.predict_backward.__func__).remote(node, yy) for node, yy in zip(self.layer, y2)]
        result = DataLayer(*ray.get(res))
        return result
    
    def _flatten_forward_(self, irr_list):
        flat = []
        shapes = []
        for i in irr_list:
            if not isinstance(i, list):
                flat.append(i)
                shapes.append(0)
            else:
                flat.extend(i)
                shapes.append(len(i))
        self.shapes = shapes
        return flat
    
    def _flatten_backward_(self, list1d):
        start = 0
        end = 0
        irr_list = []
        for d in self.shapes:
            if d == 0:
                sl = list1d[start]
                start += 1
                irr_list.append(sl)
            else:
                end = start + d
                sl = list1d[start:end]
                start = end
                irr_list.append(sl)
        return irr_list