# import copy
from .Data import Data, DataUnit, DataLayer
from .Node import Node, Operator
from .Layer import Layer
import gc

class Pipeline(Node):
    """Pipeline works with DataLayer and Layer"""
    def __init__(self, *nodes, **kwargs):
        super().__init__(**kwargs)
        _nodes_ = []
        for node in nodes:
            if isinstance(node, Pipeline):
                _nodes_.extend(node.nodes)
            elif isinstance(node, (Node, Operator)):
                _nodes_.append(node)
            else:
                raise Exception('Unknown node type.')
                
        self.nodes = _nodes_
        self.layers = None
        self.data_shapes = None
        self.current_fit = 0
        self.current_predict = 0

    def check_input(self, x, y):
        if (not isinstance(x, Data)) or (not isinstance(y, Data)):
            raise Exception('Unknown data type.')
        if not isinstance(x, DataLayer):
            x = DataLayer(x)
        if not isinstance(y, DataLayer):
            y = DataLayer(y)
        return x, y
    
    def next_layer(self, node, n):
        layer = [node.copy for i in range(n)]
        return Layer(*layer)
    
    def fit(self, x, y):
        x, y = self.check_input(x, y)
        layers = []
        for node in self.nodes:
            assert len(x) == len(y)
            layer = self.next_layer(node, len(x))
            x, y = layer.fit(x, y)
            gc.collect()
            layers.append(layer)
        self.layers = layers
        return x, y
    
    def fit_step(self, x, y):
        self.current_fit += 1
        assert self.current_fit <= len(self.nodes)
        x2, y2 = self._fit_until_(x, y)
        return x2, y2
    
    def _fit_until_(self, x, y):
        i = self.current_fit
        assert i >= 0
        layers = []
        for node in self.nodes:
            assert len(x) == len(y)
            layer = self.next_layer(node, len(x))
            x, y = layer.fit(x, y)
            layers.append(layer)
        self.layers = layers
        return x, y
    
    def predict_forward(self, x):
        if self.layers is None:
            raise Exception('Fit your model before.')
        for layer in self.layers:     
            x = layer.predict_forward(x)
        return x
    
    def predict_backward(self, y):   
        if self.layers is None:
            raise Exception('Fit your model before.')
        for layer in self.layers[::-1]:
            y = layer.predict_backward(y)
        return y

    def predict(self, x):
        y2 = self.predict_forward(x)
        y1 = self.predict_backward(y2)
        return y1

    def fit_predict(self, x, y):
        x2, y2 = self.fit(x, y)
        y1 = self.predict(x)
        return y1
    
    def __str__(self):
        pipe = f'({self.name}: '
        for node in self.nodes:
            pipe += str(node)
            if node != self.nodes[-1]:
                pipe += ' -> '
        pipe += ')'
        return pipe
        