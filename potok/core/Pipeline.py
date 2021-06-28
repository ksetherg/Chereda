# import copy
from .Data import Data, DataUnit, DataLayer
from .Node import Node, Operator
from .Layer import Layer

import gc
from pathlib import Path
import os
import shutil
from time import gmtime, strftime
from typing import List, Iterator, Tuple


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
        # self.data_shapes = None
        # self.current_fit = 0
        # self.current_predict = 0
    
    def save(self, prefix: Path) -> None:
        suffix = strftime("%y_%m_%d_%H_%M_%S", gmtime())
        ppln_name = self.name + suffix
        for i, layer in enumerate(self.layers):
            suffix_lyr = layer.name + '_' + str(i)
            prefix_lyr = prefix / ppln_name / suffix_lyr
            layer.save(prefix_lyr)

    def load(self, prefix: Path):
        for i, layer in enumerate(self.layers):
            suffix_lyr = layer.name + '_' + str(i)
            prefix_lyr = prefix / suffix_lyr
            layer.load(prefix_lyr)
        
    def _next_layer_(self, node, n):
        layer = [node.copy for i in range(n)]
        return Layer(*layer)
    
    def fit(self, x: DataLayer, y: DataLayer) -> Tuple[DataLayer, DataLayer]:
        layers = []
        for node in self.nodes:
            assert len(x) == len(y)
            layer = self._next_layer_(node, len(x))
            x, y = layer.fit(x, y)
            layers.append(layer)
        self.layers = layers
        return x, y
    
    def predict_forward(self, x: DataLayer) -> DataLayer:
        if self.layers is None:
            raise Exception('Fit your model before.')
        for layer in self.layers:     
            x = layer.predict_forward(x)
        return x
    
    def predict_backward(self, y: DataLayer) -> DataLayer:
        if self.layers is None:
            raise Exception('Fit your model before.')
        for layer in self.layers[::-1]:
            y = layer.predict_backward(y)
        return y

    def predict(self, x: DataLayer) -> DataLayer:
        y2 = self.predict_forward(x)
        y1 = self.predict_backward(y2)
        return y1

    def fit_predict(self, x: DataLayer, y: DataLayer) -> DataLayer:
        x2, y2 = self.fit(x, y)
        y1 = self.predict_backward(y2)
        # y1 = self.predict(x)
        return y1
    
    def __str__(self) -> str:
        pipe = f'({self.name}: '
        for node in self.nodes:
            pipe += str(node)
            if node != self.nodes[-1]:
                pipe += ' -> '
        pipe += ')'
        return pipe
    
    # def _if_exist_(self, path):
    #     if os.path.exists(path):
    #         shutil.rmtree(path)
        
    # def fit_step(self, x, y):
    #     self.current_fit += 1
    #     assert self.current_fit <= len(self.nodes)
    #     x2, y2 = self._fit_until_(x, y)
    #     return x2, y2
    
    # def _fit_until_(self, x, y):
    #     i = self.current_fit
    #     assert i >= 0
    #     layers = []
    #     for node in self.nodes:
    #         assert len(x) == len(y)
    #         layer = self.next_layer(node, len(x))
    #         x, y = layer.fit(x, y)
    #         layers.append(layer)
    #     self.layers = layers
    #     return x, y