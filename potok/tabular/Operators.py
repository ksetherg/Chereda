import numpy as np
from ..core import Operator, DataDict, ApplyToDataDict
from typing import Tuple, Union


class TransformY(Operator):
    def __init__(self, transform: Union[tuple, str], target: str, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform
        self.target = target

        std_funcs = {
            'exp': (np.exp, np.log),
            'log': (np.log, np.exp),
            'square': (np.square, np.sqrt),
            'sqrt': (np.sqrt, np.square),
        }

        if transform in std_funcs:
            forward, backward = std_funcs[transform]
        else:
            forward, backward = transform

        self.forward = forward
        self.backward = backward

    @ApplyToDataDict()
    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        df = y.data
        df[self.target] = self.forward(df[self.target])
        y_frwd = y.copy(data=df)
        return y_frwd

    @ApplyToDataDict()
    def y_backward(self, y_frwd: DataDict) -> DataDict:
        df = y_frwd.data
        df[self.target] = self.backward(df[self.target])
        y = y_frwd.copy(data=df)
        return y
