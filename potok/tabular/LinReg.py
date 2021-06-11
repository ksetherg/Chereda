import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from ..core import Node, ApplyToDataUnit, DataUnit, Data
from .TabularData import TabularData

class LinReg(Node):
    def __init__(self, 
                 target=None,
                 features=None,
                 weight=None,
                 **kwargs,):
 
        super().__init__(**kwargs)
        self.target = target
        self.features = features
        self.weight = weight

        self.model = None
        self.index = None

    @ApplyToDataUnit()
    def x_forward(self, x: Data) -> Data:
        x2 = x.data[self.features]
        return x2

    @ApplyToDataUnit()
    def y_forward(self, y: Data, x: Data = None, x_frwd: Data = None) -> Data:
        y2 = y.data
        y2 = y2.dropna()
        return y2

    @ApplyToDataUnit()
    def y_backward(self, y_frwd: Data) -> Data:
        y = TabularData(data=y_frwd, target=self.target)
        return y
    
    def fit(self, x, y):
        self.index = y.index

        if self.target is None:
            self.target = x['train'].target

        if self.features is None:
            self.features = x['train'].data.columns

        x_frwd = self.x_forward(x)
        y_frwd = self.y_forward(y)

        x_frwd = x_frwd.reindex(y_frwd.index)
    
        x_train = x_frwd['train']
        y_train = y_frwd['train'][self.target]

        if self.weight is not None:
            w_train = y_frwd['train'][self.weight]
            # self.model = sm.WLS(y_train, X_train, weights=w_train).fit()
        else:
            w_train = None
            # self.model = sm.OLS(y_train, X_train).fit()

        print('Training Model LinReg')
        print(f'X_train = {x_train.shape} y_train = {y_train.shape}')

        self.model = LinearRegression().fit(x_train, y_train, sample_weight=w_train)
        # y2 = self.predict_forward(x)
        return x, y

    @ApplyToDataUnit()
    def predict_forward(self, x : Data) -> Data:
        assert self.model is not None, 'Fit model before.'
        x = x.data[self.features]
        # prediction = self.model.predict(exog=X)
        prediction = self.model.predict(x)
        prediction = pd.DataFrame(prediction, index=x.index, columns=self.target)
        return prediction
    
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.y_backward(y_frwd)
        y = y.reindex(self.index)
        return y
