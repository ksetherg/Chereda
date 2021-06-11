from typing import List, Iterator, Tuple
import torch
import torch.nn.functional as F

from ..core import Operator, ApplyToDataUnit, DataUnit, Data


class NNModel(Operator):
    def __init__(self, archtr,
                 optimizer,
                 loss_func,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.archtr = archtr
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.train_loss_error = None
        self.valid_loss_error = None
            
    def x_forward(self, x: Data) -> Data:
        x_frwd = x.to(torch.device("cuda"), dtype=torch.float32)
        return x_frwd

    def y_forward(self, y: Data, x: Data = None, x_frwd: Data = None) -> Data:
        y_frwd = y.to(torch.device("cuda"), dtype=torch.long)
        return y_frwd

    def y_backward(self, y_frwd: Data) -> Data:
        return y_frwd

    def fit(self, x: Data, y: Data) -> Tuple[Data, Data]:
        self.model.train()
        x_frwd = self.x_forward(x)
        y_frwd = self.y_forward(y)
        
        self.optimizer.zero_grad()
        y_pred = self.model(x_frwd)
        loss = self.loss_func(y_pred, y_frwd)
        loss.backward()
        self.optimizer.step()
        self.train_loss_error = loss.item()
        return x, y

    def predict_forward(self, x : Data) -> Data:
        x_frwd = self.x_forward(x)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_frwd)
            prob = F.softmax(y_pred, dim=1)
        return prob

    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.y_backward(y_frwd)
        return y
