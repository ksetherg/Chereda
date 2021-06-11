from typing import List, Iterator, Tuple
import torch
import torch.nn.functional as F
import gc
from time import gmtime, strftime

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
            
    def x_forward(self, x: Data) -> Data:
        x = torch.from_numpy(x.data)
        x = torch.swapaxes(x, -1, 1)
        x_frwd = x.to(torch.device("cuda"), dtype=torch.float32)
        return x_frwd

    def y_forward(self, y: Data, x: Data = None, x_frwd: Data = None) -> Data:
        y = torch.from_numpy(y.data)
        y_frwd = y.to(torch.device("cuda"), dtype=torch.long)
        return y_frwd

    def y_backward(self, y_frwd: Data) -> Data:
        return y_frwd

    def fit(self, x: Data, y: Data) -> Tuple[Data, Data]:
        self.archtr.train()
        x_frwd = self.x_forward(x)
        y_frwd = self.y_forward(y)
        
        self.optimizer.zero_grad()
        y_pred = self.archtr(x_frwd)
        loss = self.loss_func(y_pred, y_frwd)
        loss.backward()
        self.optimizer.step()
        self.train_loss_error = loss.item()
        gc.collect()
        return x, y

    def predict_forward(self, x : Data) -> Data:
        x_frwd = self.x_forward(x)
        self.archtr.eval()
        with torch.no_grad():
            y_pred = self.archtr(x_frwd)
            logits = F.log_softmax(y_pred, dim=1)
        return logits

    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.y_backward(y_frwd)
        return y

    def get_error(self, y_pred, y_true):
        y_frwd = self.y_forward(y_true)
        error = F.nll_loss(y_pred, y_frwd)
        return error

    def _save_(self, epoch, loss):
        prefix = 'models/'
        file_name='model_weights'
        suffix = strftime("%y_%m_%d_%H_%M_%S", gmtime())
        ext = '.pth'
        path = prefix + file_name + '_' + suffix + ext
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    },
                    path)