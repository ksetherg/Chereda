from typing import List, Iterator, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import gc


from ..core import Node, ApplyToDataUnit, DataUnit, Data


class NNModel(Node):
    def __init__(self, model,
                 optimizer,
                 loss_func,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

        self.train_loss_error = None
            
    def transform_x(self, x: Data) -> Data:
        x_new = np.swapaxes(x.data, -1, 1)
        x_new = torch.from_numpy(x_new)
        x_new = x_new.to(torch.device("cuda"), dtype=torch.float32)
        return x_new

    def transform_y(self, y: Data) -> Data:
        y_new = torch.from_numpy(y.data)
        y_new = y_new.to(torch.device("cuda"), dtype=torch.long)
        return y_new
    
    def fit(self, x: Data, y: Data) -> Tuple[Data, Data]:
        self.model.train()

        x_new = self.transform_x(x)
        y_new = self.transform_y(y)
        
        self.optimizer.zero_grad()
        y_pred = self.model(x_new)
        loss = self.loss_func(y_pred, y_new)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            logits = F.log_softmax(y_pred, dim=1)
            y_frwd = y.copy(data=logits.cpu().numpy())
        gc.collect()
        return x, y_frwd

    def predict_forward(self, x : Data) -> Data:
        self.model.eval()
        x_frwd = self.transform_x(x)
        with torch.no_grad():
            y_pred = self.model(x_frwd)
            logits = F.log_softmax(y_pred, dim=1)
            pred = x.copy(data=logits.cpu().numpy())
        gc.collect()
        return pred
    
    def _state_(self, state):
        state['model'] = None
        state['optimizer'] = None
        return state

    def _save_(self, prefix: str):
        file_name='model_weights'
        ext = '.pth'
        path = prefix + file_name + '_' + ext
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                    path)

    def _load_(self, prefix: str):
        file_name = 'model_weights'
        ext = '.pth'
        path =  prefix + file_name + ext
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    