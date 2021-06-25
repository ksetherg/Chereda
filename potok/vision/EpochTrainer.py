from typing import List, Iterator, Tuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
from time import gmtime, strftime


from ..core import Node, ApplyToDataUnit, DataUnit, Data, DataLayer

class EpochTrainer(Node):
    def __init__(self, model,
                    epochs,
                    **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.epochs = epochs

    def transform_forward(self, data: DataUnit) -> DataLayer:
        if not isinstance(data, Data):
            raise Exception('Invalid data type.')
        if not isinstance(data, DataLayer):
            data = DataLayer(data)
        return data
    
    def transform_backward(self, data: DataLayer) -> DataUnit:
        if not isinstance(data, DataLayer):
            raise Exception('Invalid data type.')
        return data.args[0]

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        path_to_tb = './logs' + '/run_' + strftime("%y_%m_%d_%H_%M_%S", gmtime())   
        writer = SummaryWriter(path_to_tb)

        x, y = x.X, y.Y
        x_frwd, y_frwd = self.transform_forward(x), self.transform_forward(y)
        
        for e in range(self.epochs):
            print(f'Training Epoch: {e+1}/{self.epochs}')
            y2 = self.model.fit_predict(x_frwd, y_frwd)
            y2 = self.transform_backward(y2)
            error = self.get_error(y2, y)
            print('train_loss=', error['train'], 'valid_loss=', error['valid'])
            writer.add_scalars('Loss', {'train': error['train'],
                                        'valid': error['valid']}, e)
        return x, y2

    def predict_forward(self, x : DataUnit) -> DataUnit:
        x_frwd = self.transform_forward(x)
        y_frwd = self.model.predict(x_frwd)
        return y_frwd

    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.transform_backward(y_frwd)
        return y

    @ApplyToDataUnit()
    def get_error(self, y_pred: DataUnit, y_true: DataUnit) -> DataUnit:
        pred = torch.from_numpy(y_pred.data)
        true = torch.from_numpy(y_true.data)
        error = F.nll_loss(pred, true)
        return error
    