from typing import List, Iterator, Tuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
from time import gmtime, strftime
from pathlib import Path


from ..core import Node, ApplyToDataDict, DataDict, Data

class EpochTrainer(Node):
    def __init__(self, model,
                    epochs,
                    **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.epochs = epochs

    def _restate_(self) -> None:
        self.__dict__['model'] = None

    def _save_(self, prefix: Path = None) -> None:
        path = prefix / self.model.name
        self.model.save(path)

    def _load_(self, prefix: Path = None) -> None:
        path = prefix / self.model.name
        self.model.load(path)

    def transform_forward(self, data: DataDict) -> DataDict:
        if not isinstance(data, Data):
            raise Exception('Invalid data type.')
        if not isinstance(data, DataDict):
            data = DataDict(train=data)
        return data
    
    def transform_backward(self, data: DataDict) -> DataDict:
        if not isinstance(data, DataDict):
            raise Exception('Invalid data type.')
        return data['train']

    def fit(self, x: DataDict, y: DataDict) -> Tuple[DataDict, DataDict]:
        # path_to_tb = './logs' + '/run_' + strftime("%y_%m_%d_%H_%M_%S", gmtime())   
        # writer = SummaryWriter(path_to_tb)

        x, y = x.X, y.Y
        x_frwd, y_frwd = self.transform_forward(x), self.transform_forward(y)
        
        print("WHAT", self.model)

        assert self.epochs > 0
        for e in range(self.epochs):
            print(f'Training Epoch: {e+1}/{self.epochs}')
            y2 = self.model.fit_predict(x_frwd, y_frwd)
            y2 = self.transform_backward(y2)
            error = self.get_error(y2, y)
            print('train_loss=', error['train'], 'valid_loss=', error['valid'])
            # writer.add_scalars('Loss', {'train': error['train'],
            #                             'valid': error['valid']}, e)
        return x, y2

    def predict_forward(self, x : DataDict) -> DataDict:
        x_frwd = self.transform_forward(x)
        y_frwd = self.model.predict(x_frwd)
        return y_frwd

    def predict_backward(self, y_frwd: DataDict) -> DataDict:
        y = self.transform_backward(y_frwd)
        return y

    @ApplyToDataDict()
    def get_error(self, y_pred: DataDict, y_true: DataDict) -> DataDict:
        pred = torch.from_numpy(y_pred.data)
        true = torch.from_numpy(y_true.data)
        error = F.nll_loss(pred, true)
        return error
    