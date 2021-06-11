from ..core import Node, Operator, ApplyToDataUnit, DataUnit, Data, DataLayer

from torch.utils.data import SubsetRandomSampler, BatchSampler, RandomSampler
from typing import List, Iterator, Tuple
import math


class BatchFolder(Node):
    def __init__(self, model,
                       batch_size,
                       **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.batch_size = batch_size

    def batch_sampler(self, n, batch_size):
        indx_sampler = RandomSampler(range(n))
        batch_sampler = BatchSampler(indx_sampler, batch_size, drop_last=False)
        return batch_sampler

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:        
        train_n_splits = math.ceil(len(x['train']) / self.batch_size)
        assert len(x['valid']) >= train_n_splits, 'Batch size is too small'
        valid_batch_size = math.ceil(len(x['valid']) / train_n_splits)
        
        train_btchs = self.batch_sampler(len(x['train']), self.batch_size)
        valid_btchs = self.batch_sampler(len(x['valid']), valid_batch_size)
        
        folds = DataLayer(*[DataUnit(train=train, valid=valid) for train, valid in zip(train_btchs, valid_btchs)])
        x_folds = [x.get_by_index(indx) for indx in folds]
        y_folds = [y.get_by_index(indx) for indx in folds]
        return DataLayer(*x_folds), DataLayer(*y_folds)

    def predict_forward(self, x : DataUnit) -> DataUnit:
        x2 = model.predict_forward(x)
        return x2
    
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = model.predict_backward(y_frwd)
        return y