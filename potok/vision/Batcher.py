from ..core import Node, Operator, ApplyToDataUnit, DataUnit, Data, DataLayer

from torch.utils.data import SubsetRandomSampler, BatchSampler, RandomSampler
from typing import List, Iterator, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc


class Batcher(Node):
    def __init__(self, model,
                       batch_size,
                       **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.batch_size = batch_size
        self.train_error = None
        self.valid_error = None

    def batch_sampler(self, n, batch_size):
        indx_sampler = RandomSampler(range(n))
        batch_sampler = BatchSampler(indx_sampler, batch_size, drop_last=False)
        return batch_sampler

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        print('Training on batchs...')
        x_train, x_valid = x['train'], x['valid']
        y_train = y['train']
        
        train_btchs = self.batch_sampler(len(x_train), self.batch_size)
        valid_btchs = self.batch_sampler(len(x_valid), self.batch_size)

        train = []
        valid = []
        for train_indx in tqdm(train_btchs):
            x_tr_batch, y_tr_batch = x_train.get_by_index(train_indx), y_train.get_by_index(train_indx)
            _, y_train_pred = self.model.fit(x_tr_batch, y_tr_batch)
            train.append(y_train_pred)

        for valid_indx in tqdm(valid_btchs):
            x_val_batch = x_valid.get_by_index(valid_indx)
            y_valid_pred = self.model.predict_forward(x_val_batch)
            valid.append(y_valid_pred)

        y_frwd = DataUnit(train=y['train'].combine(train), valid=y['valid'].combine(valid))

        gc.collect()
        return x, y_frwd

    def predict_forward(self, x : DataUnit) -> DataUnit:
        x2 = ApplyToDataUnit()(self.model.predict_forward(x))
        return x2
    
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = ApplyToDataUnit()(self.model.predict_backward(y_frwd))
        return y