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

    def batch_sampler(self, indxs, batch_size):
        indx_sampler = SubsetRandomSampler(indxs)
        batch_sampler = BatchSampler(indx_sampler, batch_size, drop_last=False)
        return batch_sampler

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        print('Training on batchs...')
        x_train, x_valid = x['train'], x['valid']
        y_train, y_valid = y['train'], y['valid']
        
        train_btchs = self.batch_sampler(x_train.index, self.batch_size)
        valid_btchs = self.batch_sampler(x_valid.index, self.batch_size)

        train = []
        valid = []
        valid_errors = []
        for train_indx in tqdm(train_btchs):
            x_tr_batch, y_tr_batch = x_train.get_by_index(train_indx), y_train.get_by_index(train_indx)
            _, y_train_pred = self.model.fit(x_tr_batch, y_tr_batch)
            train.append(y_train_pred)

        for valid_indx in tqdm(valid_btchs):
            x_val_batch, y_val_batch = x_valid.get_by_index(valid_indx), y_valid.get_by_index(valid_indx)
            y_valid_pred = self.model.predict_forward(x_val_batch)
            valid.append(y_valid_pred)
            valid_loss_error = self.model.get_error(y_valid_pred, y_val_batch)
            valid_errors.append(valid_loss_error)
        
        print("Valid: ", torch.mean(torch.FloatTensor(valid_errors)))

        y_frwd = DataUnit(train=y['train'].combine(train), valid=y['valid'].combine(valid))
        y_frwd = y_frwd.reindex(y.index)

        gc.collect()
        return x, y_frwd

    @ApplyToDataUnit(mode='efficient')
    def predict_forward(self, x : DataUnit) -> DataUnit:
        print('Batcher', type(x))
        x2 = self.model.predict_forward(x)
        return x2

    @ApplyToDataUnit()
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.model.predict_backward(y_frwd)
        return y