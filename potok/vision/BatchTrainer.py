from ..core import Node, Operator, ApplyToDataUnit, DataUnit, Data, DataLayer
from .ImageData import ImageClassificationData

from typing import List, Iterator, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc


class BatchTrainer(Node):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model


    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        x_train, x_valid = x['train'], x['valid']
        y_train, y_valid = y['train'], y['valid']

        train = []
        valid = []
        train_errors = []
        valid_errors = []

        for x_tr_batch, y_tr_batch in tqdm(zip(x_train, y_train), total=len(x_train), desc='Batch training'):
            _, y_train_pred = self.model.fit(x_tr_batch, y_tr_batch)
            train.append(y_train_pred)
            train_loss_error = self.model.get_error(y_train_pred, y_tr_batch)
            train_errors.append(train_loss_error)

        for x_val_batch, y_val_batch in tqdm(zip(x_valid, y_valid), total=len(x_valid), desc='Batch validating'):
            y_valid_pred = self.model.predict_forward(x_val_batch)
            valid.append(y_valid_pred)
            valid_loss_error = self.model.get_error(y_valid_pred, y_val_batch)
            valid_errors.append(valid_loss_error)
            
        print("Train: ", torch.mean(torch.FloatTensor(train_errors)), "Valid: ", torch.mean(torch.FloatTensor(valid_errors)))
        y_frwd = DataUnit(train=train, valid=valid)

        gc.collect()
        return x, y_frwd
    
    @ApplyToDataUnit()
    def predict_forward(self, x : DataUnit) -> DataUnit:
        batches = []
        for x_batch in tqdm(x, desc='Batch predicting'):
            y_pred = self.model.predict_forward(x_batch)
            batches.append(y_pred)
        return batches
