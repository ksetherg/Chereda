from ..core import Node, Operator, ApplyToDataUnit, DataUnit, Data, DataLayer
from .ImageData import ImageClassificationData

from torch.utils.data import SubsetRandomSampler, BatchSampler, RandomSampler
from typing import List, Iterator, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc


class Batcher(Operator):
    def __init__(self, batch_size,
                       **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        self.index = None
        self.batches = None

    def x_forward(self, x: DataUnit) -> DataUnit:
        x2 = self.get_batches(x, self.batches)
        return x2

    def y_forward(self, y: DataUnit, x: DataUnit = None, x_frwd: DataUnit = None) -> DataUnit:
        y2 = self.get_batches(y, self.batches)
        return y2

    def y_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.combine(y_frwd, self.index)
        return y

    def _fit_(self, x: DataUnit, y: DataUnit = None) -> None:
        self.index = x.index
        self.batches = self.generate_batches(x)
        return None
    
    def predict_forward(self, x: DataUnit) -> DataUnit:
        self._fit_(x)
        x_frwd = self.x_forward(x)
        return x_frwd

    def _batch_sampler_(self, indxs, batch_size):
        indx_sampler = SubsetRandomSampler(indxs)
        batch_sampler = BatchSampler(indx_sampler, batch_size, drop_last=False)
        return list(batch_sampler)

    @ApplyToDataUnit()
    def generate_batches(self, x: DataUnit) -> DataUnit:
        batches = self._batch_sampler_(x.index, self.batch_size)
        return batches

    @ApplyToDataUnit()
    def get_batches(self, xy: DataUnit, batches: DataUnit) -> DataUnit:
        batched = [xy.get_by_index(indx) for indx in batches]
        return batched

    @ApplyToDataUnit()
    def combine(self, datas: list, index: list):
        combined = ImageClassificationData.combine(datas)
        combined = combined.reindex(index)
        return combined

        