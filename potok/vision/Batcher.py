from ..core import Node, Operator, ApplyToDataDict, DataDict, Data

from torch.utils.data import SubsetRandomSampler, BatchSampler


class Batcher(Operator):
    def __init__(self, batch_size,
                       **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

        self.index = None
        self.batches = None

    def x_forward(self, x: DataDict) -> DataDict:
        x2 = self.get_batches(x, self.batches)
        return x2

    def y_forward(self, y: DataDict, x: DataDict = None, x_frwd: DataDict = None) -> DataDict:
        y2 = self.get_batches(y, self.batches)
        return y2

    def y_backward(self, y_frwd: DataDict) -> DataDict:
        y = self.combine(y_frwd)
        y = y.reindex(self.index)
        return y

    def _fit_(self, x: DataDict, y: DataDict = None) -> None:
        self.index = x.index
        self.batches = self.generate_batches(x)
        return None
    
    def predict_forward(self, x: DataDict) -> DataDict:
        self._fit_(x)
        x_frwd = self.x_forward(x)
        return x_frwd

    @staticmethod
    def _batch_sampler_(indxs, batch_size):
        index_sampler = SubsetRandomSampler(indxs)
        batch_sampler = BatchSampler(index_sampler, batch_size, drop_last=False)
        return list(batch_sampler)

    @ApplyToDataDict()
    def generate_batches(self, x: DataDict) -> DataDict:
        batches = self._batch_sampler_(x.index, self.batch_size)
        return batches

    @ApplyToDataDict()
    def get_batches(self, xy: DataDict, batches: DataDict) -> DataDict:
        batched = [xy.get_by_index(indx) for indx in batches]
        return batched

    @ApplyToDataDict()
    def combine(self, datas: list):
        combined = datas[0].__class__.combine(datas)
        return combined

