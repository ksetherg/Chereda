from ..core import Node, DataLayer, DataUnit


class Validation(Node):
    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        self.folder = folder

    def fit(self, x: DataUnit, y: DataUnit) -> (DataLayer, DataLayer):
        assert isinstance(x, DataUnit) == isinstance(y, DataUnit), 'Validation works only with DataUnits.'
        self.folder.generate_folds(x, y)
        '''can be parallelized'''
        x2 = self.folder.get_folds(x)
        y2 = self.folder.get_folds(y)
        return x2, y2

    def predict_forward(self, x: DataUnit) -> DataLayer:
        assert self.folder.folds is not None, 'Fit your Folder before.'
        x2 = self.folder.get_folds(x)
        return x2

    def predict_backward(self, y: DataLayer) -> DataUnit:
        unfolded = self.folder.combine_folds(y)
        unfolded = DataUnit(train=unfolded['valid'], test=unfolded['test'])
        return unfolded
        