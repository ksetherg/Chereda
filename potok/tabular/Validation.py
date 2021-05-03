from ..core import Operator, DataLayer, DataUnit


class Validation(Operator):
    def __init__(self, folder, **kwargs):
        super().__init__(**kwargs)
        # may be it's better to merge the logic of Folder and Validation
        self.folder = folder

    def x_forward(self, x: DataUnit) -> DataLayer:
        x2 = self.folder.get_folds(x)
        return x2

    def y_forward(self, y: DataUnit, x: DataUnit, x_frwd: DataUnit) -> DataLayer:
        y2 = self.folder.get_folds(y)
        return y2

    def y_backward(self, y_frwd: DataLayer) -> DataUnit:
        y = self.folder.combine_folds(y_frwd)
        return y

    def fit(self, x: DataUnit, y: DataUnit) -> (DataLayer, DataLayer):
        self.folder.generate_folds(x, y)
        x_frwd = self.x_forward(x)
        y_frwd = self.y_forward(y, x, x_frwd)
        return x_frwd, y_frwd

    def predict_forward(self, x: DataUnit) -> DataLayer:
        x_frwd = self.x_forward(x)
        return x_frwd

    def predict_backward(self, y_frwd: DataLayer) -> DataUnit:
        y = self.y_backward(y_frwd)
        return y