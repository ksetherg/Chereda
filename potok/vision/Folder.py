from sklearn.model_selection import train_test_split

from ..core import DataDict
from ..tabular import Folder


class StratifiedImageFolder(Folder):
    def __init__(self,
                 n_folds: int = 1,
                 split_ratio: float = 0.2,
                 seed: int = 4242):
        super().__init__(n_folds=n_folds, split_ratio=split_ratio, seed=seed)
        self.split_ratio = split_ratio

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        assert x['train'] is not None, 'Train required.'

        index = x['train'].index
        strats = y['train'].Y.data
        train_idx, valid_idx = train_test_split(index,
                                                test_size=self.split_ratio, 
                                                random_state=self.seed, 
                                                stratify=strats)

        folds = {'Fold_1': DataDict(train=train_idx, valid=valid_idx)}
        self.folds = folds
        return None
