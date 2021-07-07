from typing import List
from sklearn.model_selection import train_test_split

from ..core import DataDict
from ..tabular import Folder


class StratifiedImageFolder(Folder):
    def __init__(self, n_folds: int = 1, split_ratio: float = 0.2, seed: int = 4242):
        super().__init__(n_folds=n_folds, seed=seed)
        self.split_ratio = split_ratio

    def _fit_(self, x: DataDict, y: DataDict) -> None:
        indx = x['train'].index
        strats = y['train'].Y.data
        train_idx, valid_idx = train_test_split(indx,
                                                test_size=self.split_ratio, 
                                                random_state=self.seed, 
                                                stratify=strats)
        folds = [DataDict(train_idx, valid_idx)]
        self.folds = folds
        return None