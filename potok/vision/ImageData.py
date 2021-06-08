from dataclasses import dataclass
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from ..core import Data


@dataclass(init=False)
class ImageClassificationData(Data):
    data: pd.DataFrame
        
    def __init__(self,  path: str):
        self.path = Path(path)
        self.get_pathes()
        
    def get_pathes(self):
        imgs = list(self.path.glob('**/*.jpg'))
        target = [int(i.parent.stem) for i in imgs]
        idx = list(range(len(imgs)))
        
        df = pd.DataFrame(data={'img_path': imgs, 'target': target},  index=idx)
        self.data = df.reset_index().set_index('index')
        
    @property
    def X(self) -> np.ndarray:
        X = []
        for path in self.data['img_path']: 
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            X.append(img) 
        return np.asarray(X)

    @property
    def Y(self) -> np.ndarray:
        y = self.data['target'].to_numpy()
        return y

    @property
    def index(self) -> np.ndarray:
        return self.data.index.to_numpy()

    def get_by_index(self, index) -> pd.DataFrame:
        chunk = self.data.loc[self.data.index.isin(index)]
        new = self.copy(data=chunk)
        return new

    def reindex(self, index) -> pd.DataFrame:
        df = self.data.reindex(index)
        new = self.copy(data=df)
        return new

    @classmethod
    def combine(cls, datas: List[pd.DataFrame]) -> pd.DataFrame:
        dfs = [data.data for data in datas]
        dfs = [df.set_index('img_path', append=True) for df in dfs]
        df_cmbn = pd.concat(dfs, axis=1, keys=range(len(dfs)))
        df_cmbn = df_cmbn.groupby(level=[1], axis=1).mean()
        df_cmbn = df_cmbn.reset_index(level=1)
        new = datas[0].copy(data=df_cmbn)
        return new
    