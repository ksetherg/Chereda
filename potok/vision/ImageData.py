from dataclasses import dataclass
from typing import List, Any
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import gc
from tqdm import tqdm

from ..core import Data

@dataclass(init=False)
class ImageClassificationData(Data):
    df: pd.DataFrame
    data: List[Any]
        
    def __init__(self, 
                 df: pd.DataFrame = None,
                 data: list = None,
                 path: str = None,
                 target_map: dict = None,
                 preprocessor: Any = None):
        
        self.df = df
        self.data = data

        self.preprocessor = preprocessor
        if df is None:
            assert path is not None, 'Path to image dir is required.'
            self.__post_init__(Path(path), target_map)

    def __post_init__(self, path, target_map):
        imgs = list(path.glob('**/*.jpg'))
        idx = list(range(len(imgs)))
        df = pd.DataFrame(data={'img_path': imgs},  index=idx)
        df['img_path'] = df['img_path'].astype(str)
        df = df.reset_index().set_index('index')

        if target_map is not None:
            target = [i.parent.stem for i in imgs]
            df['target'] = target
            df['target'] = df['target'].map(target_map)
        else:
            df['target'] = np.nan
       
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        chunk = self.df.iloc[index]
        if self.data is None:
            img = cv2.imread(chunk['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = self.data[index]
        target = chunk['target']
        return img, target

    def _preprocess_(self, x: list) -> list:
        prep_x = []
        for i, img in enumerate(x):
            img_new = self.preprocessor(img)
            if img_new is not None:
                prep_x.append(img_new)
            else:
                self.df.iloc[i, 1] = -1
        return prep_x

    def _load_(self) -> list:
        print('Loading images...')
        imgs = []
        for path in tqdm(self.df['img_path']): 
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            imgs.append(img)
        return imgs
    
    @property
    def X(self) -> Data:
        imgs = self.data
        if imgs is None:
            imgs = self._load_()
        if self.preprocessor is not None:
            imgs = self._preprocess_(imgs)
        X = self.copy(data=np.asarray(imgs))
        return X

    @property
    def Y(self) -> Data:
        target = self.df['target'].to_numpy()
        y = self.copy(data=target)
        return y

    @property
    def index(self) -> np.ndarray:
        return self.df.index

    def get_by_index(self, index) -> Data:
        chunk = self.df.iloc[index]
        if self.data is not None:
            batch = self.data[index]
            new = self.copy(df=chunk, data=batch)
        else:
            new = self.copy(df=chunk)
        return new

    def reindex(self, index) -> Data:
        df = self.df.reindex(index)
        new = self.copy(df=df)
        return new

    @classmethod
    def combine(cls, datas: List[pd.DataFrame]) -> Data:
        dfs = [data.df for data in datas]
        dfs = [df.set_index('img_path', append=True) for df in dfs]
        df_cmbn = pd.concat(dfs, axis=1, keys=range(len(dfs)))
        df_cmbn = df_cmbn.groupby(level=[1], axis=1).mean()
        df_cmbn = df_cmbn.reset_index(level=1)
        new = datas[0].copy(df=df_cmbn)
        return new
        