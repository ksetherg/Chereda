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
                 preprocessor: object = None):
        
        self.df = df
        self.data = data

        if df is None:
            assert path is not None, 'Path to image dir is required.'
            self.__post_init__(Path(path), target_map)
        
        if preprocessor is not None and data is None:
            self.data = np.asarray(self._preprocess_(self._load_(), preprocessor))
        elif preprocessor is not None and data is not None:
            self.data = np.asarray(self._preprocess_(data, preprocessor))

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

    def _preprocess_(self, x: list, preprocessor: object) -> list:
        prep_x = []
        for i, img in enumerate(tqdm(x)):
            img_new = preprocessor(img)
            if img_new is not None:
                prep_x.append(img_new)
            else:
                # self.df.drop(self.df.index[[i]])
                self.df.iloc[i, 1] = -1
        return prep_x

    def _load_(self) -> list:
        print('Loading imgs...')
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
            imgs = np.asarray(self._load_())
        X = self.copy(data=imgs)
        return X

    @property
    def Y(self) -> Data:
        target = self.df['target'].to_numpy()
        # target = target[~np.isnan(target)]
        y = self.copy(data=target)
        return y

    @property
    def index(self) -> np.ndarray:
        return self.df.index.to_numpy()

    def get_by_index(self, index) -> Data:
        mask = self.df.index.isin(index)
        chunk = self.df.loc[mask]
        batch = None
        if self.data is not None:
            batch = self.data[mask]
        new = self.copy(df=chunk, data=batch)
        return new

    def reindex(self, index) -> Data:
        df = self.df.reindex(index)
        new = self.copy(df=df)
        return new

    @classmethod
    def combine(cls, datas: List[Data]) -> Data:
        dfs = [data.df for data in datas]
        dfs = [df.set_index('img_path', append=True) for df in dfs]
        df_cmbn = pd.concat(dfs, axis=1, keys=range(len(dfs)))
        df_cmbn = df_cmbn.groupby(level=[1], axis=1).mean()
        df_cmbn = df_cmbn.reset_index(level=1)

        datas_list = [data.data for data in datas]
        data_cmbn = np.concatenate(datas_list)
        new = datas[0].copy(df=df_cmbn, data=data_cmbn)
        return new
        