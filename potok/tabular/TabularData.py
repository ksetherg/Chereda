import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

from ..core import Data

@dataclass(init=False)
class TabularData(Data):
    data: pd.DataFrame

    def __init__(self, 
                 data: pd.DataFrame,
                 target: list = None, 
                 ):
        self.data = data
        self.target = target

    @property
    def X(self) -> pd.DataFrame:
        columns = [col for col in self.data.columns if col not in self.target]
        X = self.copy(data=self.data[columns])
        return X

    @property
    def Y(self) -> pd.DataFrame:
        columns = [col for col in self.data.columns if col in self.target]
        Y = self.copy(data=self.data[columns])
        return Y

    @property
    def index(self):
        return self.data.index

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
        df_cmbn = pd.concat(dfs, axis=1, keys=range(len(dfs)))
        df_cmbn = df_cmbn.groupby(level=[1], axis=1).mean()
        new = datas[0].copy(data=df_cmbn)
        return new
    