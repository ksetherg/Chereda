import scipy.stats as sst
import numpy as np
import pandas as pd

from potok.tabular import TabularData


class SyntheticData:
    def __init__(self,
                 pdf_train=sst.norm(loc=-1, scale=2),
                 pdf_test=sst.norm(loc=1, scale=3),
                 target_f=np.square,
                 seed=None):
        self.pdf_train = pdf_train
        self.pdf_test = pdf_test
        self.target_f = target_f
        self.seed = seed

    def _create_sample_(self, pdf, size, noize_sigma=None):
        x = pdf.rvs(size=size, random_state=self.seed)
        y = self.target_f(x)

        if noize_sigma is not None:
            y_noize = np.random.normal(0, noize_sigma, size)
            y += y_noize

        df = pd.DataFrame({'X': x, 'Target': y}, index=list(range(size)))
        return df

    def create_train(self, size=10, noize_sigma=0.5):
        df_train = self._create_sample_(self.pdf_train, size, noize_sigma)
        return TabularData(df_train, target=['Target'])

    def create_test(self, size=10, noize_sigma=None):
        data_test = self._create_sample_(self.pdf_test, size, noize_sigma)
        return TabularData(data_test, target=['Target'])
