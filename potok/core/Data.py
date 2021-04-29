import copy
from typing import List


class Data:
    def X(self):
        raise Exception('Not implemented.')

    def Y(self):
        raise Exception('Not implemented.')

    @property
    def index(self):
        raise Exception('Not implemented.')

    def get_by_index(self, index):
        raise Exception('Not implemented')

    def reindex(self, index):
        raise Exception('Not implemented')

    @classmethod
    def combine(cls, datas: List['Data']):
        raise Exception('Not implemented')
    
    def copy(self, **kwargs):
        new_data = copy.copy(self)
        new_data.__dict__.update(kwargs)
        return new_data
