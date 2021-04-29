class Node:
    def __init__(self, **kwargs):
        if bool(kwargs) and ('name' in kwargs):
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__
        
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __str__(self):
        return self.name
    
    def _fit_(self, x, y):
        """Consider args as Units"""
        return x, y
        
    def fit(self, x, y):
        """Consider args as DataUnit"""
        return self._fit_(x, y)
        
    def _predict_forward_(self, x):
        return x
    
    def predict_forward(self, x):
        return self._predict_forward_(x)
            
    def _predict_backward_(self, y):
        return y
    
    def predict_backward(self, y):
        return self._predict_backward_(y)

    @property
    def copy(self):
        return copy.copy(self)