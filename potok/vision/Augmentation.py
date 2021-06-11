from ..core import Operator, Node
from ..core import Data, DataUnit
from ..core import ApplyToDataUnit

import copy
from typing import List, Iterator, Tuple
import numpy as np


class AlbAugment(Node):
    def __init__(self, 
                 augmentations,
                 apply_y=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.augmentations = augmentations
        self.apply_y = apply_y
    
    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        print('Augmenting data...')
        x_aug = copy.copy(x.X)
        y_aug = copy.copy(y.Y)
        x_aug, y_aug = self.augment(x_aug, y_aug, augs=self.augmentations)
        return x_aug, y_aug

    def predict_forward(self, x : DataUnit) -> DataUnit:
        x_aug = copy.copy(x.X)
        x_aug = self.augment(x_aug, augs=self.augmentations)
        return x_aug
    
    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        return y_frwd

    @ApplyToDataUnit()
    def augment(self, *args, **kwargs):
        assert len(args) != 0, 'Not enough arguments.'
        if kwargs:
            augmentations = kwargs['augs']

            if self.apply_y:
                assert len(args) == 2, 'Not enough arguments.'
                X, y = args
                imgs, msks = [], []
                for img, msk in zip(X.data, y.data):
                    augmented = augmentations(image=img, mask=msk)
                    imgs.append(augmented['image'])
                    msks.append(augmented['mask'])
                return X.copy(np.asarray(imgs)), y.copy(np.asarray(masks))
            else:
                X = args[0]
                imgs = []
                for img in X.data:
                    augmented = augmentations(image=img)
                    imgs.append(augmented['image'])

                if len(args) == 1:
                    return X.copy(np.asarray(imgs))
                elif len(args) == 2:
                    return X.copy(np.asarray(imgs)), args[1]
        else:
            return args