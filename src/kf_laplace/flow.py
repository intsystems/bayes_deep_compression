from functools import  reduce

import torch.nn as nn

from src.kf_laplace.filters import HessianFold


class KfBayessianFlow:
    '''
    Goes after in invert direction 
    '''
    def __init__(self,
        fold: HessianFold
    ):
        self.fold = fold

    def backward(self, model: nn.Module):
        return reduce(
            self.fold,
            reversed(model.modules)
        )
        
            

