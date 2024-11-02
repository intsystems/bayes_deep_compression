import torch
from abc import abstractmethod

class BasePruner:
    '''
    Takes distribution weights of
    net and decides if it can be equaled to zero
    '''
    @abstractmethod
    def prune(self, weights_distribution: torch.Tensor) -> torch.Tensor:...