import torch

class BasePruner:
    '''
    Takes distribution weights of
    net and decides if it can be equaled to zero
    '''
    def prune(self, weights_distribution: torch.Tensor) -> torch.Tensor':
        pass