from abc import abstractmethod

import torch

from src.methods.bayes.base.distribution import BaseNetDistribution


class BasePruner:
    """
    Takes distribution weights of
    net and decides if it can be equaled to zero
    """

    def __init__(self, net_distribution: BaseNetDistribution):
        self.net_distribution = net_distribution

    @abstractmethod
    def prune(self, weights_distribution: torch.Tensor) -> torch.Tensor: ...
