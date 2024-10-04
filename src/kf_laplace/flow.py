from functools import reduce

import torch.nn as nn

from src.kf_laplace.filters import HessianFold


class WeightsAccumulator:
    def __init__(
        self,
    ):
        self.sum = 0
        self.counter = 0

    def __call__(self, module: nn.Module):
        self.sum += module.data.weights
        self.counter += 1


class KfBayessianFlow:
    """
    Goes after in invert direction
    """

    def __init__(self, fold: HessianFold, weight_accumulator: WeightsAccumulator):
        self.fold = fold
        self.weight_accumulator = weight_accumulator

    def backward(self, model: nn.Module):
        return reduce(self.fold, reversed(model.modules))
