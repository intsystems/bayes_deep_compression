import torch.nn as nn


class WeightsAccumulator:
    def __init__(
        self,
    ):
        self.sum = 0
        self.counter = 0

    def __call__(self, module: nn.Module):
        self.sum += module.data.weights
        self.counter += 1
