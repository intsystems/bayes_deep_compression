import torch
import torch.nn as nn
from typing import Any


class WithGaussianPrior():
    def __init__(self) -> None:
        pass

    def __call__(self, net: nn.Module) -> nn.Module:
        ...



