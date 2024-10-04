from abc import abstractmethod
from functools import reduce


import torch
import torch.nn as nn


class BaseFilter:
    """
    Manipulates with hessian from
    """

    @abstractmethod
    def __call__(
        self,
    ): ...


class SoftMaxActivation(BaseFilter):
    def __call__(self, layer_acts: torch.Tensor):
        """
        for i == j    H = ai * (1 - ai)
        """
        Hl = -layer_acts.unsqueeze(1) * layer_acts.unsqueeze(2)
        Hl[:, torch.eye(layer_acts.shape[0]).type(torch.ByteTensor)] = layer_acts * (
            1 - layer_acts
        )
        return Hl


class HessianRecursion(BaseFilter):
    def __call__(
        self,
        prev_hessian: torch.Tensor,
        prev_weights: torch.Tensor,
        layer_pre_acts: torch.Tensor,
    ):
        """
        https://arxiv.org/pdf/1706.03662 (29)
        """
        batch_size, newside = layer_pre_acts.shape
        I = torch.eye(newside).type(torch.ByteTensor)

        B = prev_weights.data.new(batch_size, newside, newside).fill_(0)
        B[:, I] = (layer_pre_acts > 0).type(B.type())

        D = torch.zeros_like(prev_weights)
        return B @ prev_weights @ prev_hessian @ prev_weights @ B + torch.zeros_like()


class Fold:
    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    def __call__(self, module: nn.Module):
        return reduce(module, self.filters)
