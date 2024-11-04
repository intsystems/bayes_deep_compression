from abc import abstractmethod
from functools import reduce

import torch
import torch.nn as nn
import torch.nn as nn


class HessianAccumulator:
    def __init__(
        self,
    ):
        self.sum = 0
        self.counter = 0

    def map(self):
        return ...
    
    def 

    def __call__(self, module: nn.Module):
        self.sum += module.data.weights
        self.counter += 1


class HessianRecursion:
    def __init__(self, last_layer_hessian: torch.Tensor):
        self.last_layer_hessian = last_layer_hessian

    def fold(self, model: nn.Module):
        return reduce(self, reversed(model.modules))

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
