from functools import reduce
from typing import Iterator

import torch
import torch.nn as nn

from src.methods.bayes.kf_laplace.net import KfMLP


class RecurseHessian:
    def __init__(self, model, data_point: torch.Tensor):
        self.model = model
        self.data_point = data_point

    def softmax_CE_preact_hessian(self, last_layer_acts):
        side = last_layer_acts.shape[1]
        I = torch.eye(side).type(torch.ByteTensor)
        # for i != j    H = -ai * aj -- Note that these are activations not pre-activations
        Hl = -last_layer_acts.unsqueeze(1) * last_layer_acts.unsqueeze(2)
        # for i == j    H = ai * (1 - ai)
        Hl[:, I] = last_layer_acts * (1 - last_layer_acts)
        return Hl

    def propogate(self):
        outputs = self.model(self.data_point)

        return (self() for layer in zip(self.model, outputs))

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
        return B @ prev_weights @ prev_hessian @ prev_weights @ B + torch.zeros_like()

class Tracker:
    def __init__(self, generator):
        self.generator = generator
        self.cnt =0 

    def __iter__(self):
        return self

    def __next__(self):
        self.cnt += 1
        return next(self.generator)


class HessianAccumulator:
    def __init__(self, hessian_generators: Iterator[RecurseHessian]):
        tracked_hessian_generators = Tracker(hessian_generators)
        self.hessian_accumulation = reduce(
            lambda accumulated_hessian_generator, next_hessian_generator: 
                [u_elem + v_elem for u_elem, v_elem in zip(accumulated_hessian_generator, next_hessian_generator)],
            tracked_hessian_generators,
            list(next(tracked_hessian_generators)),
        )
        self.avg_hessian = [hessian/self.number_of_iteration for hessian in self.hessian_accumulation]

        
    @property
    def statistics(self):
        return (
            (
                torch.cholesky_inverse(hessian_accumulation, upper=False),
                torch.cholesky_inverse(hessian_accumulation),
            )
            for hessian_accumulation in self.hessian_accumulation
        )
