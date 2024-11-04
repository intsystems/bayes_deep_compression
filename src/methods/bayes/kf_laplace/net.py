import torch
import torch.nn as nn
from torch.nn import Linear

from src.methods.bayes.base.net import MLPBayesModel
from src.methods.bayes.kf_laplace.model import KFLinearOutput


class KfMLP(MLPBayesModel[Linear]):
    def __init__(self, layer_list: list[nn.Linear], activation: list[nn.Module]):
        super().__init__(layer_list)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        layer_output = x
        return [
            KFLinearOutput(
                pre_activation=(pre_activation := layer(layer_output)),
                activation=(layer_output := self.activation(pre_activation)),
            )
            for layer in self.bayesian_layer_list
        ]
