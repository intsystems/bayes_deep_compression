import torch
from src.methods.bayes.base.net import BayesLinearLayer, MLPBayesModel
from src.methods.bayes.kf_laplace.model import KFLinearOutput


class KfLinear(BayesLinearLayer[KFLinearOutput]):
    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, device)

    def forward(self, input: torch.Tensor):
        return KFLinearOutput(
            activation=super().forward(input), pre_activation=super().forward(input)
        )


class KFMlp(MLPBayesModel[KfLinear]):
    def __init__(self, bayesian_layer_list: list[KfLinear]):
        super().__init__(bayesian_layer_list)

    def forward(self, x: torch.Tensor):
        layer_output = KFLinearOutput(activation=x, pre_activation=x)
        return [
            layer_output := layer(layer_output.activation)
            for layer in self.bayesian_layer_list
        ]
