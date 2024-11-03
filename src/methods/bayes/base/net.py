from typing import Generic, TypeVar
from torch.nn import Linear
import torch.nn


class BayesLinearLayer(Linear):
    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, False, device, torch.float32)

LinearT = TypeVar('LinearT',bound=BayesLinearLayer)

class MLPBayesModel(torch.nn.Module, Generic[LinearT]):
    def __init__(self, bayessian_linear_layer_list: list[BayesLinearLayer]):
        self.bayessian_linear_layer_list = bayessian_linear_layer_list

    def forward(self, x: torch.Tensor[torch.float32]) -> T: ...
