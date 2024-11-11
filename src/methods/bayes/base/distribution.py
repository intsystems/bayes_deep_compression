import torch
import torch.nn as nn 
import torch.distributions as td
from torch.types import _size

class ParamDist(td.distribution.Distribution):
    @classmethod
    def from_parameter(self, p: nn.Parameter) -> 'ParamDist':
        ...
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def get_params(self) -> dict[str, nn.Parameter]: 
        ...
    def prob(self, weights):
        ...
    def log_prob(self, weights):
        ...
    def log_z_test(self):
        return torch.log(self.mean()) - torch.log(self.variance)
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        ...
    def map(self):
        ...
    def mean(self):
        ...
    def variance(self):
        ...