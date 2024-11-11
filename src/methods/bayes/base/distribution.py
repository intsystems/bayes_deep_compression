import torch
import torch.nn as nn
import torch.distributions as td
from torch.types import _size

from abs import abstractmethod


class ParamDist(td.distribution.Distribution):
    @classmethod
    def from_parameter(self, p: nn.Parameter) -> "ParamDist": ...
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_params(self) -> dict[str, nn.Parameter]: ...
    @abstractmethod
    def prob(self, weights): ...
    @abstractmethod
    def log_prob(self, weights): ...
    def log_z_test(self):
        return torch.log(self.mean()) - torch.log(self.variance)

    @abstractmethod
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor: ...
    @abstractmethod
    def map(self): ...
    @abstractmethod
    def mean(self): ...
    def variance(self): ...
