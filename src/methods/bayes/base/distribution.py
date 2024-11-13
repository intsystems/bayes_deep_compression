import torch
import torch.nn as nn
import torch.distributions as D
from torch.types import _size

from abc import abstractmethod, ABC


class ParamDist(D.distribution.Distribution, ABC):
    
    @classmethod
    @abstractmethod
    def from_parameter(self, p: nn.Parameter) -> "ParamDist": ...
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_params(self) -> dict[str, nn.Parameter]: ...
    @abstractmethod
    def prob(self, weights): ...
    @abstractmethod
    def log_prob(self, weights): ...
    @abstractmethod
    def log_z_test(self):
        return torch.log(torch.abs(self.mean)) - torch.log(self.variance)

    @abstractmethod
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor: ...
    @property
    @abstractmethod
    def map(self): ...
    @property
    @abstractmethod
    def mean(self): ...
    @abstractmethod
    def variance(self): ...
