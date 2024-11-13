from abc import ABC, abstractmethod

import torch
import torch.distributions as D
import torch.nn as nn
from torch.types import _size


class ParamDist(D.distribution.Distribution, ABC):
    @classmethod
    @abstractmethod
    def from_parameter(self, p: nn.Parameter) -> "ParamDist":
        ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_params(self) -> dict[str, nn.Parameter]:
        ...

    @abstractmethod
    def log_prob(self, weights):
        ...

    @abstractmethod
    def log_z_test(self):
        return torch.log(torch.abs(self.mean)) - torch.log(self.variance)

    @abstractmethod
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def map(self):
        """ Returns mode of the distibution. It has a sense of maximum aposteriori estimation
             for bayessian nets.
        """
        ...

    @property
    @abstractmethod
    def mean(self):
        ...

    @abstractmethod
    def variance(self):
        ...
