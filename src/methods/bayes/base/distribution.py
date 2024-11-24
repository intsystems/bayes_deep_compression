from abc import ABC, abstractmethod

import torch
import torch.distributions as D
import torch.nn as nn
from torch.types import _size


class ParamDist(D.distribution.Distribution, ABC):
    @classmethod
    @abstractmethod
    def from_parameter(self, p: nn.Parameter) -> "ParamDist":
        """
        Default initialization of ParamDist forom parameters of nn.Module

        Args:
            p (nn.Parameter): paramaters for which ParamDist should be created.
        """
        ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_params(self) -> dict[str, nn.Parameter]:
        """
        Returns dictionary of parameters that should be registered as parameters at nn.Module.
        """
        ...

    @abstractmethod
    def log_prob(self, weights):
        """ 
        Returns logarithm of probability density function of distibution evaluated at weights.

        Args:
            weights: the point at which probability should be evaluated.
        """
        ...

    @abstractmethod
    def log_z_test(self):
        """ 
        Returns parameter which is used to be compared with threshold to estimate 
        wether this parameter should be pruned. By default it is logarithm of z_test
        or equivalent of it. log_z_test = log(abs(mean)) - log(variance)
        """
        return torch.log(torch.abs(self.mean)) - torch.log(self.variance)

    @abstractmethod
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """ 
        Returns parameters sampled using reparametrization trick, so they could be used for 
        gradient estimation
        """
        ...

    @property
    @abstractmethod
    def map(self):
        """ 
        Returns mode of the distibution. It has a sense of maximum aposteriori estimation
        for bayessian nets.
        """
        ...

    @property
    @abstractmethod
    def mean(self):
        """ 
        Returns mean of the distibution. It has a sense of non-bias estimation
        for bayessian nets.
        """
        ...

    @abstractmethod
    def variance(self):
        """ 
        Returns variance of the distibution. It has a sense of error estimation
        for bayessian nets and assumed to be used in prunning.
        """
        ...
