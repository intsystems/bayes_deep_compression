from numbers import Number
from typing import Optional

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from src.methods.bayes.base.distribution import ParamDist
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.types import _size


class LogUniformVarDist(ParamDist):
    arg_constraints = {
        "param_mus": constraints.real,
        "param_std_log": constraints.real,
        "scale_mus": constraints.real,
        "scale_alphas_log": constraints.real,
    }

    @classmethod
    def from_parameter(cls, p: nn.Parameter) -> "LogUniformVarDist":
        """
        Default initialization of LogUniformVarDist forom parameters of nn.Module

        Args:
            p (nn.Parameter): paramaters for which ParamDist should be created.
        """
        param_mus = nn.Parameter(p, requires_grad=True)
        param_std_log = nn.Parameter(
            torch.log(torch.Tensor(p.shape).uniform_(1e-8, 1e-2)), requires_grad=True
        )  # (0, 0.01)
        scale_mus = nn.Parameter(torch.ones_like(p), requires_grad=True)
        scale_alphas_log = nn.Parameter(
            torch.Tensor(p.shape).uniform_(-4, -2), requires_grad=True
        )  # (-4, -2)
        return LogUniformVarDist(param_mus, param_std_log, scale_mus, scale_alphas_log)

    def __init__(
        self,
        param_mus: torch.Tensor,
        param_std_log: torch.Tensor,
        scale_mus: torch.Tensor,
        scale_alphas_log: torch.Tensor,
        validate_args: Optional[bool] =None,
    ):
        r"""_summary_

        Args:
            param_mus: $\mu$ parameter of distribution
            param_std_log: $\log(\sigma)$ parameter of distribution
            scale_mus: $\mu$ parameter scale of distribution
            scale_alphas_log: $\alpha$ parameter scale of distribution
            validate_args: alias fo validate_args of torch.distributions.sistribution
        """
        self.param_mus: nn.Parameter = nn.Parameter(param_mus)
        r"""$\mu$ parameter of distribution"""
        self.param_std_log: nn.Parameter = nn.Parameter(param_std_log)
        r"""$\log(\sigma))$ parameter of distribution"""
        self.scale_mus: nn.Parameter = nn.Parameter(scale_mus)
        r"""$\mu$ parameter scale of distribution"""
        self.scale_alphas_log: nn.Parameter = nn.Parameter(scale_alphas_log)
        r"""$\alpha$ parameter scale of distribution"""

        # (
        #     self.param_mus,
        #     self.param_std_log,
        #     self.scale_mus,
        #     self.scale_alphas_log,
        # ) = broadcast_all(
        #     self.param_mus, self.param_std_log, self.scale_mus, self.scale_alphas_log
        # )

        batch_shape = self.param_mus.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def get_params(self) -> dict[str, nn.Parameter]:
        """
        Return all parameters that should be registered as named parameters of nn.Module.
        {"param_mus", "param_std_log", "scale_mus", "scale_alphas_log"}

        Returns:
            dict[str, nn.Parameter]: parameters that should be registered as named parameters of nn.Module
        """
        return {
            "param_mus": self.param_mus,
            "param_std_log": self.param_std_log,
            "scale_mus": self.scale_mus,
            "scale_alphas_log": self.scale_alphas_log,
        }

    @property
    def map(self) -> torch.Tensor:
        """
        Return MAP(if we speaks about posterior distribution) or MLE estimation of parameters

        Returns:
            torch.Tensor: MAP estimation of parameters
        """
        return self.scale_mus * self.param_mus

    @property
    def mean(self) -> torch.Tensor:
        """
        Return mean estimation of parameters

        Returns:
            torch.Tensor: mean value of parameters
        """
        return self.scale_mus * self.param_mus

    @property
    def variance(self) -> torch.Tensor:
        """
        Return variance of parameters

        Returns:
            torch.Tensor: variance of parameters
        """
        return torch.FloatTensor([1])

    def log_prob(self, weights) -> torch.Tensor:
        """
        Return logarithm probability at weights

        Returns:
            torch.Tensor: logarithm probability at weights
        """
        return torch.FloatTensor([-1])

    def log_z_test(self) -> torch.Tensor:
        """
        Return logarithm of z-test statistic. For numerical stability it is
        -self.scale_alphas_log. This value is compared with threshold to
        consider should be parameter pruned or not.

        Returns:
            torch.Tensor: logarithm of z-test statistic
        """
        return -self.scale_alphas_log

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """
        Returns parameters sampled using reparametrization trick, so they could be used for
        gradient estimation

        Returns:
            torch.Tensor: sampled parameters
        """
        shape = self._extended_shape(sample_shape)
        param_epsilons = _standard_normal(
            shape, dtype=self.param_mus.dtype, device=self.param_mus.device
        )
        scale_epsilons = _standard_normal(
            shape, dtype=self.scale_mus.dtype, device=self.scale_mus.device
        )
        # calculate sample using reparametrization
        scale_sample = self.scale_mus + scale_epsilons * (self.scale_mus) * torch.sqrt(
            torch.exp(self.scale_alphas_log)
        )
        param_sample = scale_sample * (
            self.param_mus + param_epsilons * torch.exp(self.param_std_log)
        )
        return param_sample


class NormalReparametrizedDist(D.Normal, ParamDist):
    @classmethod
    def from_parameter(self, p: nn.Parameter) -> "NormalReparametrizedDist":
        """
        Default initialization of NormalReparametrizedDist forom parameters of nn.Module

        Args:
            p (nn.Parameter): paramaters for which ParamDist should be created.
        """
        loc = nn.Parameter(p, requires_grad=True)
        # scale4softplus = nn.Parameter(p.new(p.size()).rand_(), requires_grad=True)
        scale4softplus = nn.Parameter(
            torch.Tensor(p.shape).uniform_(-4, -2), requires_grad=True
        )
        return NormalReparametrizedDist(loc, scale4softplus)

    def __init__(self, loc, log_scale, validate_args=None) -> None:
        r"""_summary_

        Args:
            loc (torch.Tensor): $\mu$ parameter of normal distribution
            log_scale (torch.Tensor): $\log(\sigma)$ parameter of distribution
        """

        loc, log_scale = broadcast_all(loc, log_scale)

        self.loc = nn.Parameter(loc)
        """$\mu$ parameter of normal distribution"""
        self._scale = nn.Parameter(log_scale)
        """$\log(\sigma)$ parameter of distribution"""

        if isinstance(loc, Number) and isinstance(log_scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        D.Distribution.__init__(self, batch_shape, validate_args=validate_args)

    @property
    def scale(self) -> torch.Tensor:
        """
        Return scale parameter of normal distribution

        Returns:
            torch.Tensor: scale parameter of normal distribution
        """
        return F.softplus(self._scale)

    @property
    def log_scale(self) -> torch.Tensor:
        """
        Return log-scale parameter of normal distribution

        Returns:
            torch.Tensor: log-scale parameter of normal distribution
        """
        return torch.log(self.scale)

    def get_params(self) -> dict[str, nn.Parameter]:
        """
        Return all parameters that should be registered as named parameters of nn.Module.
        {"loc", "scale_"}

        Returns:
            dict[str, nn.Parameter]: parameters that should be registered as named parameters of nn.Module
        """
        return {"loc": self.loc, "scale": self._scale}

    # Legacy
    # def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
    #     shape = self._extended_shape(sample_shape)
    #     eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
    #     w = self.loc + eps * self.scale
    #     return w

    def log_z_test(self) -> torch.Tensor:
        """
        Return logarithm of z-test statistic. This value is compared with threshold to
        consider should be parameter pruned or not.

        Returns:
            torch.Tensor: logarithm of z-test statistic
        """
        return torch.log(torch.abs(self.mean)) - torch.log(self.variance)

    @property
    def map(self) -> torch.Tensor:
        """
        Return MAP(if we speaks about posterior distribution) or MLE estimation of parameters

        Returns:
            torch.Tensor: MAP estimation of parameters
        """
        return self.loc
