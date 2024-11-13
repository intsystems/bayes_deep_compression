import torch
import torch.nn as nn
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions import constraints
import torch.distributions as D
from torch.types import _size
from src.methods.bayes.base.distribution import ParamDist
from numbers import Number
import torch.nn.functional as F

class LogUniformVarDist(ParamDist):
    arg_constraints = {
        "param_mus": constraints.real,
        "param_std_log": constraints.real,
        "scale_mus": constraints.real,
        "scale_alphas_log": constraints.real,
    }

    @classmethod
    def from_parameter(cls, p: nn.Parameter) -> "LogUniformVarDist":

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
        validate_args=None,
    ):
        self.param_mus: nn.Parameter = nn.Parameter(param_mus)
        self.param_std_log: nn.Parameter = nn.Parameter(param_std_log)
        self.scale_mu: nn.Parameter = nn.Parameter(scale_mus)
        self.scale_alphas_log: nn.Parameter = nn.Parameter(scale_alphas_log)
        self.param_mus, self.param_std, self.scale_mus, self.scale_alphas_log = (
            broadcast_all(param_mus, param_std_log, scale_mus, scale_alphas_log)
        )
        batch_shape = self.param_mus.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def get_params(self) -> dict[str, nn.Parameter]:
        return {
            "param_mus": self.param_mus,
            "param_std_log": self.param_std_log,
            "scale_mus": self.scale_mus,
            "scale_alphas_log": self.scale_alphas_log,
        }

    def map(self):
        return self.scale_mus * self.param_mus
    @property
    def mean(self):
        return self.scale_mus * self.param_mus
    @property
    def variance(self):
        raise NotImplementedError()

    def prob(self, weights):
        raise NotImplementedError()

    def log_prob(self, weights):
        raise NotImplementedError()

    def log_z_test(self):
        return -self.scale_alphas_log

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        param_epsilons = _standard_normal(
            shape, dtype=self.param_mus.dtype, device=self.param_mus.device
        )
        scale_epsilons = _standard_normal(
            shape, dtype=self.scale_mus.dtype, device=self.scale_mus.device
        )
        # calculate sample using reparametrization
        scale_sample = self.scale_mus + scale_epsilons * (self.scale_mus) * torch.sqrt(
            torch.exp(self.scale_alphas_log.data)
        )
        param_sample = scale_sample * (
            self.param_mus + param_epsilons * torch.exp(self.param_std_log)
        )
        return param_sample


class NormalDist(D.Normal, ParamDist):
    @classmethod
    def from_parameter(self, p: nn.Parameter) -> ParamDist:
        loc = nn.Parameter(p.new(p.size()).zero_(), requires_grad=False)
        scale = nn.Parameter(p.new(p.size()).zero_() + 0.1, requires_grad=False)
        return NormalDist(loc, scale)
    
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale,validate_args=validate_args)
        # self.loc, self._scale = D.broadcast_all(loc, scale)
        # if isinstance(loc, Number) and isinstance(scale, Number):
        #     batch_shape = torch.Size()
        # else:
        #     batch_shape = self.loc.size()
        # D.Distribution.__init__(self, batch_shape, validate_args=validate_args)

    def get_params(self) -> dict[str, nn.Parameter]:
        return {"loc": self.loc, "scale": self.scale}
    def map(self):
        raise NotImplementedError()
    def log_z_test(self):
        return -self.scale_alphas_log
    def prob(self, weights):
        raise NotImplementedError()


class NormalReparametrizedDist(D.Normal, ParamDist):
    @classmethod
    def from_parameter(self, p: nn.Parameter) -> ParamDist:
        loc = nn.Parameter(p, requires_grad=True)
        # scale4softplus = nn.Parameter(p.new(p.size()).rand_(), requires_grad=True)
        scale4softplus = nn.Parameter(torch.Tensor(p.shape).uniform_(-4, -2) , requires_grad=True)
        return NormalReparametrizedDist(loc, scale4softplus)

    def __init__(self, loc, scale, validate_args=None):

        self.loc, self._scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        D.Distribution.__init__(self, batch_shape, validate_args=validate_args)

    @property
    def scale(self):
        return F.softplus(self._scale)

    def get_params(self) -> dict[str, nn.Parameter]:
        return {"loc": self.loc, "scale": self._scale}

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        w = self.loc + eps * self.scale
        return w

    def log_z_test(self):
        return torch.log(torch.abs(self.mean)) - torch.log(self.variance)

    def map(self):
        return self.loc
    def prob(self, weights):
        raise NotImplementedError()