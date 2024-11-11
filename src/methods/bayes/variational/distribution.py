import torch
import torch.nn as nn 
import torch.distributions as td
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions import constraints
from numbers import Number
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

class LogNormVarDist(ParamDist):
    arg_constraints = {"param_mus": constraints.real, "param_std_log": constraints.real,\
                       "scale_mus": constraints.real, "scale_alphas_log": constraints.real}
    @classmethod
    def from_parameter(cls, p: nn.Parameter) -> 'LogNormVarDist':
        
        param_mus = (nn.Parameter(p, requires_grad = True))
        param_std_log =(nn.Parameter(torch.log(torch.Tensor(p.shape).uniform_(1e-8, 1e-2)), requires_grad = True)) #(0, 0.01)
        scale_mus = (nn.Parameter(torch.ones_like(p), requires_grad = True))
        scale_alphas_log = (nn.Parameter(torch.Tensor(p.shape).uniform_(-4, -2), requires_grad = True)) #(-4, -2)
        return LogNormVarDist(param_mus, param_std_log, scale_mus, scale_alphas_log)
    def __init__(self, param_mus: torch.Tensor , param_std_log: torch.Tensor ,\
                  scale_mus: torch.Tensor, scale_alphas_log: torch.Tensor , validate_args = None):
        self.param_mus: nn.Parameter = nn.Parameter(param_mus)
        self.param_std_log: nn.Parameter  = nn.Parameter(param_std_log)
        self.scale_mu: nn.Parameter = nn.Parameter(scale_mus)
        self.scale_alphas_log: nn.Parameter = nn.Parameter(scale_alphas_log)
        self.param_mus, self.param_std, self.scale_mus, self.scale_alphas_log = \
        broadcast_all(param_mus, param_std_log, scale_mus, scale_alphas_log)
        batch_shape = self.param_mus.size()
        super().__init__(batch_shape, validate_args=validate_args)
    def get_params(self) -> dict[str, nn.Parameter]:
        return {"param_mus": self.param_mus, "param_std_log": self.param_std_log,\
                       "scale_mus": self.scale_mus, "scale_alphas_log": self.scale_alphas_log}
    def map(self):
        return self.scale_mus * self.param_mus
    def mean(self):
        return self.scale_mus * self.param_mus
    def variance(self, weights):
        raise NotImplementedError()
    def prob(self, weights):
        raise NotImplementedError()
    def log_prob(self, weights):
        raise NotImplementedError()
    def log_z_test(self):
        return -self.scale_alphas_log
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        param_epsilons = _standard_normal(shape, dtype=self.param_mus.dtype, device=self.param_mus.device)
        scale_epsilons = _standard_normal(shape, dtype=self.scale_mus.dtype, device=self.scale_mus.device)
                # calculate sample using reparametrization
        scale_sample = self.scale_mus + scale_epsilons * (self.scale_mus) * torch.sqrt(torch.exp(self.scale_alphas_log.data))
        param_sample = scale_sample * (self.param_mus + param_epsilons * torch.exp(self.param_std_log))
        return param_sample