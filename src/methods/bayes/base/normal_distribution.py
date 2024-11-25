import torch
import torch.nn as nn 
import torch.distributions as td
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions import constraints
from numbers import Number
from torch.types import _size
from .distribution import ParamDist

class Normal(td.Normal, ParamDist):    
    def __init__(self, loc, log_scale, validate_args=None) -> None:
        """This is basically Normal distribution, but it accepts logarithm of the variance
                instead of the standard deviation (a.k.a. scale)

        Args:
            loc (_type_): inital value for the mean Parameter
            log_scale (_type_): inital value for the logarithm of the $\sigma^2$ Parameter
            validate_args (_type_, optional): . Defaults to None.
        """
        # make learnable Parameters
        # td.Normal will have loc as an attribute
        loc = nn.Parameter(loc)
        self.log_scale = nn.Parameter(log_scale)

        td.Normal.__init__(loc, torch.exp(0.5 * log_scale), validate_args)

    def get_params(self) -> dict[str, nn.Parameter]: 
        return {
            "loc": self.loc,
            "log_scale": self.log_scale
        }
    
    def prob(self, weights):
        return torch.exp(self.log_prob(weights))

    def log_z_test(self):
        return torch.log(self.mean) - torch.log(self.variance)
    
    def map(self):
        return self.mean
    
    @classmethod
    def from_parameter(cls, p: nn.Parameter) -> 'ParamDist':
        # log_scale = -1 initially
        return cls(p, torch.ones_like(p) - 2)