from typing import Any

import torch
import torch.nn as nn

from torch.distributions import Normal
from .net import BaseBayesVarModule


class NormalWithLogStd(Normal):
    def __init__(self, loc, log_scale, validate_args=None):
        """ This is basically Normal distribution, but it accepts logarithm of the variance
                instead of the standard deviation (a.k.a. scale)

        Args:
            log_scale: logarithm of the $\sigma^2$
        """
        super().__init__(loc, torch.exp(0.5 * log_scale), validate_args)


class BayesVarNormModule(BaseBayesVarModule):
    """ Envelope for nn.Modules with the same normal prior on all scalar paramters and factorized normal
            distributions as the variational distibution on paramters. The prior is not required here as
            its optimal form can be computed analytically.
    """
    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)
        self.posterior_distribution_cls = NormalWithLogStd
        self.prior_distribution_cls = None

    def init_posterior_params_name(self) -> nn.ParameterDict:
        # we have only mean and variance here
        return nn.ParameterDict({'loc': nn.ParameterList(), 'log_scale': nn.ParameterList()})
    
    def init_posterior_params(self, p, posterior_params: nn.ParameterDict) -> None:
        posterior_params.loc.append(nn.Parameter(p, requires_grad = True))
        posterior_params.log_scale.append(nn.Parameter(-10 + torch.rand_like(p), requires_grad = True))

    def prune_args(self, i:int) -> dict[str, Any]:
        # pruning is based on both mean and variance
        return {
            'loc': self.posterior_params.loc[i],
            'log_scale': self.posterior_params.log_scale[i]
        }
    
    def prune_condition(self, loc: torch.Tensor, log_scale: torch.Tensor, threshold: float) -> torch.Tensor:
        # as suggeseted by Graves
        return torch.abs(
                loc / torch.exp(log_scale)
        ) >= 0.83
    
    def map(self, loc: torch.Tensor, dropout_mask: torch.Tensor) -> torch.Tensor:
        return (loc * dropout_mask).detach()

    def map_args(self, i:int) -> dict[str, Any]:
        return {
            'loc': self.posterior_params.loc[i],
            'dropout_mask': self.dropout_mask[i]
        }

