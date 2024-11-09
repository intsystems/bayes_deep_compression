import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional, Any
import torch.nn as nn
import torch.distributions as td
from torch.distributions import constraints
from src.methods.bayes.base.net import BayesModule
from src.methods.bayes.base.net import del_attr, set_attr
from torch.distributions.utils import _standard_normal, broadcast_all
from numbers import Number, Real
from torch.types import _size

class BaseBayesVarModule(BayesModule): 
    def __init_posterior_params(self, p, posterior_params: nn.ParameterDict) -> None:
        ...
    def __init_posterior_params_name(self) -> nn.ParameterDict:
        ...
    def __init_prior_params(self, p, prior_params: Optional[nn.ParameterDict]) -> None:
        ...
    def __init_prior_params_name(self) -> Optional[nn.ParameterDict]:
        ...
    def __init__(self, module: nn.Module) -> None:
        posterior_params: nn.ParameterDict = self.__init_posterior_params_name()
        prior_params: Optional[nn.ParameterDict] = self.__init_prior_params_name()
        dropout_mask = nn.ParameterList() 
        index_to_name = []
        i = 0
        for (name, p) in list(module.named_parameters()):
            p.requires_grad = False
            index_to_name.append(name)
            self.__init_posterior_params(p, posterior_params)
            self.__init_prior_params(p, posterior_params)
            dropout_mask.append(nn.Parameter(torch.ones_like(p), requires_grad = False))
            i += 1
        
        super().__init__(module=module, index_to_name = index_to_name, posterior_params = posterior_params, prior_params=prior_params) 
        self.dropout_mask = dropout_mask
    def __prune_condition(self, scale_alphas_log: torch.Tensor, threshold: float) -> torch.Tensor:
        ...
    def __prune_args(self, i:int) -> dict[str, Any]:
        ...

    def prune(self, threshold: float = -2.2) -> None:
        for i in range(len(self.dropout_mask)):
            self.dropout_mask[i].data = 1.0 * self.__prune_condition(self.__prune_args(i), threshold = threshold)
    def prune_stats(self) -> int:
        prune_cnt = 0
        for i in range(len(self.dropout_mask)):
            prune_cnt += (1 - self.dropout_mask[i]).sum()
        return prune_cnt
    def total_params(self) -> int:
        prune_cnt = 0
        for i in range(len(self.dropout_mask)):
            prune_cnt += self.dropout_mask[i].shape.numel()
        return prune_cnt
    def __map(self, scale_mus: torch.Tensor, dropout_mask: torch.Tensor, param_mus: torch.Tensor) -> torch.Tensor:
        ...
    def __map_args(self, i:int) -> dict[str, Any]:
        ...
    def set_map_params(self) -> None: 
        for i in range(len(self.posterior_params.scale_mus)):
            pt = self.__map(self.__map_args(i))
            #pt = torch.nn.Parameter(pt.to_sparse())
            del_attr(self.base_module, self.index_to_name[i].split("."))
            set_attr(self.base_module, self.index_to_name[i].split("."), pt)

class LogNormVarDist(td.distribution.Distribution):
    arg_constraints = {"param_mus": constraints.real, "param_std_log": constraints.real,\
                       "scale_mus": constraints.real, "scale_alphas_log": constraints.real}
    def __init__(self, param_mus: torch.Tensor | Number, param_std_log: torch.Tensor | Number,\
                  scale_mus: torch.Tensor | Number, scale_alphas_log: torch.Tensor | Number, validate_args = None):
        self.param_mus = param_mus
        self.param_std_log = param_std_log
        self.scale_mus = scale_mus
        self.scale_alphas_log = scale_alphas_log
        self.param_mus, self.param_std, self.scale_mus, self.scale_alphas_log = \
            broadcast_all(param_mus, param_std_log, scale_mus, scale_alphas_log)
        if isinstance(self.param_mus, Number) and isinstance(self.param_std, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.param_mus.size()
        super().__init__(batch_shape, validate_args=validate_args)
    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        param_epsilons = _standard_normal(shape, dtype=self.param_mus.dtype, device=self.param_mus.device)
        scale_epsilons = _standard_normal(shape, dtype=self.scale_mus.dtype, device=self.scale_mus.device)
                # calculate sample using reparametrization
        scale_sample = self.scale_mus + scale_epsilons * (self.scale_mus) * torch.sqrt(torch.exp(self.scale_alphas_log.data))
        param_sample = scale_sample * (self.param_mus + param_epsilons * torch.exp(self.param_std_log))
        return param_sample

class BayesVarLogNormModule(BaseBayesVarModule): 
    def __init_posterior_params(self, p, posterior_params: nn.ParameterDict) -> None:
        posterior_params.param_mus.append(nn.Parameter(p, requires_grad = True))
        posterior_params.param_std_log.append(nn.Parameter(torch.log(torch.Tensor(p.shape).uniform_(1e-8, 1e-2)), requires_grad = True)) #(0, 0.01)
        posterior_params.scale_mus.append(nn.Parameter(torch.ones_like(p), requires_grad = True))
        posterior_params.scale_alphas_log.append(nn.Parameter(torch.Tensor(p.shape).uniform_(-4, -2), requires_grad = True)) #(-4, -2)
    def __init_posterior_params_name(self) -> nn.ParameterDict:
        return nn.ParameterDict({'param_mus': nn.ParameterList(), 'param_std_log': nn.ParameterList(),
                       'scale_mus': nn.ParameterList(), 'scale_alphas_log': nn.ParameterList()})
    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)
        self.posterior_distribution_cls = LogNormVarDist
        self.prior_distribution_cls = None
    def __prune_condition(self, scale_alphas_log: torch.Tensor, threshold: float) -> torch.Tensor:
        return scale_alphas_log <= threshold
    def __prune_args(self, i:int) -> dict[str, Any]:
        return {'scale_alphas_log': self.posterior_params["scale_alphas_log"][i]}
    def __map(self, scale_mus: torch.Tensor, dropout_mask: torch.Tensor, param_mus: torch.Tensor) -> torch.Tensor:
        return scale_mus * dropout_mask * param_mus
    def __map_args(self, i:int) -> dict[str, Any]:
        return {'scale_mus': self.posterior_params.scale_mus[i],
                 'dropout_mask':self.dropout_mask[i],
                 'param_mus':self.posterior_params.param_mus[i]}