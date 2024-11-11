from abc import abstractmethod
from typing import Generic, TypeVar, Optional, Any
import torch
import torch.nn
import torch.nn as nn
import torch.distributions as td
import copy
from torch.autograd import Variable
from src.methods.bayes.base.distribution import ParamDist


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
        
class BayesModule(nn.Module):
    prior_distribution_cls: Optional[ParamDist]
    posterior_distribution_cls: type[ParamDist]
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.posterior: dict[str, ParamDist] = {}
        self.prior: dict[str, ParamDist] = {}
        i = 0
        #Itereate to create posterior and prior dist for each parameter
        for (name, p) in list(module.named_parameters()):
            p.requires_grad = False
            self.posterior[name] = self.posterior_distribution_cls.from_parameter(p)
            self.prior[name] = None
            if self.prior_distribution_cls is not None:
                self.prior[name] = self.prior_distribution_cls.from_parameter(p)
            i += 1
        self.base_module = module
        
        #key - weight_name, value - distribution_args: nn.ParameterDict  
        self.posterior_params = nn.ParameterList()
        for dist in self.posterior.values():
            self.posterior_params.append(nn.ParameterDict(dist.get_params()))
        
        self.prior_params = nn.ParameterList()
        if self.prior_distribution_cls is not None:
            for dist in self.prior.values():
                self.prior_params.append(dist.get_params())
    def sample(self) -> dict[str, nn.Parameter]:
        param_sample_dict: dict[str, nn.Parameter] = {}
        for (param_name, param_posterior) in self.posterior.items():
            param_sample = param_posterior.rsample()
            param_sample_dict[param_name] = nn.Parameter(param_sample)
            del_attr(self.base_module, param_name.split("."))
            set_attr(self.base_module, param_name.split("."), param_sample)
        return param_sample_dict 
    @property
    def weights(self) -> dict[nn.Parameter]:
        return dict(self.base_module.named_parameters())
    @property
    def device(self):
        return next(self.parameters()).device
    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)
    def eval(self) -> None:
        self.base_module.eval()
    def train(self) -> None:
        self.base_module.train()
    
    def prune(self, *args, **kwargs) -> None:
        ...

class BaseBayesModuleNet(nn.Module):
    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        super().__init__()
        self.__dict__['base_module'] = base_module
        self.module_list = module_list
    def sample(self):
        param_sample_dict: dict[str, nn.Parameter] = {}
        for module in self.module_list:
            if isinstance(module, BayesModule):
                parameter_dict = module.sample()
                param_sample_dict.update(parameter_dict)
        return param_sample_dict, self.base_module
    
    def get_distr_params(self, param_type_name: str) -> dict[str, dict[str, nn.Parameter]]:
        params_dict = {}
        for module in self.module_list:
            if isinstance(module, BayesModule):
                for key in (getattr(module, param_type_name)):
                    print(key)
                    parameter_dict = getattr(module, param_type_name)[key]
                    print(parameter_dict)
                    params_dict.setdefault(key, {})
                    params_dict[key].update(parameter_dict.get_distr_params())
        return params_dict
    @property
    def weights(self) -> dict[nn.Parameter]:
        weights = {}
        for module in self.module_list:
            module_posterior = None
            if isinstance(module, BayesModule):
                module_posterior = module.weights
            weights.update(module_posterior)
        return weights
    @property
    def device(self):
        return next(self.parameters()).device
    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)
    def flush_weights(self) -> None:
        ...
    def sample_model(self) -> nn.Module:
        self.sample()
        model = copy.deepcopy(self.base_module)
        self.flush_weights()
        return model
    def eval(self) -> None:
        self.base_module.eval()
    def train(self) -> None:
        self.base_module.train()
    def prune(self, prune_args: dict[list[str, Any]] | list[str, Any]) -> None:
        for i in range(len(self.module_list)):
            if isinstance(self.module_list[i], BayesModule):
                prune_argument = prune_args
                if isinstance(prune_argument, list):
                    prune_argument = prune_argument[i]
                self.module_list[i].prune(**prune_argument)
    @property
    def posterior(self):
        #self.params = {mus: , sigmas: }
        posteriors = {}
        for module in self.module_list:
            module_posterior = None
            if isinstance(module, BayesModule):
                module_posterior = module.posterior
            posteriors.update(module_posterior)
        return posteriors
    @property
    def prior(self):
        #self.params = {mus: , sigmas: }
        priors = {}
        for module in self.module_list:
            module_prior = None
            if isinstance(module, BayesModule):
                module_prior = module.prior
            priors.update(module_prior)
        return priors