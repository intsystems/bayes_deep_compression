from abc import abstractmethod
from typing import Generic, TypeVar, Optional, Any
import torch
import torch.nn
import torch.nn as nn
import torch.distributions as td
import copy
from torch.autograd import Variable

from src.methods.bayes.base.output import BaseOutputModel

OutputT = TypeVar("OutputT", bound=BaseOutputModel)
ModelT = TypeVar("ModelT")

class MLPBayesModel(torch.nn.Module, Generic[OutputT]):
    def __init__(self, layer_list: list[nn.Linear]):
        self.layer_list = layer_list

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[OutputT]: ...
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
    prior_distribution_cls: Optional[type[td.distribution.Distribution]]
    posterior_distribution_cls: type[td.distribution.Distribution]
    
    def __init__(self, module: nn.Module, index_to_name: list[str], posterior_params: nn.ParameterDict, prior_params: Optional[nn.ParameterDict] = None) -> None:
        super().__init__()
        self.base_module = module
        self.posterior_params = posterior_params
        self.prior_params = prior_params
        self.index_to_name = index_to_name
    @property
    def posterior_params_size(self) -> int:
        return len(next(self.posterior_params.values()))
    @property
    def prior_params_size(self) -> int:
        if(self.prior_params is None):
            return 0
        return len(next(self.posterior_params.values()))
    def sample(self)->None:
        param_sample_list = nn.ParameterList()
        for i in range(self.posterior_params_size):
            posteriror_args = {}
            for arg_name in self.posterior_params:
                posteriror_args[arg_name] = self.posterior_params[arg_name][i]
            param_sample = self.posterior_distribution_cls(**posteriror_args).rsample()
            param_sample_list.append(param_sample)
            del_attr(self.base_module, self.index_to_name[i].split("."))
            set_attr(self.base_module, self.index_to_name[i].split("."), param_sample)
        return param_sample_list
    @property
    def device(self):
        return next(self.parameters()).device
    def forward(self, *args, **kwargs):
        self.sample()
        return self.base_module(*args, **kwargs)
    def eval(self) -> None:
        self.base_module.eval()
    def train(self) -> None:
        self.base_module.train()
    def prune(self, *args, **kwargs) -> None:
        ...
    @property
    def posterior(self) -> td.distribution.Distribution:
        #self.params = {mus: , sigmas: }
        dist = self.posterior_distribution_cls(**self.posterior_params)
        return dist
    @property
    def prior(self) -> Optional[td.distribution.Distribution]:
        #self.params = {mus: , sigmas: }
        if self.prior_distribution_cls is None:
            return None
        dist = self.prior_distribution_cls(**self.prior_params)
        return dist

class BaseBayesModuleNet(nn.Module):
    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        super().__init__()
        self.__dict__['base_module'] = base_module
        self.module_list = module_list
    def sample(self):
        param_sample_list = nn.ParameterList()
        for i in range(len(self.module_list)):
            if isinstance(self.module_list[i], BayesModule):
                parameter_list = self.module_list[i].sample()
                param_sample_list.append(parameter_list)
        return param_sample_list, self.base_module
    def __get_params(self, param_name: str) -> nn.ParameterDict:
        params_dict = nn.ParameterDict()
        for i in range(len(self.module_list)):
            if isinstance(self.module_list[i], BayesModule):
                for key in (getattr(self.module_list[i], param_name)):
                    parameter_list = getattr(self.module_list[i], param_name)[key]
                    if not isinstance(parameter_list, nn.ParameterList):
                        parameter_list = nn.ParameterList(parameter_list)
                    params_dict.setdefault(key, nn.ParameterList())
                    params_dict[key].append(parameter_list)
        return params_dict
    @property
    def posterior_params(self) -> nn.ParameterDict:
        return self.__get_params('posterior_params')
    @property
    def prior_params(self) -> nn.ParameterDict:
        return self.__get_params('prior_params')
    @property
    def posterior_params_size(self) -> int:
        return len(next(self.posterior_params.values()))
    @property
    def prior_params_size(self) -> int:
        if(self.prior_params is None):
            return 0
        return len(next(self.posterior_params.values()))
    @property
    def device(self):
        return next(self.parameters()).device
    def forward(self, *args, **kwargs):
        self.sample()
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
    def posterior(self) -> list[torch.Tensor]:
        #self.params = {mus: , sigmas: }
        posteriors = []
        for i in range(len(self.module_list)):
            module_posterior = None
            if isinstance(self.module_list[i], BayesModule):
                module_posterior = self.module_list[i].posterior
            posteriors.append(module_posterior)
        return posteriors
    @property
    def prior(self) -> list[Optional[torch.Tensor]]:
        #self.params = {mus: , sigmas: }
        priors = []
        for i in range(len(self.module_list)):
            module_prior = None
            if isinstance(self.module_list[i], BayesModule):
                module_prior = self.module_list[i].prior
            priors.append(module_prior)
        return priors