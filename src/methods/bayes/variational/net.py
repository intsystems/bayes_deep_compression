import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional, Any
import torch.nn as nn
from src.methods.bayes.base.net import BayesModule, BaseBayesModuleNet
from src.methods.bayes.base.net import del_attr, set_attr
from src.methods.bayes.variational.distribution import LogNormVarDist

from torch.types import _size

class BaseBayesVarModule(BayesModule): 

    def flush_weights(self) -> None:
        for (name, p) in list(self.base_module.named_parameters()):
            del_attr(self.base_module, name.split("."))
            set_attr(self.base_module, name.split("."), torch.zeros_like(p))

    def __init__(self, module: nn.Module) -> None: 
        super().__init__(module=module) 
        self.dropout_mask: dict[str, torch.tensor] = {}
        for (name, p) in list(module.named_parameters()):
            self.dropout_mask[name] = torch.ones_like(p)
        self.flush_weights()
    def prune(self, threshold: float = 2.2) -> None:
        for param_name, dist in self.posterior.items():
            self.dropout_mask[param_name].data = 1.0 * (dist.log_z_test() >= threshold)
    def prune_stats(self) -> int:
        prune_cnt = 0
        for dropout in self.dropout_mask.values():
            prune_cnt += (1 - dropout).sum()
        return prune_cnt
    def total_params(self) -> int:
        out = sum(p.numel() for p in self.dropout_mask.values())
        return out
    def set_map_params(self, prune = True) -> None: 
        for param_name, dist in self.posterior.items():
            pt = dist.map()
            if prune == True:
                pt = pt * self.dropout_mask[param_name]
            pt = torch.nn.Parameter(pt)
            #pt = torch.nn.Parameter(pt.to_sparse())
            set_attr(self.base_module, param_name.split("."), pt)


class VarBayesModuleNet(BaseBayesModuleNet):
    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        super().__init__(base_module, module_list)
    @property
    def posterior_params(self) -> dict[str, dict[str, nn.Parameter]]:
        return self.get_params('posterior')
    @property
    def prior_params(self) -> dict[str, dict[str, nn.Parameter]]:
        return self.get_params('prior')
    def prune_stats(self) -> int:
        pruned_cnt = 0 
        for module in self.module_list:
            if isinstance(module, BayesModule):
                pruned_cnt += module.prune_stats()
        return pruned_cnt
    def total_params(self) -> int:
        return sum(module.total_params() for module in self.module_list)
    def set_map_params(self, prune: bool = True) -> None:
        for module in self.module_list:
            if isinstance(module, BayesModule):
                module.set_map_params(prune = True)

class BayesVarLogNormModule(BaseBayesVarModule): 
    def __init__(self, module: nn.Module) -> None:
        self.posterior_distribution_cls = LogNormVarDist
        self.prior_distribution_cls = None
        super().__init__(module)

