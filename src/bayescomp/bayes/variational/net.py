import torch
import torch.nn as nn
from bayescomp.bayes.base.net import BayesModule, BaseBayesModuleNet
from bayescomp.utils.attribute import del_attr, set_attr
from bayescomp.bayes.variational.distribution import LogUniformVarDist, NormalDist, NormalReparametrizedDist


class BaseBayesVarModule(BayesModule):

    def flush_weights(self) -> None:
        for name, p in list(self.base_module.named_parameters()):
            del_attr(self.base_module, name.split("."))
            set_attr(self.base_module, name.split("."), torch.zeros_like(p))

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module=module)
        self.flush_weights()

class VarBayesModuleNet(BaseBayesModuleNet):
    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        super().__init__(base_module, module_list)

    @property
    def posterior_params(self) -> dict[str, dict[str, nn.Parameter]]:
        return self.get_params("posterior")

    @property
    def prior_params(self) -> dict[str, dict[str, nn.Parameter]]:
        return self.get_params("prior")


class LogUniformVarBayesModule(BaseBayesVarModule):
    def __init__(self, module: nn.Module) -> None:
        self.posterior_distribution_cls = LogUniformVarDist
        self.prior_distribution_cls = None
        self.is_prior_trainable = False
        super().__init__(module)

class NormalVarBayesModule(BaseBayesVarModule): 
    def __init__(self, module: nn.Module) -> None:
        self.posterior_distribution_cls = NormalReparametrizedDist
        self.prior_distribution_cls = NormalDist
        self.is_prior_trainable = False
        super().__init__(module)