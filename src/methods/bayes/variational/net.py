import torch
import torch.nn as nn
from src.methods.bayes.base.net import BaseBayesModuleNet, BayesModule
from src.methods.bayes.variational.distribution import LogUniformVarDist, NormalDist, NormalReparametrizedDist
from src.utils.attribute import del_attr, set_attr


class BaseBayesVarModule(BayesModule):
    is_posterior_trainable = True

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
    """Envelope for nn.Modules with the same normal prior on all scalar paramters and factorized normal
    distributions as the variational distibution on paramters. The prior is not required here as
    its optimal form can be computed analytically.
    """

    def __init__(self, module: nn.Module) -> None:
        self.posterior_distribution_cls = NormalReparametrizedDist
        self.prior_distribution_cls = NormalDist
        self.is_prior_trainable = False
        super().__init__(module)
