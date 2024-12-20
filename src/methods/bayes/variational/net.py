import torch
import torch.nn as nn
from src.methods.bayes.base.net import BaseBayesNet, BayesLayer
from src.methods.bayes.variational.distribution import (
    LogUniformVarDist,
    NormalReparametrizedDist,
)
from src.utils.attribute import del_attr, set_attr


class BaseBayesVarLayer(BayesLayer):
    """Base Envelope for nn.Modules with some prior and posterior
    distributions as the variational distibution on paramters. This module should be used
    as parent class for all variational methods.
    """

    is_posterior_trainable = True

    def flush_weights(self) -> None:
        """
        This method simply zeros all weights that will be calculated by this layer and
        set them as tensors, so it will work properly when layer is initialized.
        """
        for name, p in list(self.base_module.named_parameters()):
            del_attr(self.base_module, name.split("."))
            set_attr(self.base_module, name.split("."), torch.zeros_like(p))

    def __init__(self, module: nn.Module) -> None:
        """_summary_

        Args:
            module (nn.Module): custom Module layer which is going to be converted to BaseBayesVarLayer
        """
        super().__init__(module=module)
        self.flush_weights()


class VarBayesNet(BaseBayesNet):
    """The whole net that contains all layers that should be tranfomed to bayesian modules.
    This net is used for variational bayesian methods.
    """

    def __init__(self, base_module: nn.Module, module_dict: dict[str, nn.Module]):
        """_summary_

        Args:
            base_module (nn.Module): custom Module which is going to have some BayesModule as submodules
            module_dict (dict[str, nn.Module]): all submodules of the base_module supposed to be trained. This
                may be nn.Module or BayesModule. Such division is required because base_module is not
                registred as Module in this class.
        """
        super().__init__(base_module, module_dict)


class LogUniformVarLayer(BaseBayesVarLayer):
    """Envelope for nn.Modules with the same LogUniform prior on all scalar paramters and factorized normal
    distributions as the variational distibution on paramters. The prior is not required here as
    its optimal form can be computed analytically.
    """

    def __init__(self, module: nn.Module) -> None:
        """_summary_

        Args:
            module (nn.Module): custom Module layer which is going to be converted to LogUniformVarLayer
        """
        self.posterior_distribution_cls = LogUniformVarDist
        self.prior_distribution_cls = None
        self.is_prior_trainable = False
        super().__init__(module)


class NormalVarBayesLayer(BaseBayesVarLayer):
    """Envelope for nn.Modules with the same normal prior on all scalar paramters and factorized normal
    distributions as the variational distibution on paramters. The prior is not required here as
    its optimal form can be computed analytically.
    """

    def __init__(self, module: nn.Module) -> None:
        """_summary_

        Args:
            module (nn.Module): custom Module layer which is going to be converted to NormalVarBayesLayer
        """
        self.posterior_distribution_cls = NormalReparametrizedDist
        self.prior_distribution_cls = None
        self.is_prior_trainable = False
        super().__init__(module)
