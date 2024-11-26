import torch.nn as nn
from src.methods.bayes.base.distribution import ParamDist
from src.methods.bayes.base.net_distribution import BaseNetDistribution


class VarBayesModuleNetDistribution(BaseNetDistribution):
    """
    Class for distribution of variance nets. This class sees nets as elements of distribution.
    It helps sample nets from this distribution or estimate statistics of distribution.
    For this purpose it have base module architecture and distribution of parameters for
    each of it weights.
    """
    def __init__(self, base_module: nn.Module, weight_distribution: dict[str, ParamDist]) -> None:
        """_summary_

        Args:
            base_module (nn.Module): custom module layer which is going to be converted to BayesModule
            weight_distribution (dict[str, ParamDist]): posteror distribution for each parameter of moudule
        """
        super().__init__(base_module=base_module, weight_distribution=weight_distribution)
