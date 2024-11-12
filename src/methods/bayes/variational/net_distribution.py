import torch.nn as nn
from src.methods.bayes.base.net_distribution import BaseNetDistribution
from src.methods.bayes.base.distribution import ParamDist


class VarBayesModuleNetDistribution(BaseNetDistribution):
    def __init__(
        self, base_module: nn.Module, weight_distribution: dict[str, ParamDist]
    ) -> None:
        super().__init__(
            base_module=base_module, weight_distribution=weight_distribution
        )
