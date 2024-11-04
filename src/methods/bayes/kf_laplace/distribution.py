import torch
import torch.nn as nn

from methods.bayes.kf_laplace.optimization import HessianAccumulator
from src.methods.bayes.base.distribution import (BaseNetDistribution,
                                                 BaseNetEnsemble)
from src.methods.bayes.kf_laplace.net import KfLinear, KfMLP


class KFLaplaceMLPDistribution(BaseNetDistribution):
    def __init__(self, accumulator: HessianAccumulator):
        self

    def sample_layer(self):
        # H = Qi (kron) HHi
        # sample isotropic unit variance mtrix normal
        Z = MAP.data.new(MAP.size()).normal_(mean=0, std=1)
        # AAT = HHi
        #     A = torch.cholesky(HHinv, upper=False)
        # BTB = Qi
        #     B = torch.cholesky(Qinv, upper=True)
        all_mtx_sample = MAP + torch.matmul(torch.matmul(lower_HHinv, Z), upper_Qinv)

        weight_mtx_sample = all_mtx_sample[:, :-1]
        bias_mtx_sample = all_mtx_sample[:, -1]

        return weight_mtx_sample, bias_mtx_sample

    def sample_net(self):
        return nn.ModuleList([self.sample_layer() for layer in self.accumulator])


class KfLaplaceEnsemble(BaseNetEnsemble[KFLaplaceMLPDistribution]): ...
