from dataclasses import dataclass

import torch

from src.methods.bayes.base.trainer import BaseBayesTrainer, TrainerParams
from src.methods.bayes.kf_laplace.distribution import KFLaplaceMLPDistribution
from src.methods.bayes.kf_laplace.net import KfMLP
from src.methods.bayes.kf_laplace.optimization import (HessianAccumulator,
                                                       RecurseHessian)
from src.resource.dataset import DatasetLoader


@dataclass
class KfTrainerParams(TrainerParams): ...


class KFEagerTraining(BaseBayesTrainer):
    def __init__(
        self,
        model: KfMLP,
        dataset_loader: DatasetLoader,
        params: KfTrainerParams,
        report_chain,
    ):
        super().__init__(params, report_chain, dataset_loader=dataset_loader)
        self.model = model
        self.loss = self.dataset_loader.loss

    def train(
        self,
    ):
        map_net = self.train_map()
        return self.train_posterior(map_net)

    def train_map(self):
        for train_sample in self.dataset_loader:
            self.loss.forward(self.model.forward(train_sample.x_train))

            self.params.optimizer.step()
            self.model.zero_grad()
        return self.model

    @torch.no_grad
    def train_posterior(self, map_net: KfMLP):
        return KFLaplaceMLPDistribution(
            accumulator=HessianAccumulator(
                hessian_generators=(
                    RecurseHessian(train_sample) for train_sample in self.dataset_loader
                )
            )
        )
