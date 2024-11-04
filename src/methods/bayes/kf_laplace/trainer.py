from src.methods.bayes.base.trainer import BaseBayesTrainer
from src.methods.bayes.kf_laplace.net import KFMlp
from methods.bayes.kf_laplace.loss import HessianRecursion
from src.methods.bayes.base.trainer import TrainerParams
from methods.bayes.kf_laplace.loss import HessianAccumulator
from src.resource.dataset import DatasetLoader


from dataclasses import dataclass


@dataclass
class KfTrainerParams(TrainerParams): ...


class KFEagerTraining(BaseBayesTrainer):
    def __init__(
        self, model: KFMlp, dataset_loader: DatasetLoader, params, report_chain
    ):
        super().__init__(params, report_chain, dataset_loader=dataset_loader)
        self.model = model

    def train(
        self,
    ):
        map_net = self.train_map()
        return self.train_posterior(map_net)

    def train_map(self):
        for train_sample in self.dataset_loader:
            self.model(train_sample)
            self.model.zero_grad()

    def train_posterior(self, map_net: KFMlp):
        for train_sample in self.dataset_loader:
            self.map_net
            train_sample
        accumulator = HessianAccumulator()
        HessianRecursion()
        return accumulator
