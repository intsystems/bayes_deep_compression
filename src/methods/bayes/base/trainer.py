from dataclasses import dataclass
from typing import Generic, TypeVar

from torch.optim.optimizer import Optimizer

from methods.bayes.base.optimization import BaseLoss
from src.methods.bayes.base.distribution import NetDistribution
from src.methods.report.base import ReportChain
from src.resource.dataset import DatasetLoader

ModelT = TypeVar("ModelT")


@dataclass
class TrainerParams:
    num_epochs: int
    optimizer: Optimizer


class BaseBayesTrainer(Generic[ModelT]):
    def __init__(
        self,
        params: TrainerParams,
        report_chain: ReportChain,
        dataset_loader: DatasetLoader,
    ):
        self.report_chain = report_chain
        self.dataset_loader = dataset_loader
        self.params = params

    def train(self, *args, **kwargs) -> NetDistribution: ...
