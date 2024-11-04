from src.methods.bayes.base.distribution import NetDistribution
from typing import Generic, TypeVar
from src.methods.report.base import ReportChain
from dataclasses import dataclass
from src.resource.dataset import DatasetLoader
from src.methods.bayes.base.loss import BaseLoss
from torch.optim.optimizer import Optimizer

ModelT = TypeVar("ModelT")


@dataclass
class TrainerParams:
    num_epochs: str
    optimizer: Optimizer
    loss: BaseLoss


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
