from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, Iterable
from torch.optim.optimizer import Optimizer

from bayescomp.bayes.base.net_distribution import BaseNetDistribution
from bayescomp.report.base import ReportChain


@dataclass
class TrainerParams:
    num_epochs: int
    optimizer: Optimizer


ModelT = TypeVar("ModelT")


class BaseBayesTrainer(Generic[ModelT]):
    def __init__(
        self,
        params: TrainerParams,
        report_chain: Optional[ReportChain],
        train_dataset: Iterable,
        eval_dataset: Iterable,
    ):
        self.params = params
        self.report_chain = report_chain
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self, *args, **kwargs) -> BaseNetDistribution: ...
