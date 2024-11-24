from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar

from src.methods.bayes.base.net_distribution import BaseNetDistribution
from src.methods.report.base import ReportChain
from torch.optim.optimizer import Optimizer


@dataclass
class TrainerParams:
    num_epochs: int
    optimizer: Optimizer


ModelT = TypeVar("ModelT")


class BaseBayesTrainer(Generic[ModelT], ABC):
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

    @abstractmethod
    def train(self, *args, **kwargs) -> BaseNetDistribution:
        ...
