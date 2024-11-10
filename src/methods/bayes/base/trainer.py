from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, Iterable
import torch
from torch.optim.optimizer import Optimizer

from src.methods.bayes.base.optimization import BaseLoss
from src.methods.bayes.base.distribution import BaseNetDistribution
from src.methods.report.base import ReportChain
from src.methods.bayes.base.net import ModelT

@dataclass
class TrainerParams:
    num_epochs: int
    optimizer: Optimizer


class BaseBayesTrainer(Generic[ModelT]):
    def __init__(
        self,
        params: TrainerParams,
        report_chain: Optional[ReportChain],
        dataset: Iterable,
    ):
        self.report_chain = report_chain
        self.dataset = dataset
        self.params = params

    def train(self, *args, **kwargs) -> BaseNetDistribution[ModelT]: ...
