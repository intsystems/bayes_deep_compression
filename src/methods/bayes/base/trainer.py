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
    """Abstcract trainer class that is used to provide simple interface to train 
    bayesian modules. Could be used to create your trainers."""
    def __init__(
        self,
        params: TrainerParams,
        report_chain: Optional[ReportChain],
        train_dataset: Iterable,
        eval_dataset: Iterable,
    ):
        """_summary_

        Args:
            params (TrainerParams): trianing params that is used to fine-tune training
            report_chain (Optional[ReportChain]): All callback that should be return by each epoch
            train_dataset (Iterable): Dataset on which model should be trained
            eval_dataset (Iterable): Dataset on which epoch of training model should be evaluated
        """
        self.params = params
        """Storing any trianing params that is used to fine-tune training"""
        self.report_chain = report_chain
        """All callback that should be return by each epoch"""
        self.train_dataset = train_dataset
        """Dataset on which model should be trained"""
        self.eval_dataset = eval_dataset
        """Dataset on which epoch of training model should be evaluated"""

    @abstractmethod
    def train(self, *args, **kwargs) -> BaseNetDistribution:
        """It simply train provided model with tarin parameters that is stores in params

        Returns:
            BaseNetDistribution: Distribution of nets that could be used
            to sample models or getting map estimation (the most probable) by result of train.
        """
        ...
