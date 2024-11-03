from src.methods.bayes.base.distribution import NetDistribution
from typing import Generic, TypeVar
from src.methods.report.base import ReportChain
from dataclasses import dataclass

ModelT = TypeVar('ModelT')

@dataclass
class TrainerParams:
    num_epochs: str

class BaseBayesTrainer(Generic[ModelT]):
    def __init__(self, params:TrainerParams, report_chain: ReportChain, ):
        self.report_chain = report_chain


    def train(self, *args, **kwargs) -> NetDistribution:
        ...   