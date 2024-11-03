from src.methods.bayes.base.trainer import BaseBayesTrainer
from src.methods.bayes.base.net import MLPBayesModel
from src.methods.bayes.kf_laplace.hessian import HessianRecursion
from dataclasses import dataclass


class KFEagerTraining(BaseBayesTrainer):
    def __init__(self, model: MLPBayesModel, visualizer:):
        self.model = model

    def train(self,)


class KFPosteriorTrainer(BaseBayesTrainer):
    def __init__(self):
        ...
        
    def train(self,model:, dataset:):
        ...
