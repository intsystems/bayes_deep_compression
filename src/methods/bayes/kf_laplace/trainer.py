from src.methods.bayes.base.trainer import BaseBayesTrainer
from src.methods.bayes.kf_laplace.net import KFMlp
from src.methods.bayes.kf_laplace.hessian import HessianRecursion
from src.methods.bayes.base.trainer import TrainerParams
from src.methods.bayes.kf_laplace.hessian import HessianAccumulator
from dataclasses import dataclass

@dataclass
class KfTrainerParams(TrainerParams):
    ...


class KFEagerTraining(BaseBayesTrainer):
    def __init__(self, model: KFMlp, params, report_chain):
        super().__init__(params, report_chain)
        self.model = model

    def train(self,):
        map_net = self.train_map()
        return self.train_posterior()


    def train_map(self):
        KFMlp()

    def train_posterior(self, map_net: KFMlp):
        accumulator = HessianAccumulator()
        HessianRecursion()
        return accumulator
        
        



class KFPosteriorTrainer(BaseBayesTrainer):
    def __init__(self):
        ...
        
    def train(self,model:, dataset:):
        ...
