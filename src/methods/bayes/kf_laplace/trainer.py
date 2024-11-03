from src.methods.bayes.base.trainer import BaseBayesTrainer
from src.methods.bayes.kf_laplace.hessian import HessianRecursion

class KFTrainer(BaseBayesTrainer):
    def __init__(self):
        
