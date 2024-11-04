from dataclasses import dataclass

import torch.nn as nn
from torch.optim.sgd import SGD

from src.methods.bayes.kf_laplace.trainer import KFEagerTraining, KfTrainerParams


@dataclass
class KFParamsComponent(KfTrainerParams):
    num_epochs = 10
    optimizer = SGD()
    loss = nn.MSELoss()


class KFEagerTrainingComponent(KFEagerTraining):
    def __init__(self, model, visualizer):
        super().__init__(model, visualizer)
