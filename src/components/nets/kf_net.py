import torch.nn as nn
from src.methods.bayes.kf_laplace.net import KfMLP


class KfMLPComponent(KfMLP):
    def __init__(self, input_dim, output_dim):
        super().__init__(
            activation=nn.ReLU(inplace=True),
            bayesian_layer_list=list[
                nn.Linear(in_features=input_dim, out_features=2 * input_dim),
                nn.Linear(in_features=2 * input_dim, out_features=output_dim),
            ],
        )
