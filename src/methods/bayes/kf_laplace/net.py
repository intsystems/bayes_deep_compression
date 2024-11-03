import torch
from src.methods.bayes.base.net import BayesLinearLayer

class KfLinear(BayesLinearLayer):
    def __init__(self, in_features, out_features, device=None):
        super().__init__(in_features, out_features, device)

    def forward(self, input):
        return super().forward(input)
    
class 