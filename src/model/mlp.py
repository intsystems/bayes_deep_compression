"""
there is some simple multilayer perceptron defined
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class MLP(nn.Module):
    def __init__(self, dimentions: List[int]):
        super(MLP, self).__init__()
        submodules = []
        for i in range(len(dimentions) - 2):
            submodules.append(nn.Linear(dimentions[i],dimentions[i+1]))
            submodules.append(nn.ReLU(inplace = True))
        submodules.append(nn.Linear(dimentions[-2],dimentions[-1]))

        self.net = nn.Sequential( *submodules)
    def forward(self, x):
        return self.net(x)