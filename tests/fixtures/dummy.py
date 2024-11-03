import torch
import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.fc1(x))