import torch.nn as nn 
from tests.fixture import dummy_net, DummyNet

def test_dummy(dummy_net:  DummyNet):
    assert dummy_net.fc1 == nn.Linear(1,2)

