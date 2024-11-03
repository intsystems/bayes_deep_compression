import torch.nn as nn
from tests.fixture import dummy_net, DummyNet

from src.cli.handlers.compress import CompressHandler

def test_linear_compression(dummy_net: DummyNet):
    assert CompressHandler().compress_linear_model(dummy_net)


def test_dummy(dummy_net: DummyNet):
    assert dummy_net.fc1 == nn.Linear(1, 2)
