""" Common fixtures for bayes modules tests
"""
import pytest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.bayes.variational.optimization import *
from src.methods.bayes.variational.net import *


@pytest.fixture
def num_test_samples(request) -> int:
    """ Number of trials on testing bayes model's forward()
    """
    return request.config.getoption("--num_test_samples")

@pytest.fixture
def model_dim(request) -> int:
    """ Dimensionality of 1d models in tests
    """
    return request.config.getoption("--model_dim")

@pytest.fixture(params=[LogUniformVarLayer, NormalVarBayesLayer])
def bayes_module_cls(request) -> BayesLayer:
    return request.param


class SimpleMLP(nn.Sequential):
    def __init__(self, dim: int):
        super().__init__(
            nn.LayerNorm(dim),
            *([nn.Linear(dim, dim), nn.ReLU()] * 2),
            nn.Linear(dim, dim)
        )

@pytest.fixture(params=[nn.Linear, SimpleMLP])
def module(request, model_dim) -> nn.Module:
    """ Provide simple 1d Modules for the tests
    """
    if issubclass(request.param, nn.Linear):
        return request.param(model_dim, model_dim)
    elif issubclass(request.param, SimpleMLP):
        return request.param(model_dim)
    else:
        raise ValueError("Module is not adapted for test yet.")
