""" Testing different loss classes based on ELBO
"""
import pytest
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.bayes.variational.optimization import *
from src.methods.bayes.variational.net import *


@pytest.fixture(params=[LogUniformVarKLLoss, NormVarKLLoss, VarRenuiLoss])
def loss_cls(request) -> type[VarDistLoss]:
    return request.param


def test_simple_loss(loss_cls: type[VarDistLoss], 
                     bayes_module_cls: type[BayesLayer], 
                     module: nn.Module,
                     model_dim: int,
                     num_test_samples: int
):
    # create bayes module
    bayes_module: BayesLayer = bayes_module_cls(module)
    # create loss
    loss_func = loss_cls()

    for _ in range(num_test_samples):
        # compute variational loss
        params_sample = bayes_module.sample()
        try:
            var_loss: torch.Tensor = loss_func(
                param_sample_dict=params_sample, 
                posterior=bayes_module.posterior,
                prior=bayes_module.prior
            )
        # if bayes module and loss are incompatible we will try to catch it here
        except AttributeError:
            continue

        assert isinstance(var_loss, torch.Tensor)

        # compute fit loss
        fit_loss = F.mse_loss(
            bayes_module(torch.rand((model_dim, ))),
            torch.zeros((model_dim, ))
        )

        # aggregate losses
        agg_result = loss_func.aggregate(
            [fit_loss],
            [var_loss],
            beta=1
        )
        assert isinstance(agg_result.fit_loss, torch.Tensor)
        assert isinstance(agg_result.dist_loss, torch.Tensor)
        assert isinstance(agg_result.total_loss, torch.Tensor)
        assert torch.allclose(agg_result.total_loss, agg_result.dist_loss + agg_result.fit_loss)

        # populate gradients
        agg_result.total_loss.backward()
        # check gradients of distribution parameters
        for distr in bayes_module.net_distribution.weight_distribution.values():
            for distr_param in distr.get_params().values():
                assert distr_param.requires_grad is True
                assert not (distr_param.grad is None)
                assert not distr_param.grad.allclose(torch.zeros_like(distr_param.grad))

        bayes_module.zero_grad()

