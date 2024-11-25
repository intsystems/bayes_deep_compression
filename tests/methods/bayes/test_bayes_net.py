""" Testing foundamental building block of our library - BayesModule, and
     another high-order envelope BaseBayesModuleNet.
"""
import pytest

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.bayes.base.net import *
from src.methods.bayes.variational.net import *


def test_simple_bayes_module(
        bayes_module_cls: type[BayesLayer], 
        module: nn.Module,
        model_dim: int,
        num_test_samples: int
):

    module = module.to("cpu")
    num_module_params = len(list(module.parameters()))

    # create bayes module
    bayes_module: BayesLayer = bayes_module_cls(module)

    # check we have the same number of distributions as Parameters in initial module
    assert len(bayes_module.posterior) == num_module_params
    # check attributes consistency
    assert bayes_module.base_module is module
    # check device consistency
    assert bayes_module.device == torch.device("cpu")

    for i in range(num_test_samples):
        # alternate module's regime
        if i % 2 == 0:
            bayes_module.train()
        else:
            bayes_module.eval()

        # sample module parameters
        param_samples = bayes_module.sample()
        # compute loss and populate gradients of distribution parameters
        model_output = bayes_module(10 * torch.rand(model_dim))
        loss = F.mse_loss(model_output, torch.zeros_like(model_output))
        loss.backward()

        # check gradients of distribution parameters
        for distr in bayes_module.net_distribution.weight_distribution.values():
            for distr_param in distr.get_params().values():
                assert distr_param.requires_grad is True
                assert not (distr_param.grad is None)
                assert not distr_param.grad.allclose(torch.zeros_like(distr_param.grad))

        bayes_module.zero_grad()

        # check gradients to be zero
        for distr in bayes_module.net_distribution.weight_distribution.values():
            for distr_param in distr.get_params().values():
                assert distr_param.grad.allclose(torch.zeros_like(distr_param.grad))


def test_simple_bayes_net(
        model_dim: int,
        num_test_samples: int
):
    # build mixed module
    linears_1 = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Linear(model_dim, model_dim))
    linear_2 = nn.Linear(model_dim, model_dim)
    batch_norm = nn.BatchNorm1d(model_dim)
    base_module = nn.Sequential(
        linears_1,
        nn.Sigmoid(),
        linear_2,
        nn.Sigmoid()
    )
    # make copy of base_module
    base_module_copy = deepcopy(base_module)

    # make bayes net out of base_module
    bayes_net = VarBayesNet(
        base_module,
        nn.ModuleList([
            LogUniformVarLayer(linears_1),
            batch_norm,
            linear_2
        ])
    )

    # check device
    assert bayes_net.device == torch.device("cpu")

    for i in range(num_test_samples):
        # alternate module's regime
        if i % 2 == 0:
            bayes_net.train()
        else:
            bayes_net.eval()

        # sample module parameters
        param_samples = bayes_net.sample()
        # compute loss and populate gradients of distribution parameters
        model_output = bayes_net(10 * torch.rand(model_dim))
        loss = F.mse_loss(model_output, torch.zeros_like(model_output))
        loss.backward()

        # check gradients of distribution parameters
        for distr in bayes_net.posterior.values():
            for distr_param in distr.get_params().values():
                assert distr_param.requires_grad is True
                assert distr_param.grad is not None
                assert not distr_param.grad.allclose(torch.zeros_like(distr_param.grad))

        # check gradients of non-bayes submodules
        for submodule in bayes_net.module_list:
            if not isinstance(submodule, BayesLayer):
                for param in submodule.parameters():
                    assert param.requires_grad is True
                    assert not (param.grad is None)
                    assert not param.grad.allclose(torch.zeros_like(param.grad))

        bayes_net.zero_grad()

    # get ordinary Module as sample from BayessianNet
    sampled_module = bayes_net.sample_model()
    # check it's submodules and Parameters
    assert list(
        dict(sampled_module.named_modules()).keys()
    ) == list(
        dict(base_module_copy.named_modules()).keys()
    )
    assert len(list(sampled_module.parameters())) == len(list(base_module_copy.parameters()))
    