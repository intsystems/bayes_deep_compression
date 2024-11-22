""" Testing net distribution class. It is a number of ParamDist for some module
"""
import pytest
from warnings import warn

import numpy as np
import torch
import torch.nn as nn

from src.methods.bayes.variational.distribution import LogUniformVarDist, NormalReparametrizedDist, ParamDist
from src.methods.bayes.base.net_distribution import BaseNetDistribution, BaseNetDistributionPruner, BaseNetEnsemble


@pytest.mark.parametrize("module", [nn.Linear(20, 20), nn.Conv2d(10, 10, 3)])
@pytest.mark.parametrize("weight_dist", [LogUniformVarDist, NormalReparametrizedDist])
@pytest.mark.parametrize("bias_dist", [NormalReparametrizedDist, NormalReparametrizedDist])
def test_simple_net_distribution(module: nn.Module, weight_dist: ParamDist, bias_dist: ParamDist):
    NUM_SAMPLES = 10

    # create net distribution
    weight_distribution = {
        "weight": weight_dist.from_parameter(module.weight),
        "bias": bias_dist.from_parameter(module.bias)
    }
    net_distr = BaseNetDistribution(module, weight_distribution)

    # create copy of the base_module inside net_distr
    net_distr.detach_()
    assert net_distr.base_module is not module

    # sample parameters of the base module
    for _ in range(NUM_SAMPLES):
        net_distr.sample_model()
        # check that new values were generated
        assert not torch.allclose(module.weight, net_distr.base_module.weight)
        assert not torch.allclose(module.bias, net_distr.base_module.bias)

    # set mean/map values into module
    net_distr.set_mean_params()
    for param_name in weight_distribution:
        assert isinstance(getattr(net_distr.base_module, param_name), torch.Tensor)
    net_distr.set_map_params()
    for param_name in weight_distribution:
        assert isinstance(getattr(net_distr.base_module, param_name), torch.Tensor)

    # build nn.Module inside net_distr
    base_module = net_distr.get_model()
    for param_name in weight_distribution:
        assert isinstance(getattr(net_distr.base_module, param_name), nn.Parameter)

    # build copy of nn.Module inside net_distr
    base_module_copy = net_distr.get_model_snapshot()
    assert base_module is not base_module_copy


def test_incorrect_net_distribution():
    # module without parameters
    module = nn.ReLU()
    # create net distribution
    weight_distribution = {
        "weight": NormalReparametrizedDist(torch.rand(5), torch.rand(5)),
    }
    net_distr = BaseNetDistribution(module, weight_distribution)

    # check that we cannot sample params
    with pytest.raises(Exception):
        net_distr.sample_model()



@pytest.mark.parametrize("module", [nn.Linear(20, 20), nn.Conv2d(10, 10, 3)])
@pytest.mark.parametrize("weight_dist", [LogUniformVarDist, NormalReparametrizedDist])
@pytest.mark.parametrize("bias_dist", [NormalReparametrizedDist, NormalReparametrizedDist])
@pytest.mark.parametrize("threshold", np.linspace(0, 1, 10).tolist() + [-1, 5])
def test_simple_pruner(module: nn.Module, weight_dist: ParamDist, bias_dist: ParamDist, threshold: float):
    NUM_SAMPLES = 10

    # compute true value of module's parameters
    base_module_total_params = sum([param.numel() for param in module.parameters()])

    # create net distribution
    weight_distribution = {
        "weight": weight_dist.from_parameter(module.weight),
        "bias": bias_dist.from_parameter(module.bias)
    }
    net_distr = BaseNetDistribution(module, weight_distribution)
    # create pruner object
    pruner = BaseNetDistributionPruner(net_distr)

    # correctness check
    assert pruner.total_params() == base_module_total_params

    # prune weights of the base module inside BaseNetDistribution
    pruner.prune(threshold)
    if threshold < 0 or threshold > 1:
        warn("Threshold may be inncorrect, but no exception is raised", RuntimeWarning)

    # check prune statistics
    assert pruner.total_params() == base_module_total_params
    assert pruner.prune_stats() >= 0
    assert pruner.prune_stats() <= pruner.total_params()
