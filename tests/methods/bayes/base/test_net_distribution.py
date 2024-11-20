""" Testing net distribution class. It is a number of ParamDist for some module
"""
import pytest

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
