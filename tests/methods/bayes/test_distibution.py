""" Testing distribution classes. They were used in our project papers. 
"""
import pytest

import torch
import torch.nn as nn

from src.methods.bayes.variational.distribution import LogUniformVarDist, NormalReparametrizedDist


@pytest.fixture(params=[
    (LogUniformVarDist, 4),
    (NormalReparametrizedDist, 2)
])
def dist_setup(request):
    """ fixture yeilds distibution class and number of dsitribution parameters
    """
    return request.param

def test_param_dist(dist_setup):
    SHAPE = (2, 2)
    NUM_SAMPLES = 10
    SAMPLE_SHAPE = (1, 5)

    dist_class  = dist_setup[0]
    num_dist_params = dist_setup[1]

    # create distribution out of given parameters
    dist = dist_class(
        *[torch.rand(SHAPE) for _ in range(num_dist_params)]
    )
    # create distibtion out of given tensor
    dist = LogUniformVarDist.from_parameter(torch.rand(SHAPE))

    # access parameters
    for param in dist.get_params().values():
        assert isinstance(param, nn.Parameter)

    # test distribution properties
    dist_prop_names = ["mean", "variance", "map"]
    for prop_name in dist_prop_names:
        try:
            prop = getattr(dist, prop_name)
            assert isinstance(prop, torch.Tensor)
        except NotImplementedError:
            print(f"{prop_name} is not implemented in {dist.__class__.__name__}!")

    # test getting log(p)
    for _ in range(NUM_SAMPLES):
        assert dist.log_prob(20 * torch.rand(SHAPE) - 10) <= 0.

    # z-test
    dist.log_z_test()

    # test r-sampling
    for _ in range(NUM_SAMPLES):
        sample = dist.rsample(SAMPLE_SHAPE)
        assert sample.requires_grad == True
        assert sample.shape == [*SAMPLE_SHAPE].extend([*SHAPE])




