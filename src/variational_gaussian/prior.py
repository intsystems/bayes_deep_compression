import torch
import torch.nn as nn
from typing import Any


class WithGaussianPrior(type):
    def __init__(cls, name, bases, namespace):
        super(WithGaussianPrior, cls).__init__(name, bases, namespace)

    def _base_forward_pre_hook(module, *args):
        # sample weights here
        pass

    def _linear_forward_pre_hook(module, *args, output):
        # do local reparemtrization trick here
        pass

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        """ rewrite submodules of the user's module. Samples weights from posterior to every submodule
                before forward(). For Linear submodules uses local parametrization trick.
        """
        base_nn: nn.Module = super(WithGaussianPrior, cls).__call__(*args, **kwds)

        for module in base_nn.modules():
            if type(module) is not nn.Linear:
                module.register_forward_pre_hook(WithGaussianPrior._base_forward_pre_hook)
            else:
                module.register_forward_pre_hook(WithGaussianPrior._linear_forward_pre_hook)

            # add means and stds for module's weights for sampling and ELBO-training
            module.mean = nn.Parameter(...)
            module.std = nn.Parameter(...)

        # rewrite forward so it accepts new parameter - number of samples from posterior
        base_nn.forward = ...

        # this method prune weights of the model and return MAP-estimation of the NN.
        # So user can use returned NN as usual nn.Module
        base_nn.get_pruned = ...

        return base_nn


