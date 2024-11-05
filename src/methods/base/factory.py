from typing import Generic, TypeVar
import torch.nn as nn


BayesModelT = TypeVar('BayesModelT')
BayesLossT = TypeVar('BayesLossT')
PriorDistributionT= TypeVar('PriorDistributionT')


class BaseBayesLossFactory(Generic[BayesLossT]):
    ...

class BaseBayesNetFactory(Generic[BayesModelT]):
    bayes_model_cls: type[BayesModelT]
    prior: type[PriorDistributionT]

    def fabric(self, net: nn.Module) -> BayesModelT:
        self._p_names = []
        for p_name, par in net.named_parameters(recurse=False):
            # make gaussian params and initalize them with random samples
            # we keep normal's std as log(sigma^2)
            self.register_parameter(
                self._to_mean_param_name(p_name),
                nn.Parameter(par.detach())
            )
            self.register_parameter(
                self._to_std_param_name(p_name),
                nn.Parameter(-20 + torch.rand_like(par, requires_grad=True, device=par.device))
            )

            self._p_names.append(p_name)

        for p_name in self._p_names:
            delattr(self._net, p_name)

        # recursively make bayessian submodules and link them to current bayes_module
        for m_name, module in self._net.named_children():
            self.register_module(
                self._to_bayes_name(m_name),
                MakeModuleBayessian(module)
            )