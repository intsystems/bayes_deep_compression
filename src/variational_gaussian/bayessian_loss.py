from typing import Callable

import torch

from .bayes_nn import MakeModuleBayessian


class MakeLossBayessian():
    def __init__(self, loss: Callable, bayessian_nn: MakeModuleBayessian):
        """complement NN's loss function to ELBO loss
        """
        self.prime_loss = loss
        self._bayessian_nn = bayessian_nn

    def __call__(self, target, predict):
        # optimal \alpha parameters on the E-step
        # sigma_opt here is the squared sigma
        mu_opt = 0
        sigma_opt = 0
        # number of posterior random variables = num of parameters
        n_params = 0

        # compute mu
        for p_name, param in self._bayessian_nn.named_parameters():
            if p_name.find("mean") != -1:
                mu_opt += param.sum()
                n_params += param.numel()
        mu_opt /= n_params

        # compute sigma
        for p_name, param in self._bayessian_nn.named_parameters():
            if p_name.find("mean") != -1:
                sigma_opt += torch.sum((param - mu_opt) ** 2)
            else:
                sigma_opt += torch.exp(param.sum())
        sigma_opt /= n_params

        # compute kl-loss between posterior and prior
        kl_loss = 0
        for p_name, param in self._bayessian_nn.named_parameters():
            if p_name.find("mean") != -1:
                kl_loss += torch.sum(0.5 * (1 / sigma_opt) * (param - mu_opt) ** 2)
            else:
                kl_loss += -0.5 * param.sum() + param.numel() * 0.5 * torch.log(sigma_opt)
                kl_loss += torch.sum(-0.5 + 0.5 * torch.exp(param) / sigma_opt)

        return kl_loss + self.prime_loss(target, predict)
