import torch
import torch.nn as nn


class AddBayessianLoss():
    def __init__(self, loss, bayessian_nn):
        """change loss function for neural network for ELBO loss
        """
        self.init_loss = loss
        self._bayessian_nn = bayessian_nn

    def __call__(self, target, predict):
        # compute kl-loss based on self._bayessian_nn
        kl_loss = ...

        return kl_loss + self.init_loss(target, predict)
