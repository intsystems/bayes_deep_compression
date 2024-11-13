"""
В этом файле определен класс, делающий
байесовскую модель из поданной ей на вход
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.priors.priors import isotropic_gauss, laplace_prior
from torch.autograd import Variable


def remove_from(name, *dicts_or_sets):
    for d in dicts_or_sets:
        if name in d:
            if isinstance(d, dict):
                del d[name]
            else:
                d.discard(name)


def reset_param(mod, name, value):
    dcts = []
    for a in ["__dict__", "_buffers", "_modules", "_non_persistent_buffers_set"]:
        try:
            eval(f"mod.append(self.{a})")
        except Exception:
            pass
    remove_from(name, dcts)
    mod._parameters[name] = value


class ModelHandler(nn.Module):
    def __init__(self, model, prior=laplace_prior(mu=0, b=0.2), posterior=isotropic_gauss()):
        super(ModelHandler, self).__init__()
        self.mu_params = nn.ParameterList(
            [
                nn.Parameter(t.new(t.size()).uniform_(-0.1, 0.1), requires_grad=True)
                for t in model.parameters()
            ]
        )
        self.sigma_params = nn.ParameterList(
            [
                nn.Parameter(t.new(t.size()).uniform_(-0.1, 0.1), requires_grad=True)
                for t in model.parameters()
            ]
        )
        self.model = model

        self.param_names = self.model.state_dict().keys()
        self.prior = prior
        self.posterior = posterior

    def override_params(self, with_noise=False):
        for name, mu_tens, sigm_tens in zip(self.param_names, self.mu_params, self.sigma_params):
            # семплируем
            lqw, lpw = 0.0, 0.0
            if with_noise:
                eps_tens = Variable(mu_tens.new(mu_tens.size()).normal_())
                std_tens = 1e-6 + F.softplus(sigm_tens, beta=1, threshold=20)
                new_tens = mu_tens + 1 * eps_tens * std_tens
                # отмечаем что нужно обновлять
                new_tens.retain_grad()

                # TODO: после определения общего класса для приоров сделать возможноссть выхывать с другими классами
                lqw += self.posterior.loglike(new_tens, mu_tens, std_tens)
                lpw += self.prior.loglike(new_tens)
            else:
                new_tens = mu_tens
            # находим модуль, в котором нужно заменить Parameter.
            # последний элемент в названии -- название Parameter, предыдущие -- вложенные модули
            parts = name.split(".")
            prev = parts[:-1]
            nxt = parts[-1]
            a = self.model
            for submodel in prev:
                if isinstance(a, nn.Sequential) or hasattr(a, "__getitem__"):
                    a = a[int(submodel)]
                else:
                    a = a.__getattr__(submodel)

            reset_param(a, nxt, new_tens)
        return lqw, lpw

    def eval(self):
        if self.training:
            self.override_params(with_noise=False)
        super().eval()

    def forward(self, x):
        if not self.training:
            #  TODO: сейчас семплируется по последним зашумленным весам,
            # нужно сначала один раз поменять веса на последнюю оценку средних
            return self.model(x), torch.tensor(0), torch.tensor(0)
        lqw, lpw = self.override_params(with_noise=True)
        return self.model(x), lqw, lpw

    def loss(self, x, y):
        labels, lqw, lpw = self(x)
        cross_entropy = F.cross_entropy(labels, y, reduction="sum")
        loss = cross_entropy  # +  (lqw - lpw)/x.shape[0] # нормируем на размер батча

        predicted_labels = torch.mean((cross_entropy.argmax(-1) == y).float())
        return {
            "total": loss,
            "lqw": lqw,
            "lpw": lpw,
            "cross_entropy": cross_entropy,
            "acc": predicted_labels,
        }
