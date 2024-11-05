from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeModuleBayessian(nn.Module):
    def __new__(cls, net: nn.Module):
        """ make individual class name for returned bayessian module
        """
        class RenamedBayessianModule(cls): pass

        # set new name for returned object's class
        RenamedBayessianModule.__name__ = "Bayessian" + net.__class__.__name__
        new_qual_name = net.__class__.__qualname__
        # we remove "MakeModuleBayessian" and "RenamedBayessianModule" from __qualname__
        new_qual_name = ".".join(new_qual_name.split(".")[:-2] + [RenamedBayessianModule.__name__])

        return super().__new__(RenamedBayessianModule)

    def __init__(self, net: nn.Module) -> None: 
        super().__init__()

        # we do not want the net to be registered as submodule
        self.__dict__["_net"] = net

        # create bayessian params for net's own params
        # tensor's device is inherited
        self._p_names = []
        for p_name, par in self._net.named_parameters(recurse=False):
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

        # Special case for nn.Linear.
        # It's bayessian version is optimized using Local Reparameterization Trick.
        # It applies linear transform to weight_mean and weight_std, obtaining ouput's distribution
        # paramters. Then samples from this distribution therefore decreasing number of 
        # required samples.
        # I adhere to the principle of not overwriting net.forward().
        if isinstance(self._net, nn.Linear):
            def linear_pre_hook(lin_module: nn.Linear, args):
                lin_input: torch.Tensor = args[0]

                # detach from previous sampling graph and sample parapms
                lin_module.weight = self.get_parameter(self._to_mean_param_name("weight")).clone()
                lin_module.bias = self.get_parameter(self._to_mean_param_name("bias")).clone()
                
                # save linear transform with std params
                lin_module._temp_std_transform = F.linear(
                    lin_input, 
                    torch.exp(0.5 * self.get_parameter(self._to_std_param_name("weight")))
                )

            self._net.register_forward_pre_hook(linear_pre_hook)

            def linear_post_hook(lin_module: nn.Linear, args, lin_output: torch.Tensor):
                # add std part for weight paramter and std part for bias parameter
                lin_output = lin_output + \
                    torch.randn(lin_output.shape[-1]) * lin_module._temp_std_transform + \
                    torch.randn(lin_output.shape[-1]) * torch.exp(0.5 * self.get_parameter(self._to_std_param_name("bias")))

                return lin_output
      
            self._net.register_forward_hook(linear_post_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # samples all submodule's params and calls net's forward()
        self._sample_params()
        
        return self._net(x)
    
    def sample_estimation(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """ Samples several model's output based on posterior distribution.
            The method is similar to __call__ but prepends "samples" dimension to the output

        Args:
            n_samples (int): number of posterior samples
        """

        return [self(x) for _ in range(n_samples)]
    
    def get_map_module(self, prune: bool = False) -> nn.Module:
        """ Gives MAP estimation of net's params based on their posterior.
            Returned object is an ordinary nn.Module
        """
        with torch.no_grad():
            out_net = deepcopy(self._net)

            # put map estimates into net's params
            self._populate_with_map_estimate(out_net, prune)

        return out_net

    def _populate_with_map_estimate(self, net: nn.Module, prune: bool):
        """ populates  net's params with map estimations (means of gaussian prior)
        """
        # put in module's direct params
        for p_name in self._p_names:
            posterior_mean = self.get_parameter(self._to_mean_param_name(p_name)).clone()

            if prune:
                # bound number 0.83 is suggested by Graves
                posterior_std = torch.exp(self.get_parameter(self._to_std_param_name(p_name)))
                mask = torch.abs(
                    posterior_mean / posterior_std
                ) >= 0.83

                print(p_name, (mask == 0).sum())

                setattr(net, p_name, nn.Parameter(posterior_mean * mask))
            else:
                setattr(net, p_name, nn.Parameter(posterior_mean))

        for m_name, submodule in net.named_children():
            bayes_submodule = self.get_submodule(self._to_bayes_name(m_name))
            bayes_submodule._populate_with_map_estimate(submodule, prune)

    def _sample_params(self):
        """ Samples from param's distributions and put result into param's tensor.
            Linear parameters are not sampled here, but when forward() is called
        """
        # nn.Linear is sampled during forward()
        if isinstance(self._net, nn.Linear):
            return
        
        # detach from previous sampling graph
        # and sample new params
        for p_name in self._p_names:
            sampled_par = self.get_parameter(self._to_mean_param_name(p_name)).clone()
            sampled_par = sampled_par + torch.randn_like(sampled_par) * \
                torch.exp(0.5 * self.get_parameter(self._to_std_param_name(p_name)))
            setattr(self._net, p_name, sampled_par)

        # sample submodule parameters
        for bayes_module in self.children():
            bayes_module._sample_params()

    def _to_bayes_name(self, name: str) -> str:
        """adds "bayes" prefix to name
        """
        return "bayes_" + name
  
    def _to_mean_param_name(self, name: str) -> str:
        return name + "_mean"

    def _to_std_param_name(self, name: str) -> str:
        return name + "_log_std"




