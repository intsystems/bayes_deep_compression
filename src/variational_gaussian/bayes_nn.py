from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeModuleBayessian(nn.Module):
    class HiddenModule:
        """ wrapper around nn.Module. Class's instances won't be registered as submodules 
                in parent nn.Module.
        """
        def __init__(self, net: nn.Module) -> None:
            self._net = net

        def __getattribute__(self, name: str) -> Any:
            return self._net.__getattribute__(name)
        
        def __setattr__(self, name: str, value: Any) -> None:
            self._net.__setattr__(name, value)

    def __new__(cls, net: nn.Module):
        """ make individual class name for return bayessian module
        """
        class RenamedBayessianModule(cls): pass

        # set new name for returned object's class
        RenamedBayessianModule.__name__ = "Bayessian" + net.__class__.__name__
        new_qual_name = net.__class__.__qualname__
        # we remove "MakeModuleBayessian" and "RenamedBayessianModule" from __qualname__
        new_qual_name = ".".join(new_qual_name.split(".")[:-2].append(RenamedBayessianModule.__name__))

        return super().__new__(RenamedBayessianModule)

    def __init__(self, net: nn.Module) -> None:
        super().__init__()

        # this is wrapper around the nn.Module
        self._net: nn.Module = self.HiddenModule(net)

        # create bayessian params for net's own params
        # tensor's device is inherited
        for p_name, par in self._net.named_parameters(recurse=False):
            # net's params are not leafs anymore
            par.is_leaf = False

            # make gaussian params and initalize them with N(0, 1)
            # we keep normal's std as log(sigma^2)
            self.register_parameter(
                self._to_mean_param_name(p_name), 
                nn.Parameter(torch.rand_like(par, requires_grad=True, device=par.device))
            )
            self.register_parameter(
                self._to_std_param_name(p_name), 
                nn.Parameter(torch.rand_like(par, requires_grad=True, device=par.device))
            )

        # recursively make bayessian submodules and link them to current bayes_module
        for m_name, module in self._net.named_modules():
            # skip current module
            if m_name == "":
                continue

            self.register_module(
                self._to_bayes_name(m_name),
                MakeModuleBayessian(module)
            )

        # Special case for nn.Linear.
        # It's bayessian version is optimized using Local Reparameterization Trick.
        # It applies linear transform to weight_mean and weight_std, obtaining ouput's distribution
        # paramters. Then samples from this distribution therefore decreasing number of 
        # required samples .
        # I adhere to the principle of not overwriting net.forward().
        if isinstance(self._net._net, nn.Linear):
            def linear_pre_hook(lin_module: nn.Linear, lin_input: torch.Tensor):
                # put weight means in Linear.weight to perform transform
                lin_module.weight.copy_(
                    self.get_parameter(self._to_mean_param_name("weight"))
                )
                # put bias means in Linear.bias to perform transform
                lin_module.bias.copy(
                    self.get_parameter(self._to_mean_param_name("bias"))
                )
                # save linear transform with std params
                lin_module._temp_std_transform = F.linear(
                    lin_input, 
                    torch.exp(self.get_parameter(self._to_std_param_name("weight")))
                )

            net.register_forward_pre_hook(linear_pre_hook)

            def linear_post_hook(lin_module: nn.Linear, lin_output: torch.Tensor):
                # add std part for weight paramter and std part for bias parameter
                lin_output = lin_output + \
                    torch.randn(lin_output.shape[-1]) * lin_module._temp_std_transform + \
                    torch.randn(lin_output.shape[-1]) * torch.exp(self.get_parameter(self._to_std_param_name("bias")))
                del lin_module._temp_std_transform

                return lin_output
      
            net.register_forward_hook(linear_post_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # samples all submodule's params and calls net's forward()
        self._sample_params()
        
        return self._net.forward(x)
    
    def sample_estimation(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """ Samples several model's output based on posterior distribution.
            The method is similar to __call__ but prepends "samples" dimension to the output

        Args:
            n_samples (int): number of posterior samples
        """
        posterior_samples = [super().__call__(x).unsqueeze(dim=0) for _ in range(n_samples)]
        posterior_samples = torch.concat(posterior_samples, dim=0)

        return posterior_samples
    
    def get_map_module(self, prune: bool = False) -> nn.Module:
        """ Gives MAP estimation of net's params based on their posterior.
            Returned object is an ordinary nn.Module
        """
        with torch.no_grad():
            # put map estimates into net's params
            self._populate_with_map_estimate(prune)

            # make net's copy and load map state dict here
            map_module_copy: nn.Module = deepcopy(self._net._net)
            map_module_copy.load_state_dict(self._net._net.state_dict())

        return map_module_copy

    def _populate_with_map_estimate(self, prune: bool):
        """ populates  net's params with map estimations (means for gaussian prior)
        """
        # put in module's direct params
        for p_name, par in self._net.named_parameters(recurse=False):
            posterior_mean = self.get_parameter(self._to_mean_param_name(p_name))

            if prune:
                # bound number 0.83 is suggested from Graves
                posterior_std = torch.exp(self.get_parameter(self._to_std_param_name(p_name)))
                mask = torch.abs(
                    posterior_mean / posterior_std
                ) < 0.83
                par.copy_(posterior_mean * mask)
            else:
                par.copy_(posterior_mean)

            par.is_leaf = True

        # sample submodule parameters
        for b_m_name, bayes_module in self.named_modules():
            # skip current module
            if b_m_name == "":
                continue

            bayes_module._populate_with_map_estimate(prune)

    def _sample_params(self):
        """ Samples from param's distributions and put result into param's tensor.
            Linear parameters are not sampled here, but when forward() is called
        """
        # nn.Linear is sampled during forward()
        if isinstance(self._net._net, nn.Linear):
            return

        # sample module's direct params
        for p_name, par in self._net.named_parameters(recurse=False):
            sampled_par = self.get_parameter(self._to_mean_param_name(p_name))
            sampled_par += torch.randn_like(sampled_par) * \
                torch.exp(self.get_parameter(self._to_std_param_name(p_name)))
            par.copy_(sampled_par)

        # sample submodule parameters
        for b_m_name, bayes_module in self.named_modules():
            # skip current module
            if b_m_name == "":
                continue

            bayes_module._sample_params()

    def _to_bayes_name(self, name: str) -> str:
        """adds "bayes" prefix to name
        """
        return "bayes" + name
  
    def _to_mean_param_name(self, name: str) -> str:
        return name + "_mean"

    def _to_std_param_name(self, name: str) -> str:
        return name + "_log_std"




