class VariationalGaussBayesLayer:
    def __init__(self):
        ...

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

class 


    d

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
