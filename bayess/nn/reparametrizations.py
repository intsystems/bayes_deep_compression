from abc import abstractmethod
from typing import TypeVar, Generic, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class InitargsHandler:
    prior_initargs: Dict[str, Any]
    posterior_initargs: Dict[str, Any]

# TODO: перенести в другое место
class ReparamNormal(D.Normal):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=None):
        self.loc, self._scale = loc, scale
        batch_shape = self.loc.size()
        D.Normal.__base__.__init__(self, batch_shape, validate_args=validate_args)

    @property
    def scale(self):
        return F.softplus(self._scale)
    
    
DistributionType = TypeVar("DistributionType", bound=D.distribution.Distribution)


class BasePriorPosteriorReparam(nn.Module, Generic[DistributionType]):
    PRIOR: DistributionType
    POSTERIOR: DistributionType

    @classmethod
    def factory(cls, *args, **kwargs):
        """
        this function allows to get basic factory for class instances
        """

        def f(shape):
            return cls(shape, *args, **kwargs)
        return f
    
    def __init__(self, shape, *args, **kwargs) -> None:
        
        nn.Module.__init__(self,)

        initargs: InitargsHandler = self.get_initargs(shape)

        self.prior: self.PRIOR = self.PRIOR(**initargs.prior_initargs)
        self.posterior: self.PRIOR = self.POSTERIOR(**initargs.posterior_initargs)

        # чтобы параметры появились в model.parameters
        # наверное это не совсем хорошо, можно просто возвращать список параметров пользователю в модель 
        for name, val in initargs.posterior_initargs.items():
            if isinstance(val, nn.Parameter):
                full_name = f"_{name}"
                self.register_parameter(full_name, val)

        assert self.posterior.has_rsample == True, \
                "posterior distribution should have '.has_rsample == True'"

        # if one want to calculate kl divergence, we check that it is implemented
        # if self.extra_calculations.kl_divergence:
            # _ = self.get_kl()
    @abstractmethod
    def get_initargs(self, shape) -> InitargsHandler:
        """
        init parameters for prior and posterior initialization

        for prior parameters, set require_grad=False
        for posterior parameters, set require_grad=True
        """

    def get_kl(self,):
        """
        this function may raise 'NotImplementedError' if there is no method registered via 'register_kl()'

        More information:
        https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.kl:~:text=PROPERTY%20variance-,KL%20Divergence,-torch.distributions.kl
        """
        return self.kl_func(self.prior, self.posterior)
    
    def prior_likelihood(self, p: torch.tensor):
        return self.prior.log_prob(p)
    
    def posterior_likelihood(self, p: torch.tensor):
        return self.posterior.log_prob(p)
    
    def rsample(self) -> torch.Tensor:
        param = self.posterior.rsample()
        return param
    
class NormalMixNormalReparam(BasePriorPosteriorReparam):
    PRIOR: D.MixtureSameFamily = D.MixtureSameFamily
    POSTERIOR: DistributionType = ReparamNormal

    @classmethod
    def factory(cls, num_classes: int = 2, 
                 class_weights: torch.Tensor = torch.tensor([0.1, 0.9]), 
                 class_scales: torch.Tensor = torch.tensor([0.001, 10.])) -> None:
        """
        Overrided factory function, since this Reparametrization needs hyperparameters for initialization
        """
        
        assert num_classes == len(class_weights) == len(class_scales)
        assert torch.isclose(sum(class_weights), torch.tensor(1.))
        assert torch.all(class_scales > 0)
        assert torch.all(class_weights > 0)

        def f(shape):
            return cls(shape, class_weights=class_weights, class_scales=class_scales)
        return f
    
    def __init__(self, shape, class_weights, class_scales) -> None:
        self.class_weights = class_weights
        self.class_scales = class_scales
        super().__init__(shape)

    def get_initargs(self, shape) -> InitargsHandler:
        """
        init loc and scale parameters

        for prior parameters, set require_grad=False
        for posterior parameters, set require_grad=True
        
        """
        loc = nn.Parameter(torch.rand(shape), requires_grad=True)
        scale_f = nn.Parameter(torch.randn(shape), requires_grad=True)

        posterior_params = {"loc": loc, "scale": scale_f}

        # prior params

        num_classes = len(self.class_weights)

        mix = D.Categorical(self.class_weights)

        scales = torch.rand(num_classes, *shape)
        for i in range(num_classes):
            scales[i] = self.class_scales[i]

        comp = D.Independent(D.Normal(torch.zeros(num_classes, *shape), scales), len(shape))

        prior_params = {"mixture_distribution": mix ,"component_distribution":comp}
        initargs = InitargsHandler(prior_initargs=prior_params, 
                                   posterior_initargs=posterior_params)
        return initargs            
    
class NormalNormalReparam(BasePriorPosteriorReparam):
    PRIOR: DistributionType = D.Normal
    POSTERIOR: DistributionType = ReparamNormal
    
    def get_initargs(self, shape) -> InitargsHandler:
        """
        init loc and scale parameters

        for prior parameters, set require_grad=False
        for posterior parameters, set require_grad=True
        
        """
        loc = nn.Parameter(torch.rand(shape), requires_grad=True)
        scale_f = nn.Parameter(torch.randn(shape), requires_grad=True)

        posterior_params = {"loc": loc, "scale": scale_f}

        loc_prior = nn.Parameter(torch.zeros(shape), requires_grad=False)
        scale_prior = nn.Parameter(torch.ones(shape), requires_grad=False)
        prior_params = {"loc": loc_prior, "scale": scale_prior }
        initargs = InitargsHandler(prior_initargs=prior_params, 
                                   posterior_initargs=posterior_params)
        return initargs
    