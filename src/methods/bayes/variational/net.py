from typing import Dict, List

import torch
import torch.nn as nn

from src.methods.bayes.base.net import BaseBayesModuleNet, BayesModule
from src.methods.bayes.variational.distribution import LogUniformVarDist, NormalDist, NormalReparametrizedDist
from src.utils.attribute import del_attr, set_attr
from src.methods.bayes.variational.optimization import VarDistLoss, VarRenuiLoss

from src.methods.bayes.variational.trainer import VarBayesTrainer, VarTrainerParams
from src.methods.bayes.variational.trainer import VarBayesTrainer, VarTrainerParams, Beta_Scheduler_Plato, CallbackLossAccuracy

from src.methods.report.base import ReportChain
from src.methods.report.variational import VarBaseReport

class BaseBayesVarModule(BayesModule):

    @classmethod
    def from_module(cls, module: nn.Module):
        var_module = cls(module)
        model = VarBayesModuleNet(module, nn.ModuleList([var_module]))
        return model

    is_posterior_trainable = True

    def flush_weights(self) -> None:
        for name, p in list(self.base_module.named_parameters()):
            del_attr(self.base_module, name.split("."))
            set_attr(self.base_module, name.split("."), torch.zeros_like(p))

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module=module)
        self.flush_weights()


class VarBayesModuleNet(BaseBayesModuleNet):
    # TODO: set default prior distribution to use renui loss
    # as a default parameter loss function
    prior_distribution_class = NormalReparametrizedDist

    def __init__(self, base_module: nn.Module, module_list: nn.ModuleList):
        super().__init__(base_module, module_list)

    def fit(self, dataset: torch.utils.data.Dataset,
            trainer_params: Dict[str, any] = {
                    "num_epochs": 1,
                    "beta": 1e-2,
                    "prune_threshold": -5,
                    "num_samples": 5
                },
            dataloader_params: Dict[str, any] = {
                    "val_ratio": 0.2,
                    "batch_size": 128
                },
            callback_losses: Dict[str, nn.Module] = {
                    'accuracy': CallbackLossAccuracy()
                },
            optimizer_cls=torch.optim.Adam,
            optimizer_params: Dict[str, any] = {
                    "lr": 1e-3
                },
            fit_loss=nn.CrossEntropyLoss(reduction="sum"),
            distribution_loss: VarDistLoss = VarRenuiLoss(),
            report_chain=ReportChain([VarBaseReport()])
            ):

        # model = self
        optimizer = optimizer_cls(self.parameters(), **optimizer_params)

        # prepare dataloaders from data
        val_size = int(dataloader_params["val_ratio"] * len(dataset))
        train_size = len(dataset) - val_size
        batch_size = dataloader_params['batch_size']
        t_dataset, v_dataset = torch.utils.data.random_split(dataset, \
                                                        [train_size, val_size])

        # Create DataLoaders for the training and validation sets
        train_loader = torch.utils.data.DataLoader(t_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True)

        eval_loader = torch.utils.data.DataLoader(v_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True)

        train_params = VarTrainerParams(
                optimizer=optimizer,
                fit_loss=fit_loss,
                dist_loss=distribution_loss,
                callback_losses=callback_losses,
                **trainer_params
                )

        trainer = VarBayesTrainer(train_params, report_chain, train_loader, eval_loader)
        trainer.train(self)
        return self, trainer

    @property
    def posterior_params(self) -> dict[str, dict[str, nn.Parameter]]:
        return self.get_params("posterior")

    @property
    def prior_params(self) -> dict[str, dict[str, nn.Parameter]]:
        return self.get_params("prior")


class LogUniformVarBayesModule(BaseBayesVarModule):
    def __init__(self, module: nn.Module) -> None:
        self.posterior_distribution_cls = LogUniformVarDist
        self.prior_distribution_cls = None
        self.is_prior_trainable = False
        super().__init__(module)


class NormalVarBayesModule(BaseBayesVarModule):
    """Envelope for nn.Modules with the same normal prior on all scalar paramters and factorized normal
    distributions as the variational distibution on paramters. The prior is not required here as
    its optimal form can be computed analytically.
    """

    def __init__(self, module: nn.Module) -> None:
        self.posterior_distribution_cls = NormalReparametrizedDist
        self.prior_distribution_cls = NormalDist
        self.is_prior_trainable = False
        super().__init__(module)
