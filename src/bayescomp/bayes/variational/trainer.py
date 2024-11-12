from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Any
import torch
from tqdm.notebook import tqdm
from bayescomp.bayes.base.trainer import TrainerParams, BaseBayesTrainer
from bayescomp.bayes.variational.net_distribution import VarBayesModuleNetDistribution
from bayescomp.bayes.base.net_distribution import BaseNetDistributionPruner
from bayescomp.bayes.variational.optimization import VarKLLoss
from bayescomp.bayes.variational.net import VarBayesModuleNet
from bayescomp.report.base import ReportChain


class Beta_Scheduler:
    def __init__(self, beta: float, ref=None, *args, **kwargs) -> None:
        self.ref = self
        self.beta = beta
        if ref is not None:
            self.ref = ref
            self.beta = ref.beta

    def step(self, loss) -> None: ...
    def __float__(self) -> None:
        return self.ref.beta


class Beta_Scheduler_Plato(Beta_Scheduler):
    def __init__(
        self,
        beta: float = 1e-2,
        alpha: float = 1e-1,
        patience: int = 10,
        is_min: bool = True,
        threshold: float = 0.01,
        eps: float = 1e-08,
        max_beta: float = 1.0,
        min_beta: float = 1e-9,
        ref: Optional[Beta_Scheduler] = None,
    ):
        super().__init__(beta, ref)
        self.cnt_upward = 0
        self.prev_opt = None
        self.patience = patience
        self.alpha = alpha
        self.is_min = is_min
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.threshold = threshold
        self.eps = eps

    def step(self, loss):
        if (self.prev_opt is not None) and (
            loss > self.prev_opt - abs(self.prev_opt) * self.threshold
        ):
            self.cnt_upward += 1
        else:
            self.cnt_upward = 0
        if (self.cnt_upward < self.patience) ^ (self.is_min):
            new_beta = self.alpha * self.ref.beta
            if abs(new_beta - self.ref.beta) > self.eps:
                new_beta = min(new_beta, self.max_beta)
                new_beta = max(new_beta, self.min_beta)
                self.ref.beta = new_beta
        if self.prev_opt is None:
            self.prev_opt = loss
        elif self.is_min:
            self.prev_opt = min(self.prev_opt, loss)
        else:
            self.prev_opt = max(self.prev_opt, loss)

    def __float__(self):
        return self.ref.beta


class CallbackLoss:
    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(self): ...
    def step(self, *args, **kwargs): ...
    def zero(self) -> None: ...


class CallbackLossAccuracy(CallbackLoss):
    def __init__(self) -> None:
        self.sum_acc = 0
        self.samples = 0

    def __call__(self):
        return self.sum_acc / self.samples

    def step(self, output, label):
        _, predicted = torch.max(output.data, 1)
        self.sum_acc += (predicted == label).sum().item() / label.size(0)
        self.samples += 1

    def zero(self) -> None:
        self.sum_acc = 0
        self.samples = 0


@dataclass
class VarTrainerParams(TrainerParams):
    fit_loss: Callable
    dist_loss: VarKLLoss
    num_samples: int
    prune_threshold: float = -2.2
    beta: float = 0.02
    callback_losses: Optional[dict[CallbackLoss]] = None


class VarBayesTrainer(BaseBayesTrainer[VarBayesModuleNet]):
    @dataclass
    class EvalResult:
        val_loss: float
        fit_loss: float
        dist_loss: float
        cnt_prune_parameters: int
        cnt_params: int
        custom_losses: dict[str, Any]

    @dataclass
    class TrainResult:
        total_loss: float
        fit_loss: float
        dist_loss: float

    def __init__(
        self,
        params: VarTrainerParams,
        report_chain: Optional[ReportChain],
        train_dataset: Iterable,
        eval_dataset: Iterable,
        post_train_step_func: list[
            Callable[[BaseBayesTrainer, TrainResult], None]
        ] = [],
    ):
        super().__init__(params, report_chain, train_dataset, eval_dataset)
        self.post_train_step_func = post_train_step_func

    def train(self, model: VarBayesModuleNet) -> VarBayesModuleNetDistribution:
        losses = []
        val_losses = []
        # Train the model
        for epoch in tqdm(range(self.params.num_epochs)):
            if self.params.callback_losses is not None:
                for custom_loss in self.params.callback_losses.values():
                    custom_loss.zero()
            train_num_batch = 0
            train_loss = 0
            train_dist_loss = 0
            train_fit_loss = 0
            for i, (objects, labels) in enumerate(self.train_dataset):
                train_output = self.train_step(model, objects, labels)
                self.__post_train_step(train_output)
                losses.append(train_output.total_loss.item())
                train_loss += train_output.total_loss
                train_dist_loss += train_output.dist_loss
                train_fit_loss += train_output.fit_loss
                train_num_batch += 1
            train_loss /= train_num_batch
            train_dist_loss /= train_num_batch
            train_fit_loss /= train_num_batch

            # Save model
            if i % 10 == 0:
                torch.save(model.state_dict(), "model.pt")

            # Train step callback
            callback_dict = {
                "num_epoch": epoch + 1,
                "total_num_epoch": self.params.num_epochs,
                "total_loss": train_loss,
                "kl_loss": train_dist_loss,
                "fit_loss": train_fit_loss,
            }
            if self.params.callback_losses is not None:
                for loss_name, custom_loss in self.params.callback_losses.items():
                    callback_dict[loss_name] = custom_loss()

            # Eval Step
            eval_result = self.eval(model, self.eval_dataset)
            val_losses.append(eval_result.val_loss)
            # Eval step callback
            callback_dict.update({"val_total_loss": eval_result.val_loss})
            if self.params.callback_losses is not None:
                for loss_name, custom_loss in self.params.callback_losses.items():
                    callback_dict["val_" + loss_name] = custom_loss()

            # Some additional information to callback
            callback_dict.update(
                {
                    "cnt_prune_parameters": eval_result.cnt_prune_parameters,
                    "cnt_params": eval_result.cnt_params,
                    "beta": self.params.beta,
                    "losses": losses,
                    "val_losses": val_losses,
                }
            )

            # if is not None let's callback
            if isinstance(self.report_chain, ReportChain):
                self.report_chain.report(callback_dict)
        return VarBayesModuleNetDistribution(model.base_module, model.posterior)

    def train_step(self, model: VarBayesModuleNet, objects, labels) -> dict:
        device = model.device
        # Forward pass
        objects = objects.to(device)
        labels = labels.to(device)
        fit_loss_total = 0
        dist_losses = []
        fit_losses = []
        for j in range(self.params.num_samples):
            param_sample_list = model.sample()
            outputs = model(objects)
            fit_losses.append(self.params.fit_loss(outputs, labels))
            dist_losses.append(
                self.params.dist_loss(param_sample_list, model.posterior, model.prior)
            )

            if self.params.callback_losses is not None:
                for custom_loss in self.params.callback_losses.values():
                    custom_loss.step(outputs, labels)
        aggregation_output = self.params.dist_loss.aggregate(
            fit_losses, dist_losses, self.params.beta
        )
        total_loss, fit_loss_total, dist_loss_total = (
            aggregation_output.total_loss,
            aggregation_output.fit_loss,
            aggregation_output.dist_loss,
        )
        # Backward pass and optimization
        self.params.optimizer.zero_grad()
        total_loss.backward()
        self.params.optimizer.step()

        return VarBayesTrainer.TrainResult(total_loss, fit_loss_total, dist_loss_total)

    def __post_train_step(self, train_result: TrainResult) -> None:
        for func in self.post_train_step_func:
            func(self, train_result)

    def eval(
        self, model: VarBayesModuleNet, eval_dataset
    ) -> "VarBayesTrainer.EvalResult":
        # Evaluate the model on the validation set
        device = model.device
        if self.params.callback_losses is not None:
            for custom_loss in self.params.callback_losses.values():
                custom_loss.zero()
        net_distributon = VarBayesModuleNetDistribution(
            model.base_module, model.posterior
        )
        net_distributon_pruner = BaseNetDistributionPruner(net_distributon)
        with torch.no_grad():
            batches = 0
            dist_losses = []
            fit_losses = []
            for objects, labels in eval_dataset:
                labels = labels.to(device)
                objects = objects.to(device)
                fit_loss = 0
                # Sampling model's parameters to evaluate distance between variational and prior
                for j in range(self.params.num_samples):
                    param_sample_list = net_distributon.sample_params()
                    dist_losses.append(
                        self.params.dist_loss(
                            param_sample_list, model.posterior, model.prior
                        )
                    )

                # Set model parameters to map and prune it
                net_distributon.set_map_params()
                net_distributon_pruner.prune(self.params.prune_threshold)
                # get basic model for evaluation
                eval_model = net_distributon.get_model()
                outputs = eval_model(objects)
                for j in range(self.params.num_samples):
                    fit_losses.append(self.params.fit_loss(outputs, labels))

                batches += 1
            aggregation_output = self.params.dist_loss.aggregate(
                fit_losses, dist_losses, self.params.beta
            )
            # calculate fit loss based on mean and standard deviation of output
            aggregation_output = self.params.dist_loss.aggregate(
                fit_losses, dist_losses, self.params.beta
            )
            val_total_loss, fit_loss_total, dist_loss_total = (
                aggregation_output.total_loss,
                aggregation_output.fit_loss,
                aggregation_output.dist_loss,
            )
            if self.params.callback_losses is not None:
                for custom_loss in self.params.callback_losses.values():
                    custom_loss.step(outputs, labels)
            cnt_prune_parameters = net_distributon_pruner.prune_stats()
            cnt_params = net_distributon_pruner.total_params()
        custom_losses = {}
        if self.params.callback_losses is not None:
            for loss_name, custom_loss in self.params.callback_losses.items():
                custom_losses["val_" + loss_name] = custom_loss()
        out = VarBayesTrainer.EvalResult(
            val_total_loss,
            fit_loss_total,
            dist_loss_total,
            cnt_prune_parameters,
            cnt_params,
            custom_losses,
        )
        return out
