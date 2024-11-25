from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Union

import torch
from src.methods.bayes.base.net_distribution import BaseNetDistributionPruner
from src.methods.bayes.base.trainer import BaseBayesTrainer, TrainerParams
from src.methods.bayes.variational.net import VarBayesNet
from src.methods.bayes.variational.net_distribution import VarBayesModuleNetDistribution
from src.methods.bayes.variational.optimization import VarDistLoss
from src.methods.report.base import ReportChain
from tqdm.notebook import tqdm


class Beta_Scheduler:
    """
    Abstract class for beta scheduler a scale parameter between
    Distance loss and Data loss, the higher beta value is the more important
    Distance loss is. It is recommended to start with small value (< 0.1) and
    increase it through learning.
    """

    def __init__(self, beta: float, ref: 'Beta_Scheduler'| 'VarTrainerParams' =None, *args, **kwargs) -> None:
        """_summary_

        Args:
            beta (float): initial beta value
            ref (Beta_Scheduler| VarTrainerParams): reference to trainer parameters which contains beta attribute 
                or another Beta_Shelduer
        """
        self.ref = self
        self.beta = beta
        if ref is not None:
            self.ref = ref
            "Refernce to trainer parameters which contain beta attribute"
            self.beta = ref.beta
            "Beta value"

    def step(self, loss) -> None:
        """
        Abstract class for beta scheduler a scale parameter between
        Distance loss and Data loss, the higher this value is the more important
        Distance loss is. It is recommended to start with small value (< 0.1) and
        increase it through learning.
        """
        ...

    def __float__(self) -> None:
        return self.ref.beta


class Beta_Scheduler_Plato(Beta_Scheduler):
    """
    Class for plato beta scheduler a scale parameter between
    Distance loss and Data loss, the higher this value is the more important
    Distance loss is. It is recommended to start with small value (< 0.1) and
    increase it through learning. It increase it when target loss stops
    improving fo more then patience steps. Use ref to specify trainer parameters
    which contain beta attribute in other way you should assign it manually
    """

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
        ref=None,
    ):
        """_summary_

        Args:
            beta (float): initial beta value
            alpha (float): factor of beta value by which it multiplied
            patience (int): Number of steps of loss non-improvement which schelduer should tolerate 
                before changing beta value
            is_min (bool): Is it minimiztion problem. So method would consider that loss
                stops improving when it starts maximizing
            threshold (float): Algorithm consider loss stops imporving if the next loss is 
                more(for minimiztion) then min loss more then threshold
            eps (float): Mininum change of beta. If delta is less there will be no change
            max_beta (float): Beta would be maxcliped to this value. To work properly
                ref should reference to trainer parameter
            min_beta (float): Beta would be mincliped to this value. To work properly
                ref should reference to trainer parameter
            ref (Beta_Scheduler| VarTrainerParams): reference to trainer parameters which contains beta attribute 
                or another Beta_Shelduer
        """
        super().__init__(beta, ref)
        self.cnt_upward = 0
        self.prev_opt = None
        self.patience = patience
        """Number of steps of loss non-improvement which schelduer should tolerate 
        before changing beta value"""
        self.alpha = alpha
        """Factor by which beta value changing"""
        self.is_min = is_min
        """Is it minimiztion problem. So method would consider that loss
        stops improving< when it starts maximizing"""
        self.max_beta = max_beta
        """Beta would be maxcliped to this value. To work properly
        ref should reference to trainer parameter"""
        self.min_beta = min_beta
        """Beta would be mincliped to this value. To work properly
        ref should reference to trainer parameter"""
        self.threshold = threshold
        """Algorithm consider loss stops imporving if the next loss is 
        more(for minimiztion) then min loss more then threshold"""
        self.eps = eps
        """Mininum change of beta. If delta is less there will be no change"""

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
    """Abstract class for additional losses that should
    be calculated each train step"""

    def __init__(self, *args, **kwargs) -> None: ...

    def __call__(self):
        """Function should return aggregated loss that is
        calculed through whole states and return it
        """
        ...

    def step(self, *args, **kwargs) -> None:
        """Method should calculate train step loss
        using information that train step provided"""
        ...

    def zero(self) -> None:
        """Method should resets values to initial"""
        ...


class CallbackLossAccuracy(CallbackLoss):
    """Class for accuracy losses for classification problem
    to add them in callback"""

    def __init__(self) -> None:
        self.zero()

    def __call__(self) -> float:
        """Function returns mean accuracy for whole
        train steps.

        Returns:
            float: mean accuracy
        """
        return self.sum_acc / self.samples

    def step(self, output, label) -> None:
        """Method should calculate accuracy for train

        Args:
            output (torch.tensor): predicted logits for each class
            label (torch.tensor): validatation labels for each object
        """
        _, predicted = torch.max(output.data, 1)
        self.sum_acc += (predicted == label).sum().item() / label.size(0)
        self.samples += 1

    def zero(self) -> None:
        """Method resets values to initial"""
        self.sum_acc = 0
        self.samples = 0


@dataclass
class VarTrainerParams(TrainerParams):
    """Class for VarBayesTrainer parameters"""

    fit_loss: Callable
    """Loss for data of non-bayesian model. There could be used 
    any usual loss that is appropiated for this model and task."""
    dist_loss: VarDistLoss
    """Loss for distributions of bayesian-model. This loss set up method
    that you are choose to use. Select it carefully as not all 
    losses and distribution are compatible"""
    num_samples: int
    """Number of samples that are used for estimation of losses.
    Increasing it lowers variance and improves learning in cost of computaion time"""
    prune_threshold: float = -2.2
    """Threshold by which parameters are pruned. The lower it is the more are pruned.
    Could be any real number."""
    beta: float = 0.02
    """Beta is scale factor betwenn distance loss and data loss. The higher beta value 
    is the more important distance loss is. It is recommended to start with 
    small value (< 0.1) and increase it through learning."""
    callback_losses: Optional[dict[CallbackLoss]] = None
    """All additional losses that should be added to callback"""


class VarBayesTrainer(BaseBayesTrainer[VarBayesNet]):
    """Trainer that is used for all variational methods all it parameters are stored
    in params(VarTrainerParams) attribute"""

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
        post_train_step_func: Union[
            None, list[Callable[[BaseBayesTrainer, TrainResult], None]]
        ] = None,
    ):
        """_summary_

        Args:
            params (TrainerParams): trianing params that is used to fine-tune training
            report_chain (Optional[ReportChain]): All callback that should be return by each epoch
            train_dataset (Iterable): Dataset on which model should be trained
            eval_dataset (Iterable): Dataset on which epoch of training model should be evaluated
            post_train_step_func (Union[None, list[Callable[[BaseBayesTrainer, TrainResult], None]]):
                functions that should be executed after each train step
        ] 
        """
        super().__init__(params, report_chain, train_dataset, eval_dataset)

        # B8006
        if post_train_step_func is None:
            post_train_step_func = []

        self.post_train_step_func = post_train_step_func

    def train(self, model: VarBayesNet) -> VarBayesModuleNetDistribution:
        """It simply train provided model with tarin parameters that is stores in params

        Args:
            model (VarBayesModuleNet): Any variational bayesian model that should be trained

        Returns:
            VarBayesModuleNetDistribution: Distribution of variational nets that could be used
            to sample models or getting map estimation (the most probable) by result of train.
        """
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
            for _, (objects, labels) in enumerate(self.train_dataset):
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
            if epoch % 10 == 0:
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

    def train_step(self, model: VarBayesNet, objects, labels) -> TrainResult:
        """
        Train step for specific batch
        """
        device = model.device
        # Forward pass
        objects = objects.to(device)
        labels = labels.to(device)
        fit_loss_total = 0
        dist_losses = []
        fit_losses = []
        for _ in range(self.params.num_samples):
            param_sample_dict = model.sample()
            outputs = model(objects)
            fit_losses.append(self.params.fit_loss(outputs, labels))
            dist_losses.append(
                self.params.dist_loss(
                    param_sample_dict=param_sample_dict,
                    posterior=model.posterior,
                    prior=model.prior,
                )
            )

            if self.params.callback_losses is not None:
                for custom_loss in self.params.callback_losses.values():
                    custom_loss.step(outputs, labels)
        aggregation_output = self.params.dist_loss.aggregate(
            fit_losses, dist_losses, float(self.params.beta)
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

        # print(dict(model.named_parameters()))
        """
        for name, params in dict(model.named_parameters()).items():
            print(name)
            print(params.grad)
        """

        return VarBayesTrainer.TrainResult(total_loss, fit_loss_total, dist_loss_total)

    def __post_train_step(self, train_result: TrainResult) -> None:
        """Functions that should exectuted after each taraining"""
        for func in self.post_train_step_func:
            func(self, train_result)

    def eval(
        self, model: VarBayesNet, eval_dataset
    ) -> "VarBayesTrainer.EvalResult":
        """Evalute model on dataset using stored train parameters
        Args:
            model (VarBayesModuleNet): Variational bayesian model that should be evaulted
            eval_dataset: datatest on which model should be evaluted
        Returns:
            VarBayesTrainer.EvalResult: Evaluation result that is stored in VarBayesTrainer.EvalResult format"""
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
                # Sampling model's parameters to evaluate distance between variational and prior
                for _ in range(self.params.num_samples):
                    param_sample_dict = net_distributon.sample_params()
                    dist_losses.append(
                        self.params.dist_loss(
                            param_sample_dict=param_sample_dict,
                            posterior=model.posterior,
                            prior=model.prior,
                        )
                    )

                # Set model parameters to map and prune it
                net_distributon.set_map_params()
                net_distributon_pruner.prune(self.params.prune_threshold)
                # get basic model for evaluation
                eval_model = net_distributon.get_model()
                outputs = eval_model(objects)
                for _ in range(self.params.num_samples):
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

    def eval_thresholds(
        self, model: VarBayesNet, thresholds: List[float]
    ) -> List["VarBayesTrainer.EvalResult"]:
        """Simillar to eval() but evaluate for a list of prune threshold

        Args:
            model (VarBayesModuleNet): Variational bayesian model that should be evaulted
            thresholds (List[float]): list of prune thresholds on which model should be evaluted
            
        Returns:
            List[VarBayesTrainer.EvalResult]: Evaluation result that is stored in VarBayesTrainer.EvalResult format"""
        old_thr = self.params.prune_threshold

        eval_resilts = []
        for thr in thresholds:
            self.params.prune_threshold = thr
            tmp_eval = self.eval(model, self.eval_dataset)
            eval_resilts.append(tmp_eval)

        self.params.prune_threshold = old_thr

        return eval_resilts
