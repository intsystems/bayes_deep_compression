from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Optional, Iterable, Any
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm.notebook import tqdm

from src.methods.bayes.base.trainer import TrainerParams, BaseBayesTrainer
from methods.bayes.variational.net_distribution import VarBayesModuleNetDistribution
from src.methods.bayes.variational.optimization import VarKLLoss
from src.methods.bayes.variational.net import VarBayesModuleNet
from src.methods.report.base import ReportChain

class Beta_Shelduer():
    def __init__(self, beta: float,  ref = None, *args, **kwargs) -> None:
        self.ref = self
        self.beta = beta
        if(ref is not None):
            self.ref = ref
            self.beta = ref.beta
        
    def step(self, loss) -> None:
        ...
    def __float__(self) -> None:
        return self.ref.beta

class Beta_Shelduer_Plato(Beta_Shelduer):
    def __init__(self, beta: float = 1e-2, alpha: float = 1e-1, patience: int = 10, is_min: bool = True, threshold: float = 0.01, eps: float=1e-08, max_beta: float= 1., min_beta:float=1e-9, ref: Optional[Beta_Shelduer]= None):
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
        if (self.prev_opt is not None) and (loss > self.prev_opt - abs(self.prev_opt) * self.threshold):
            self.cnt_upward += 1
        else:
            self.cnt_upward = 0
        if (self.cnt_upward < self.patience) ^ (self.is_min):
            new_beta = self.alpha * self.ref.beta
            if abs(new_beta - self.ref.beta) > self.eps:
                new_beta = min(new_beta, self.max_beta)
                new_beta = max(new_beta, self.min_beta)
                self.ref.beta =  new_beta
        if self.prev_opt is None:
            self.prev_opt = loss
        elif self.is_min == True:
            self.prev_opt = min(self.prev_opt, loss)
        else:
            self.prev_opt = max(self.prev_opt, loss)
    def __float__(self):
        return self.ref.beta
    
class CallbackLoss():
    def __init__(self, *args, **kwargs) -> None:
        ...
    def __call__(self):
        ...
    def step(self, *args, **kwargs):
        ...
    def zero(self):
        ...
class CallbackLossAccuracy():
    def __init__(self) -> None:
        self.sum_acc = 0
        self.samples = 0
    def __call__(self):
        return self.sum_acc / self.samples
    def step(self, output, label):
        _, predicted = torch.max(output.data, 1) 
        self.sum_acc += (predicted == label).sum().item() / label.size(0)
        self.samples += 1
    def zero(self):
        self.sum_acc = 0
        self.samples = 0
@dataclass
class VarTrainerParams(TrainerParams):
    fit_loss: Callable
    dist_loss: VarKLLoss
    num_samples: int
    batch_size: int
    prune_threshold: float = -2.2
    beta: float = 0.02
    callback_losses: Optional[dict[CallbackLoss]]

    
class VarBayesTrainer(BaseBayesTrainer[VarBayesModuleNet]):
    def __init__(
        self,
        params: TrainerParams,
        report_chain: Optional[ReportChain],
        train_dataset: Iterable,
        eval_dataset: Iterable,
    ):
        super().__init__(params, report_chain,train_dataset, eval_dataset)

    def train(self, model: VarBayesModuleNet) -> VarBayesModuleNetDistribution:
        
        losses = [] 
        accuracies = [] 
        val_losses = [] 
        val_accuracies = [] 
        # Train the model 
        device = model.device
        for epoch in tqdm(range(self.params.num_epochs)): 
            for i, (objects, labels) in enumerate(self.train_dataset): 
               
            
            
            losses.append(total_loss.item())   
            

            eval_result = self.eval()
            val_accuracies.append(eval_result.val_accuracy) 
            val_losses.append(eval_result.val_loss)   
            if(i % 10 == 0):
                torch.save(model.state_dict(), 'model.pt')
            if isinstance(self.report_chain, ReportChain):
                callback_dict = {
                    'num_epoch': epoch+1,
                    'total_num_epoch': self.params.num_epochs,
                    'total_loss': total_loss.item(),
                    'kl_loss': KL_loss_total,
                    'fit_loss': fit_loss_total / self.params.num_samples,
                    'val_total_loss': eval_result.val_loss,
                    'val_accuracy': eval_result.val_accuracy,
                    'cnt_prune_parameters': eval_result.cnt_prune_parameters,
                    'cnt_params': eval_result.cnt_params,
                    'beta': self.params.beta,
                    'losses': losses,
                    'accuracies': accuracies,
                    'val_losses': eval_result.val_losses,
                }
                for loss_name, custom_loss in self.callback_losses.items():
                    callback_dict[loss_name] = custom_loss()
                self.report_chain.report(callback_dict)
    def train_step(self, model: VarBayesModuleNet, fit_loss, dist_loss, callback_losses:Optional[dict[CallbackLoss]]) -> dict:
        device = model.device
        # Forward pass 
        objects=objects.to(device) 
        labels=labels.to(device) 
        fit_loss_total = 0 
        dist_losses = []
        
        for j in range(self.params.num_samples):
            param_sample_list, base_module_snapshot = model.sample()
            outputs = model(objects)
            fit_loss_total = fit_loss_total + self.params.fit_loss(outputs, labels)  
            dist_losses.append(self.params.dist_loss(outputs, labels, model.posterior, model.prior))
            
            if callback_losses is not None:
                for custom_loss in callback_losses.values():
                     custom_loss.step(outputs, labels)
        dist_loss_total = self.params.dist_loss.aggregate(dist_losses)
        total_loss = (fit_loss_total) / (self.params.num_samples) + self.params.beta * dist_loss_total

        # Backward pass and optimization 
        self.params.optimizer.zero_grad() 
        total_loss.backward() 
        self.params.optimizer.step()
        return {'total_loss': total_loss}
        
    def eval(self, model: VarBayesModuleNet) -> dict:
        # Evaluate the model on the validation set 
        device = model.device
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            model.prune({'threshold': self.params.prune_threshold})
            for objects, labels in self.eval_dataset: 
                labels=labels.to(device) 
                objects=objects.to(device) 
                fit_loss_total = 0 
                model.set_map_params()
                outputs = model(objects)
                KL_loss_total = self.params.kl_loss(model.posterior_params, model.prior_params)
                # calculate fit loss based on mean and standard deviation of output
                fit_loss_total = fit_loss_total + self.params.fit_loss(outputs, labels)  
                total_loss = fit_loss_total + float(self.params.beta)* KL_loss_total
                val_loss += total_loss.item() 
                
                _, predicted = torch.max(outputs.data, 1) 
            total = labels.size(0) 
            
            correct = (predicted == labels).sum().item() 
            val_acc += correct / total 
            
            
            cnt_prune_parameters = model.prune_stats()
            cnt_params = model.total_params()
        out = VarBayesTrainer.EvalReuslt(val_acc, val_loss, cnt_prune_parameters, cnt_params)
        return out 
