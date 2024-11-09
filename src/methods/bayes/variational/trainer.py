from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Optional, Iterable
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm.notebook import tqdm

from src.methods.bayes.base.trainer import TrainerParams, BaseBayesTrainer
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
    
@dataclass
class VarTrainerParams(TrainerParams):
    fit_loss: nn.Module
    kl_loss: VarKLLoss
    num_samples: int
    batch_size: int
    val_percent: float
    prune_threshold: float | Beta_Shelduer = -2.2
    beta: float | Beta_Shelduer = 0.02
    beta_KL: Optional[Beta_Shelduer] = None

    
class VarBayesTrainer(BaseBayesTrainer):
    def __init__(
        self,
        params: VarTrainerParams,
        report_chain: Optional[ReportChain],
        dataset:Iterable,
    ):
        super().__init__(params, report_chain, dataset)

    def train(self, model: VarBayesModuleNet) -> None:
        val_size    = int(self.params.val_percent * len(self.dataset)) 
        train_size  = len(self.dataset) - val_size 
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset,  
                                                                [train_size,  
                                                                    val_size]) 
        
        # Create DataLoaders for the training and validation sets 
        train_loader = torch.utils.data.DataLoader(train_dataset,  
                                                batch_size=self.params.batch_size,  
                                                shuffle=True, 
                                                pin_memory=True) 
        val_loader = torch.utils.data.DataLoader(val_dataset,  
                                                batch_size=self.params.batch_size,  
                                                shuffle=False, 
                                                pin_memory=True) 
        losses = [] 
        accuracies = [] 
        val_losses = [] 
        val_accuracies = [] 
        # Train the model 


        device = model.device
        for epoch in tqdm(range(self.params.num_epochs)): 
            for i, (images, labels) in enumerate(train_loader): 
                # Forward pass 
                images=images.to(device) 
                labels=labels.to(device) 
                fit_loss_total = 0 
                for j in range(self.params.num_samples):
                    outputs = model(images)
                    # calculate fit loss based on mean and standard deviation of output
                    #fit_loss_total = fit_loss_total + criterion(outputs, labels)  
                    
                    KL_loss_total = self.params.kl_loss(model.posterior_params)
                    fit_loss_total = fit_loss_total + self.params.fit_loss(outputs, labels)  
                total_loss = (fit_loss_total) / (self.params.num_samples) + float(self.params.beta) * KL_loss_total
                
                # Backward pass and optimization 
                self.params.optimizer.zero_grad() 
                total_loss.backward() 
                self.params.optimizer.step()
                if isinstance(self.params.beta, Beta_Shelduer):
                    self.params.beta.step(fit_loss_total)
                if isinstance(self.params.beta_KL, Beta_Shelduer): 
                    self.params.beta_KL.step(KL_loss_total)  
        
                _, predicted = torch.max(outputs.data, 1) 
            acc = (predicted == labels).sum().item() / labels.size(0) 
            accuracies.append(acc) 
            losses.append(total_loss.item())   
                
            # Evaluate the model on the validation set 
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                model.prune({'threshold': self.params.prune_threshold})
                for images, labels in val_loader: 
                    labels=labels.to(device) 
                    images=images.to(device) 
                    fit_loss_total = 0 
                    for j in range(self.params.num_samples):
                        model.set_map_params()
                        outputs = model(images)
                        KL_loss_total = self.params.kl_loss(model.posterior_params)
                        # calculate fit loss based on mean and standard deviation of output
                        fit_loss_total = fit_loss_total + self.params.fit_loss(outputs, labels)  
                    total_loss = (fit_loss_total)/(self.params.num_samples) + float(self.params.beta)* KL_loss_total
                    val_loss += total_loss.item() 
                    
                    _, predicted = torch.max(outputs.data, 1) 
                total = labels.size(0) 
                
                correct = (predicted == labels).sum().item() 
                val_acc += correct / total 
                val_accuracies.append(acc) 
                val_losses.append(total_loss.item())   
                
                cnt_prune_parameters = model.prune_stats()
                cnt_params = model.total_params()
            if(i % 10 == 0):
                torch.save(model.state_dict(), 'model.pt')
                
            print(f'Epoch [{epoch+1}/{self.params.num_epochs}],Loss:{total_loss.item()}, KL Loss: {KL_loss_total}. FitLoss: {fit_loss_total / self.params.num_samples}, Validation Loss:{val_loss},Accuracy:{acc},Validation Accuracy:{val_acc}, Prune parameters: {cnt_prune_parameters}/{cnt_params}, Beta: {float(self.params.beta)}')