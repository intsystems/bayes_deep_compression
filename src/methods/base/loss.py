     

class BaseLoss(torch.nn.Module,Generic[T]):
   def forward(self, model_output: T) -> torch.float32: ..
class BayesModel(torch.nn.Module,Generic[T):
    def forward(self, x:torch.Tensor[torch.float32]) -> T: ...