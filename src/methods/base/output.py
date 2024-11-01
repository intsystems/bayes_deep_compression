import torch.nn
from typing import Generic

from typing import TypeVar
from src.methods.base.result import BaseOutputModel

T = TypeVar('T', bound=BaseOutputModel)

class BaseBayesModel(torch.nn.Module,Generic[T]):
    def forward(self, x:torch.Tensor[torch.float32]) -> T: ...