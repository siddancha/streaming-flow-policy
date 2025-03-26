from torch import Tensor
import torch.nn as nn

class Policy (nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, obs: Tensor) -> Tensor:
        pass
