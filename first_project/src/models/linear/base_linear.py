import torch
from torch import nn
import torch.nn.functional as F
import math

from src.models.norm.base_norm import BaseNorm

class BaseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
        if bias :
            self.bias = nn.Parameter(torch.ones(out_features))
        else :
            self.bias = None
        self.norm = BaseNorm(in_features)
        self.reset_parameter()
        
    def reset_parameter(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        x_norm = self.norm(x)        
        return F.linear(x_norm, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
        