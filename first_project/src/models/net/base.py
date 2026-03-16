from src.models.linear.base_linear import BaseLinear
from torch import nn
from src.models.net.model_base import build_layers, ModelUtilMixin

class BaseNet(ModelUtilMixin):
    def __init__(self, 
                 in_features, 
                 hidden_features,
                 num_classes: int = 10, 
                 depth: int = 1,                                  
                 dropout: int = 0, 
                 activation_option: str = "gelu"
                 ):
        super().__init__()
        
        self.flatten = nn.Identity()
        layers, width = build_layers(
            in_features = in_features,
            hidden_features = hidden_features,
            depth = depth,
            dropout = dropout,
            linear_cls = BaseLinear,
            activation_option = activation_option,            
        )
        layers.append(nn.BatchNorm2d(width))
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, num_classes, bias=True),
        )
        
    
    
