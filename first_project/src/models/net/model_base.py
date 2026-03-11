from torch import nn, Tensor
from typing import Any
from pathlib import Path

def build_layers(
    in_features: int,
    hidden_features: int,
    depth: int,
    dropout: float,
    linear_cls,    
    activation_option: str = "gelu"
):
    """
        레이어를 만들어내는 빌더 함수입니다.
    """    
    layers = []
    width = in_features
    num_layers = max(0, depth)

    if num_layers > 0:
        for _ in range(num_layers):
            layers.append(linear_cls(width, hidden_features))
            
            if activation_option == "gelu":
                # GELU + Dropout 조합
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))                 
                
            elif activation_option == "relu":
                # ReLU + Dropout 조합
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            else :
                # 설정하지 않은 경우는 활성화 및 드롭아웃 비활성화
                pass           
                 
            width = hidden_features

    return layers, width

def set_model_output_dir(output_dir: str) -> None:
    global _MODEL_OUTPUT_DIR
    _MODEL_OUTPUT_DIR = output_dir

class ModelUtilMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.Identity()
        self.classifier = nn.Identity()
        
    def set_output_dir(self, output_dir: str) -> None:        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_model_output_dir(str(self.output_dir))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        h = self.feature_extractor(x)
        y = self.classifier(h)
        return y