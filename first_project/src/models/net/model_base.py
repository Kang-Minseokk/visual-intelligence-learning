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
        self.flatten = nn.Identity()
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
    
class ViTLowbitClassifier(ModelUtilMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.to_patch_embedding = nn.Identity()
        self.dropout = nn.Identity()
        self.transformer = nn.Identity()
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Identity()        

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.to_patch_embedding(x)
        h2 = self.dropout(h1)
        h3 = self.transformer(h2)
        h4 = self.to_latent(h3)
        y = self.mlp_head(h4)
        return y

    def get_parameters_data(self):
        # get parameters for logging or analysis
        to_patch_embedding = list(self.to_patch_embedding.parameters())
        dropout = list(self.dropout.parameters())
        transformer = list(self.transformer.parameters())
        to_latent = list(self.to_latent.parameters())
        mlp_head = list(self.mlp_head.parameters())
        return {
            'to_patch_embedding': to_patch_embedding,
            'dropout': dropout,
            'transformer': transformer,
            'to_latent': to_latent,
            'mlp_head': mlp_head
        }