"""
Compact Convolutional Transformer (CCT)
Hassani et al., "Escaping the Big Data Paradigm with Compact Transformers", 2021

Standard ViT와의 핵심 차이점:
  1. Conv Tokenizer: linear patch embedding 대신 conv 레이어로 토큰 생성
     → 소규모 데이터(CIFAR)에서 필요한 귀납적 편향(translation invariance) 제공
     → 위치 임베딩 불필요 (conv가 위치 정보를 암묵적으로 포함)
  2. Sequence Pooling: CLS 토큰 대신 attention-weighted pooling
     → 전체 시퀀스를 골고루 참조하여 소규모 데이터에서 더 안정적
  3. Stochastic Depth (DropPath): transformer 전용 핵심 정규화
     → residual block 전체를 layer별 확률로 drop
     → CNN의 dropout보다 transformer에 적합한 정규화 방식
     → DeiT, Swin, ViT 계열에서 over-fitting 해결에 필수 사용
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.net.model_base import ModelUtilMixin


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath): layer i의 residual block 전체를 drop_prob 확률로 skip.

    CNN의 dropout이 개별 activation을 끄는 것과 달리,
    이 기법은 residual F(x) 전체를 확률적으로 0으로 만들어
    "이 layer가 없을 때도 동작해야 한다"는 압력을 모델에 부여.

    학습 시: x = x + drop_path(F(x))  ← 확률 drop_prob로 F(x)가 0
    추론 시: x = x + F(x)              ← 항상 적용 (스케일 보정 포함)

    depth=14 기준 각 layer i의 drop_prob = (i / depth) * stochastic_depth_rate
    → 얕은 layer는 거의 drop하지 않고, 깊은 layer일수록 더 자주 drop
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # batch 차원만 랜덤, 나머지 차원은 broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep_prob)
        return x * noise / keep_prob   # 기댓값 유지를 위해 스케일 보정


class ConvTokenizer(nn.Module):
    """
    Convolutional patch tokenizer.

    CIFAR (32x32), n_conv_layers=2 기준 spatial 변화:
      32x32 --conv+pool--> 16x16 --conv+pool--> 8x8  →  N = 64 tokens
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        n_conv_layers: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 1,
    ):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_conv_layers):
            layers += [
                nn.Conv2d(ch, embed_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding),
            ]
            ch = embed_dim
        self.tokenizer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)                        # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)          # [B, N, embed_dim]


class TransformerLayer(nn.Module):
    """
    Pre-LayerNorm transformer block + Stochastic Depth.

    drop_path_prob: 이 layer의 stochastic depth drop 확률
      - depth=14, stochastic_depth_rate=0.1 기준
        layer 0: 0.0,  layer 7: 0.05,  layer 13: ~0.093
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = StochasticDepth(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.drop_path(self.attn(normed, normed, normed)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SequencePooling(nn.Module):
    """
    CCT의 SeqPool: attention-weighted pooling으로 시퀀스 전체를 종합.
      g(Z) = softmax(Z @ w)^T @ Z
    CLS 토큰 방식보다 소규모 데이터에서 수렴이 빠르고 성능이 높음.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_weight = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn_weight(x), dim=1)   # [B, N, 1]
        return (w * x).sum(dim=1)                        # [B, embed_dim]


class CCT(ModelUtilMixin):
    """
    CCT 모델. WRN과 동일한 출력 인터페이스(dict) 사용.

    Args:
        embed_dim:            토큰 임베딩 차원
        depth:                transformer 레이어 수
        n_heads:              multi-head attention 헤드 수
        mlp_ratio:            MLP hidden dim = embed_dim * mlp_ratio
        dropout:              MLP 및 attention output dropout
        attn_dropout:         attention weight dropout
        stochastic_depth_rate: 가장 깊은 layer의 drop_path 확률 상한
                               각 layer i의 실제 확률 = (i / depth) * stochastic_depth_rate
        n_conv_layers:        conv tokenizer 레이어 수
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 14,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        stochastic_depth_rate: float = 0.1,
        n_conv_layers: int = 2,
        num_classes: int = 100,
        num_coarse_classes: int = 20,
    ):
        super().__init__()

        self.tokenizer = ConvTokenizer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            n_conv_layers=n_conv_layers,
        )

        # 각 layer에 선형 증가하는 drop_path 확률 할당
        drop_path_probs = [
            (i / max(depth - 1, 1)) * stochastic_depth_rate
            for i in range(depth)
        ]
        self.encoder = nn.Sequential(*[
            TransformerLayer(
                embed_dim, n_heads, mlp_ratio, dropout, attn_dropout,
                drop_path_prob=drop_path_probs[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.seq_pool = SequencePooling(embed_dim)

        self.fine_head = nn.Linear(embed_dim, num_classes)
        self.coarse_head = nn.Linear(embed_dim, num_coarse_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> dict:
        x = self.tokenizer(x)    # [B, N, embed_dim]
        x = self.encoder(x)      # [B, N, embed_dim]
        x = self.norm(x)         # [B, N, embed_dim]
        h = self.seq_pool(x)     # [B, embed_dim]
        return {
            "fine_logits": self.fine_head(h),
            "coarse_logits": self.coarse_head(h),
        }
