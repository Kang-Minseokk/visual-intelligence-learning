"""
Compact Convolutional Transformer (CCT)
Hassani et al., "Escaping the Big Data Paradigm with Compact Transformers", 2021

Standard ViT와의 핵심 차이점:
  1. Conv Tokenizer: linear patch embedding 대신 conv 레이어로 토큰 생성
     → 소규모 데이터(CIFAR)에서 필요한 귀납적 편향(translation invariance) 제공
     → 위치 임베딩 불필요 (conv가 위치 정보를 암묵적으로 포함)
  2. Sequence Pooling: CLS 토큰 대신 attention-weighted pooling
     → 전체 시퀀스를 골고루 참조하여 소규모 데이터에서 더 안정적
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.net.model_base import ModelUtilMixin


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
    """Pre-LayerNorm transformer block (post-norm 대비 학습 안정성 높음)."""
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
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
        embed_dim:      토큰 임베딩 차원
        depth:          transformer 레이어 수
        n_heads:        multi-head attention 헤드 수
        mlp_ratio:      MLP hidden dim = embed_dim * mlp_ratio
        dropout:        MLP 및 attention output dropout
        attn_dropout:   attention weight dropout
        n_conv_layers:  conv tokenizer 레이어 수
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
        self.encoder = nn.Sequential(*[
            TransformerLayer(embed_dim, n_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
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
