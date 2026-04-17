from __future__ import annotations

import torch
from torch import nn

from src.models.net.model_base import ModelUtilMixin


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class HybridBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        self.conv_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

        self.layer_scale1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.layer_scale2 = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        attn_tokens, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)
        attn_map = attn_tokens.transpose(1, 2).reshape(b, c, h, w)

        conv_map = self.conv_mixer(x)
        mixed = attn_map + conv_map
        x = x + self.drop_path(self.layer_scale1.view(1, c, 1, 1) * mixed)

        tokens = x.flatten(2).transpose(1, 2)
        mlp_tokens = self.mlp(self.norm2(tokens))
        mlp_map = mlp_tokens.transpose(1, 2).reshape(b, c, h, w)
        x = x + self.drop_path(self.layer_scale2.view(1, c, 1, 1) * mlp_map)

        return x


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DHVT(ModelUtilMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_coarse_classes: int = 20,
        dims: tuple[int, int, int] = (192, 320, 512),
        depths: tuple[int, int, int] = (3, 6, 4),
        heads: tuple[int, int, int] = (3, 5, 8),
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.2,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()

        if not (len(dims) == len(depths) == len(heads) == 3):
            raise ValueError("DHVT expects 3-stage configuration for dims/depths/heads.")
        for stage_idx, (dim, num_heads) in enumerate(zip(dims, heads), start=1):
            if dim % num_heads != 0:
                raise ValueError(
                    f"Invalid DHVT config at stage{stage_idx}: embed_dim({dim}) must be divisible by num_heads({num_heads})."
                )

        self.flatten = nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        total_blocks = sum(depths)
        dpr = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        self.stage1 = nn.Sequential(
            *[
                HybridBlock(
                    dim=dims[0],
                    num_heads=heads[0],
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=attn_dropout,
                    drop_path=dpr[dp_idx + i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depths[0])
            ]
        )
        dp_idx += depths[0]

        self.down1 = Downsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(
            *[
                HybridBlock(
                    dim=dims[1],
                    num_heads=heads[1],
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=attn_dropout,
                    drop_path=dpr[dp_idx + i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depths[1])
            ]
        )
        dp_idx += depths[1]

        self.down2 = Downsample(dims[1], dims[2])
        self.stage3 = nn.Sequential(
            *[
                HybridBlock(
                    dim=dims[2],
                    num_heads=heads[2],
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=attn_dropout,
                    drop_path=dpr[dp_idx + i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depths[2])
            ]
        )

        self.norm = nn.BatchNorm2d(dims[2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_drop = nn.Dropout(dropout)

        self.fine_head = nn.Linear(dims[2], int(num_classes))
        self.coarse_head = nn.Linear(dims[2], int(num_coarse_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.flatten(x)
        x = self.stem(x)

        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)

        x = self.norm(x)
        x = self.pool(x).flatten(1)
        x = self.head_drop(x)

        return {
            "fine_logits": self.fine_head(x),
            "coarse_logits": self.coarse_head(x),
        }
