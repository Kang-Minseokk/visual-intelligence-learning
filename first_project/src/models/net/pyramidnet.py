from torch import nn

from src.models.linear.pyramidnet_linear import PyramidBasicBlock
from src.models.net.model_base import ModelUtilMixin


class PyramidNet(ModelUtilMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        depth: int = 110,
        alpha: int = 48,
        dropout: float = 0.0,
    ):
        super().__init__()

        if (depth - 2) % 6 != 0:
            raise ValueError(f"PyramidNet depth must satisfy 6n+2, got: {depth}")

        n = (depth - 2) // 6
        add_rate = float(alpha) / (3 * n)

        self.flatten = nn.Identity()
        self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.in_channels = 16
        self.featuremap_dim = float(self.in_channels)

        self.layer1 = self._make_group(num_blocks=n, stride=1, add_rate=add_rate, dropout=dropout)
        self.layer2 = self._make_group(num_blocks=n, stride=2, add_rate=add_rate, dropout=dropout)
        self.layer3 = self._make_group(num_blocks=n, stride=2, add_rate=add_rate, dropout=dropout)

        self.feature_extractor = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_channels, num_classes),
        )

    def _make_group(self, num_blocks: int, stride: int, add_rate: float, dropout: float) -> nn.Sequential:
        layers = []
        current_stride = stride
        for _ in range(num_blocks):
            self.featuremap_dim += add_rate
            out_channels = int(round(self.featuremap_dim))
            layers.append(
                PyramidBasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=current_stride,
                    drop_rate=dropout,
                )
            )
            self.in_channels = out_channels
            current_stride = 1
        return nn.Sequential(*layers)
