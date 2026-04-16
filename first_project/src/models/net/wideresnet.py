from torch import nn

from src.models.linear.wrn_linear import WideBasicBlock
from src.models.net.model_base import ModelUtilMixin


class WideResNet(ModelUtilMixin):
    def __init__(
        self,
        in_channels: int,
        depth: int = 28,
        widen_factor: int = 10,
        dropout: float = 0.0,
        num_classes: int = 10,
        num_coarse_classes: int = 20,
    ):
        super().__init__()

        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must satisfy 6n+4, got: {depth}")

        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.flatten = nn.Identity()
        self.stem = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_group(widths[0], widths[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_group(widths[1], widths[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_group(widths[2], widths[3], n, stride=2, dropout=dropout)

        self.feature_extractor = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            nn.BatchNorm2d(widths[3]),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            self.pool,
            nn.Flatten(),
            nn.Linear(widths[3], num_classes),
        )

        self.fine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(widths[3], num_classes),
        )

    @staticmethod
    def _make_group(in_channels: int, out_channels: int, num_blocks: int, stride: int, dropout: float):
        layers = [WideBasicBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        h = self.feature_extractor(x)
        return self.fine_head(h)
