from torch import Tensor, nn

from src.models.linear.pyramidnet_linear import PyramidBasicBlock
from src.models.net.model_base import ModelUtilMixin


class PyramidNet(ModelUtilMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 100,
        num_coarse: int = 20,
        depth: int = 110,
        alpha: int = 48,
        **kwargs,
    ):
        super().__init__()

        if (depth - 2) % 6 != 0:
            raise ValueError(f"PyramidNet depth must satisfy 6n+2, got: {depth}")

        n = (depth - 2) // 6
        add_rate = float(alpha) / (3 * n)

        self.flatten = nn.Identity()
        self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)

        self.in_channels = 16
        self.featuremap_dim = float(self.in_channels)

        self.layer1 = self._make_group(num_blocks=n, stride=1, add_rate=add_rate)
        self.layer2 = self._make_group(num_blocks=n, stride=2, add_rate=add_rate)
        self.layer3 = self._make_group(num_blocks=n, stride=2, add_rate=add_rate)

        self.feature_extractor = nn.Sequential(
            self.stem,
            self.stem_bn,
            self.layer1,
            self.layer2,
            self.layer3,
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.fine_head = nn.Linear(self.in_channels, num_classes)
        self.coarse_head = nn.Linear(self.in_channels, num_coarse)

        self.classifier = nn.Identity()

    def _make_group(self, num_blocks: int, stride: int, add_rate: float) -> nn.Sequential:
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
                )
            )
            self.in_channels = out_channels
            current_stride = 1
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> dict:
        feat = self.feature_extractor(x)
        feat = self.pool(feat)
        return {
            "fine_logits": self.fine_head(feat),
            "coarse_logits": self.coarse_head(feat),
        }