import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PyramidBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.pad_channels = out_channels - in_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(out_channels)

        # shortcut에는 학습 파라미터 없음 (zero-padding 방식)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def _shortcut(self, x: Tensor) -> Tensor:
        # stride=2이면 average pooling으로 spatial 축소
        if self.stride != 1:
            x = F.avg_pool2d(x, 2, 2)

        # 채널 부족분을 0으로 패딩
        if self.pad_channels > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.pad_channels))

        return self.shortcut_bn(x)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        return out + self._shortcut(x)