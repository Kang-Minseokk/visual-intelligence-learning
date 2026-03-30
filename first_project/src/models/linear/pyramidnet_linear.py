from torch import Tensor, nn


class PyramidBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop_rate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + self.shortcut(x)
