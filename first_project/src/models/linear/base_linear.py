from torch import nn

class BaseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_features)
    
    def forward(self, x):
        x = self.conv(x)
        return self.norm(x)
    
    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_features}, out_channels={self.out_features}, '
            f'kernel_size=3, stride=1, padding=1, bias={self.conv.bias is not None}'
        )
        