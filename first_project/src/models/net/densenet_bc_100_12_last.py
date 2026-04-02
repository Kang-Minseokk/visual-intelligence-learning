from torch import nn

from src.models.linear.densenet_bc_100_12_last import DenseBlockBC, TransitionLayerBC
from src.models.net.model_base import ModelUtilMixin


class DenseNetBC100x12Last(ModelUtilMixin):
    """DenseNet-BC (L=100, k=12) for CIFAR-sized inputs."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        growth_rate: int = 12,
        compression: float = 0.5,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        block_config=(16, 16, 16),
    ):
        super().__init__()

        if tuple(block_config) != (16, 16, 16):
            raise ValueError(f"DenseNet-BC(100,12) requires block_config=(16,16,16), got: {block_config}")
        if growth_rate != 12:
            raise ValueError(f"DenseNet-BC(100,12) requires growth_rate=12, got: {growth_rate}")
        if not (0 < compression <= 1.0):
            raise ValueError(f"compression must be in (0, 1], got: {compression}")

        num_init_features = 2 * growth_rate

        self.flatten = nn.Identity()
        features = [
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = DenseBlockBC(
                num_layers=int(num_layers),
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            features.append(block)
            num_features = num_features + int(num_layers) * growth_rate

            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                features.append(
                    TransitionLayerBC(
                        in_channels=num_features,
                        out_channels=out_features,
                        drop_rate=drop_rate,
                    )
                )
                num_features = out_features

        features.append(nn.BatchNorm2d(num_features))
        features.append(nn.ReLU(inplace=True))
        self.feature_extractor = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

        self.initialize_weights_kaiming()
