import torch
from torch import nn

from src.models.linear.vit_linear import Transformer
from src.models.net.model_base import ModelUtilMixin


class CCT(ModelUtilMixin):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 100,
        num_coarse_classes: int = 20,
        image_size: int = 32,
        embedding_dim: int = 256,
        transformer_layers: int = 7,
        transformer_heads: int = 4,
        mlp_ratio: float = 2.0,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        n_conv_layers: int = 1,
        tokenizer_pooling_kernel_size: int = 3,
        tokenizer_pooling_stride: int = 2,
        tokenizer_pooling_padding: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        if n_conv_layers < 1:
            raise ValueError(f"n_conv_layers must be >= 1, got: {n_conv_layers}")
        if embedding_dim % transformer_heads != 0:
            raise ValueError(
                f"embedding_dim must be divisible by transformer_heads, got {embedding_dim} and {transformer_heads}"
            )

        self.flatten = nn.Identity()

        tokenizer_layers = []
        in_ch = int(in_channels)
        for _ in range(int(n_conv_layers)):
            tokenizer_layers.extend(
                [
                    nn.Conv2d(
                        in_ch,
                        int(embedding_dim),
                        kernel_size=int(kernel_size),
                        stride=int(stride),
                        padding=int(padding),
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(
                        kernel_size=int(tokenizer_pooling_kernel_size),
                        stride=int(tokenizer_pooling_stride),
                        padding=int(tokenizer_pooling_padding),
                    ),
                ]
            )
            in_ch = int(embedding_dim)

        self.tokenizer = nn.Sequential(*tokenizer_layers)

        reduced_size = int(image_size)
        for _ in range(int(n_conv_layers)):
            reduced_size = ((reduced_size + 2 * int(padding) - int(kernel_size)) // int(stride)) + 1
            reduced_size = (
                (reduced_size + 2 * int(tokenizer_pooling_padding) - int(tokenizer_pooling_kernel_size))
                // int(tokenizer_pooling_stride)
            ) + 1
        if reduced_size <= 0:
            raise ValueError("Tokenizer reduced spatial size is non-positive. Check tokenizer hyperparameters.")

        seq_len = reduced_size * reduced_size
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, int(embedding_dim)))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(float(emb_dropout))

        self.transformer = Transformer(
            dim=int(embedding_dim),
            depth=int(transformer_layers),
            heads=int(transformer_heads),
            dim_head=int(embedding_dim) // int(transformer_heads),
            mlp_dim=int(float(mlp_ratio) * int(embedding_dim)),
            dropout=float(dropout),
        )

        self.attention_pool = nn.Linear(int(embedding_dim), 1)
        self.fine_head = nn.Linear(int(embedding_dim), int(num_classes))
        self.coarse_head = nn.Linear(int(embedding_dim), int(num_coarse_classes))
        self.feature_extractor = nn.Identity()
        self.classifier = nn.Identity()

    def forward(self, x):
        x = self.flatten(x)
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)
        attn = self.attention_pool(x).softmax(dim=1)
        pooled = (attn * x).sum(dim=1)

        return {
            "fine_logits": self.fine_head(pooled),
            "coarse_logits": self.coarse_head(pooled),
        }
