import torch.nn as nn


class FineLogitOnlyWrapper(nn.Module):
    """Wraps a multi-output model to return only fine_logits as a plain tensor.

    Required by official_eval.evaluate() which expects model(x) → Tensor.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out["fine_logits"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
