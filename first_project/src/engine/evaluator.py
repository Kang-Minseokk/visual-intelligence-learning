from __future__ import annotations
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from ..utils.metrics import accuracy

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in tqdm(loader, desc='eval'):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_acc  += accuracy(logits, y) * bsz
            n += bsz
        return total_loss / n, total_acc / n
