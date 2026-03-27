from __future__ import annotations
from tqdm import tqdm
from pathlib import Path
from torch import nn
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class Trainer:
    def __init__(self, model, device, lr: float, weight_decay: float, log_interval: int, out_dir: str, config=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.config = config or {}
        
        optimizer_name = str(self.config.get("optimizer", "sgd")).lower()
        if optimizer_name == "sgd":
            self.optimizer = SGD(
                model.parameters(),
                lr=float(lr),
                momentum=float(self.config.get("momentum")),
                weight_decay=float(weight_decay),
                nesterov=bool(self.config.get("nesterov", True)),
            )
        elif optimizer_name == "adam":
            self.optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        else :
            raise ValueError(f"Unsupported Optimizer! : {optimizer_name}")
        
        label_smoothing = float(self.config.get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mixup_alpha = float(self.config.get("mixup_alpha", 0.0))
        self.scheduler = self._build_scheduler()
        self.log_interval = log_interval
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = out_dir / 'checkpoints'
        self.model.set_output_dir(str(ckpt_dir))
        
    def _build_scheduler(self):
        scheduler_name = str(self.config.get("scheduler", "cosine")).lower()
        if scheduler_name in {"none", "off", ""}:
            return None
        if scheduler_name != "cosine":
            raise ValueError(f"Unsupported scheduler! : {scheduler_name}")
        
        total_epochs = int(self.config.get("epochs", 1))
        warmup_epochs = int(self.config.get("warmup_epochs", 0))
        min_lr = float(self.config.get("min_lr", 0.0))

        if total_epochs <= 0:
            return None

        if warmup_epochs > 0:
            if total_epochs <= warmup_epochs:
                return LinearLR(
                    self.optimizer,
                    start_factor=1.0 / float(max(1, warmup_epochs)),
                    end_factor=1.0,
                    total_iters=total_epochs,
                )

            warmup = LinearLR(
                self.optimizer,
                start_factor=1.0 / float(max(1, warmup_epochs)),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_epochs - warmup_epochs),
                eta_min=min_lr,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

        return CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_epochs),
            eta_min=min_lr,
        )

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def _apply_mixup(self, x, y):
        if self.mixup_alpha <= 0.0:
            return x, y, y, 1.0

        lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item())
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_one_epoch(self, loader, epoch: int, track_flips: bool = False):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(loader, desc=f"train epoch {epoch}")
        for step, (x, y) in enumerate(progress_bar):            
                        
            x, y = x.to(self.device), y.to(self.device)
            x, y_a, y_b, lam = self._apply_mixup(x, y)
            logits = self.model(x)            
            loss = lam * self.criterion(logits, y_a) + (1.0 - lam) * self.criterion(logits, y_b)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if step % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "LR": f"{current_lr:.6f}"})
        
        return running_loss / (step + 1)