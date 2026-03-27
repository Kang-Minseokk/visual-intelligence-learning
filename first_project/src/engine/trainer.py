from __future__ import annotations
import copy
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class Trainer:
    def __init__(self, model, device, lr: float, weight_decay: float, log_interval: int, out_dir: str, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}

        optimizer_name = str(self.config.get("optimizer", "sgd")).lower()
        if optimizer_name == "sgd":
            self.optimizer = SGD(
                model.parameters(),
                lr=float(lr),
                momentum=float(self.config.get("momentum", 0.9)),
                weight_decay=float(weight_decay),
                nesterov=bool(self.config.get("nesterov", True)),
            )
        elif optimizer_name == "adam":
            self.optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        label_smoothing = float(self.config.get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.mixup_alpha = float(self.config.get("mixup_alpha", 0.0))
        self.cutmix_alpha = float(self.config.get("cutmix_alpha", 0.0))
        self.cutmix_prob = float(self.config.get("cutmix_prob", 0.0))
        self.eval_with_ema = bool(self.config.get("eval_with_ema", True))
        self.ema_decay = float(self.config.get("ema_decay", 0.0))
        self.ema_model = None
        if self.ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)

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
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

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

    def get_eval_model(self):
        if self.eval_with_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        model_state = self.model.state_dict()
        ema_state = self.ema_model.state_dict()
        for key, value in ema_state.items():
            value.copy_(value * self.ema_decay + model_state[key] * (1.0 - self.ema_decay))

    def _apply_mixup(self, x, y):
        if self.mixup_alpha <= 0.0:
            return x, y, y, 1.0

        lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item())
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _apply_cutmix(self, x, y):
        if self.cutmix_alpha <= 0.0:
            return x, y, y, 1.0

        lam = float(torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item())
        index = torch.randperm(x.size(0), device=x.device)

        _, _, h, w = x.size()
        cut_ratio = torch.sqrt(torch.tensor(1.0 - lam, device=x.device)).item()
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        cx = torch.randint(0, w, (1,), device=x.device).item()
        cy = torch.randint(0, h, (1,), device=x.device).item()

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)

        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        box_area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - (box_area / float(w * h))
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_one_epoch(self, loader, epoch: int, track_flips: bool = False):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(loader, desc=f"train epoch {epoch}")
        for step, (x, y) in enumerate(progress_bar):            
                        
            x, y = x.to(self.device), y.to(self.device)

            if self.cutmix_alpha > 0.0 and torch.rand(1, device=x.device).item() < self.cutmix_prob:
                x, y_a, y_b, lam = self._apply_cutmix(x, y)
            else:
                x, y_a, y_b, lam = self._apply_mixup(x, y)

            logits = self.model(x)
            loss = lam * self.criterion(logits, y_a) + (1.0 - lam) * self.criterion(logits, y_b)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._update_ema()

            running_loss += loss.item()
            if step % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "LR": f"{current_lr:.6f}"})
        
        return running_loss / (step + 1)