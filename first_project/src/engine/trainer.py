from __future__ import annotations
from tqdm import tqdm
from pathlib import Path
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class Trainer:
    def __init__(self, model, device, lr: float, weight_decay: float, log_interval: int, out_dir: str, config=None, label_info=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.config = config or {}
        self.label_info = label_info or {}
        self.fine_to_coarse = self._build_fine_to_coarse_tensor(self.label_info.get("fine_to_coarse"))
        
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
        self.lambda_coarse = float(self.config.get("lambda_coarse", 1.0))
        self.coarse_warmup_epochs = int(self.config.get("coarse_warmup_epochs", 0))
        self.lambda_hier = float(self.config.get("lambda_hier", 0.0))
        self.hier_temperature = float(self.config.get("hier_temperature", 1.0))
        self.scheduler = self._build_scheduler()
        self.log_interval = log_interval
        
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = out_dir / 'checkpoints'
        self.model.set_output_dir(str(ckpt_dir))

    def _build_fine_to_coarse_tensor(self, fine_to_coarse):
        if fine_to_coarse is None:
            return None
        return torch.tensor([int(v) for v in fine_to_coarse], dtype=torch.long, device=self.device)

    def _to_coarse_targets(self, fine_targets):
        if self.fine_to_coarse is None:
            raise ValueError("fine_to_coarse mapping is required for coarse loss.")
        return self.fine_to_coarse[fine_targets]

    def _current_lambda_coarse(self, epoch: int) -> float:
        if self.coarse_warmup_epochs <= 0:
            return self.lambda_coarse
        warmup_ratio = min(1.0, float(max(epoch, 1)) / float(self.coarse_warmup_epochs))
        return self.lambda_coarse * warmup_ratio

    def _coarse_probs_from_fine_logits(self, fine_logits: torch.Tensor) -> torch.Tensor:
        if self.fine_to_coarse is None:
            raise ValueError("fine_to_coarse mapping is required for hierarchy consistency loss.")

        fine_probs = F.softmax(fine_logits, dim=1)
        num_coarse = int(self.fine_to_coarse.max().item()) + 1
        coarse_probs = torch.zeros(
            fine_probs.size(0),
            num_coarse,
            device=fine_probs.device,
            dtype=fine_probs.dtype,
        )
        coarse_index = self.fine_to_coarse.unsqueeze(0).expand(fine_probs.size(0), -1)
        coarse_probs.scatter_add_(1, coarse_index, fine_probs)
        return coarse_probs

    def _hierarchy_consistency_loss(self, fine_logits: torch.Tensor, coarse_logits: torch.Tensor):
        if self.fine_to_coarse is None:
            return None

        temperature = max(1e-6, self.hier_temperature)
        with torch.no_grad():
            coarse_target_probs = self._coarse_probs_from_fine_logits(fine_logits)

        coarse_log_probs = F.log_softmax(coarse_logits / temperature, dim=1)
        return F.kl_div(coarse_log_probs, coarse_target_probs, reduction="batchmean") * (temperature ** 2)

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, dict):
            fine_logits = outputs.get("fine_logits")
            coarse_logits = outputs.get("coarse_logits")
            if fine_logits is None:
                raise ValueError("Model output dict must include 'fine_logits'.")
            return fine_logits, coarse_logits
        return outputs, None
        
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
        running_fine_loss = 0.0
        running_coarse_loss = 0.0
        running_hier_loss = 0.0
        current_lambda_coarse = self._current_lambda_coarse(epoch)
        progress_bar = tqdm(loader, desc=f"train epoch {epoch}")
        for step, (x, y) in enumerate(progress_bar):            
                        
            x, y = x.to(self.device), y.to(self.device)
            x, y_a, y_b, lam = self._apply_mixup(x, y)
            outputs = self.model(x)
            fine_logits, coarse_logits = self._extract_logits(outputs)

            fine_loss = lam * self.criterion(fine_logits, y_a) + (1.0 - lam) * self.criterion(fine_logits, y_b)
            coarse_loss = None
            hier_loss = None
            if coarse_logits is not None and self.lambda_coarse > 0.0:
                coarse_y_a = self._to_coarse_targets(y_a)
                coarse_y_b = self._to_coarse_targets(y_b)
                coarse_loss = lam * self.criterion(coarse_logits, coarse_y_a) + (1.0 - lam) * self.criterion(coarse_logits, coarse_y_b)
                loss = fine_loss + current_lambda_coarse * coarse_loss
            else:
                loss = fine_loss

            if coarse_logits is not None and self.lambda_hier > 0.0:
                hier_loss = self._hierarchy_consistency_loss(fine_logits, coarse_logits)
                if hier_loss is not None:
                    loss = loss + self.lambda_hier * hier_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_fine_loss += fine_loss.item()
            if coarse_loss is not None:
                running_coarse_loss += coarse_loss.item()
            if hier_loss is not None:
                running_hier_loss += hier_loss.item()
            if step % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                postfix = {
                    "Loss": f"{loss.item():.5f}",
                    "Fine": f"{fine_loss.item():.5f}",
                    "LR": f"{current_lr:.6f}",
                    "LamC": f"{current_lambda_coarse:.3f}",
                }
                if coarse_loss is not None:
                    postfix["Coarse"] = f"{coarse_loss.item():.5f}"
                if hier_loss is not None:
                    postfix["Hier"] = f"{hier_loss.item():.5f}"
                progress_bar.set_postfix(postfix)
        
        return running_loss / (step + 1)