from __future__ import annotations
import copy
from tqdm import tqdm
from pathlib import Path
import torch
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
        self.coarse_to_fine = self._build_coarse_to_fine()
        self.superclass_smooth_alpha = float(self.config.get("superclass_smooth_alpha", 0.0))

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

    def _build_coarse_to_fine(self):
        """fine_to_coarse의 역방향 맵: coarse_idx → List[fine_idx]"""
        fine_to_coarse_list = self.label_info.get("fine_to_coarse")
        if fine_to_coarse_list is None:
            return None
        num_coarse = max(int(v) for v in fine_to_coarse_list) + 1
        coarse_to_fine = [[] for _ in range(num_coarse)]
        for fine_idx, coarse_idx in enumerate(fine_to_coarse_list):
            coarse_to_fine[int(coarse_idx)].append(fine_idx)
        return coarse_to_fine

    def _superclass_aware_ce(self, fine_logits, fine_targets, alpha: float):
        """
        같은 superclass의 sibling classes에만 label smoothing mass를 분배하는 CE loss.
          - correct class:  1 - alpha
          - each sibling:   alpha / n_siblings  (보통 alpha / 4)
          - non-sibling:    0

        CE loss gradient 효과:
          ∂loss/∂logit_sibling  = softmax(sibling) - alpha/n_siblings
            → sibling 확률이 alpha/n_siblings 미만이면 올림, 이상이면 내림
            → sibling들이 자연스럽게 top-2~5를 차지하도록 유도

        표준 CE와 비교:
          표준 CE: sibling과 non-sibling을 동일하게 내림
          이 loss: sibling은 적게 내리거나 올림, non-sibling은 강하게 내림
        """
        B, C = fine_logits.shape
        device = fine_logits.device

        coarse_targets = self.fine_to_coarse[fine_targets]              # [B]
        # [B, C]: True if fine class j belongs to the same superclass as sample i's target
        same_sc_mask = (self.fine_to_coarse.unsqueeze(0) == coarse_targets.unsqueeze(1))  # [B, C]
        correct_mask = F.one_hot(fine_targets, num_classes=C).bool()    # [B, C]
        sibling_mask = same_sc_mask & ~correct_mask                     # [B, C]

        n_siblings = sibling_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

        soft_targets = torch.zeros(B, C, device=device)
        soft_targets[correct_mask] = 1.0 - alpha
        soft_targets += sibling_mask.float() * (alpha / n_siblings)     # broadcast [B, C]

        log_probs = F.log_softmax(fine_logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

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

        if self.fine_to_coarse is not None:
            # Same-superclass-only mixup:
            # cross-superclass mixup은 top-5 ranking에 다른 superclass 클래스를
            # 높은 logit으로 올리도록 직접 학습시키기 때문에 superclass_match@5를 악화시킴.
            # 같은 superclass 내부끼리만 섞으면 augmentation 효과는 유지하면서
            # superclass 구조를 보호할 수 있음.
            coarse_y = self.fine_to_coarse[y]                            # [B]
            B = x.size(0)

            # [B, B]: True if samples i and j share the same superclass (excluding self)
            same_sc = (coarse_y.unsqueeze(0) == coarse_y.unsqueeze(1))  # [B, B]
            same_sc.fill_diagonal_(False)

            # 각 sample에 대해 같은 superclass 내에서 random pair 선택
            rand_vals = torch.rand(B, B, device=x.device)
            rand_vals[~same_sc] = -1.0                                   # 다른 superclass는 선택 불가
            index = rand_vals.argmax(dim=1)                              # [B]

            # 같은 superclass sample이 batch에 없는 경우 self-mix (사실상 no-mix)
            has_pair = same_sc.any(dim=1)
            index = torch.where(has_pair, index, torch.arange(B, device=x.device))
        else:
            index = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index]
        return mixed_x, y, y[index], lam

    def train_one_epoch(self, loader, epoch: int, track_flips: bool = False):
        self.model.train()
        running_loss = 0.0
        running_fine_loss = 0.0
        running_coarse_loss = 0.0
        progress_bar = tqdm(loader, desc=f"train epoch {epoch}")
        for step, (x, y) in enumerate(progress_bar):            
                        
            x, y = x.to(self.device), y.to(self.device)
            x, y_a, y_b, lam = self._apply_mixup(x, y)
            outputs = self.model(x)
            fine_logits, coarse_logits = self._extract_logits(outputs)

            if self.superclass_smooth_alpha > 0.0 and self.coarse_to_fine is not None:
                fine_loss = (lam * self._superclass_aware_ce(fine_logits, y_a, self.superclass_smooth_alpha)
                             + (1.0 - lam) * self._superclass_aware_ce(fine_logits, y_b, self.superclass_smooth_alpha))
            else:
                fine_loss = lam * self.criterion(fine_logits, y_a) + (1.0 - lam) * self.criterion(fine_logits, y_b)
            coarse_loss = None
            if coarse_logits is not None and self.lambda_coarse > 0.0:
                coarse_y_a = self._to_coarse_targets(y_a)
                coarse_y_b = self._to_coarse_targets(y_b)
                coarse_loss = lam * self.criterion(coarse_logits, coarse_y_a) + (1.0 - lam) * self.criterion(coarse_logits, coarse_y_b)
                loss = fine_loss + self.lambda_coarse * coarse_loss
            else:
                loss = fine_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._update_ema()

            running_loss += loss.item()
            running_fine_loss += fine_loss.item()
            if coarse_loss is not None:
                running_coarse_loss += coarse_loss.item()
            if step % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                postfix = {
                    "Loss": f"{loss.item():.5f}",
                    "Fine": f"{fine_loss.item():.5f}",
                    "LR": f"{current_lr:.6f}",
                }
                if coarse_loss is not None:
                    postfix["Coarse"] = f"{coarse_loss.item():.5f}"
                progress_bar.set_postfix(postfix)
        
        return running_loss / (step + 1)