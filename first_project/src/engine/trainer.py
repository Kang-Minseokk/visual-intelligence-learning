from __future__ import annotations
import copy
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau, StepLR

class Trainer:
    def __init__(self, model, device, lr: float, weight_decay: float, log_interval: int, out_dir: str, config=None):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        self.label_info = label_info or {}
        self.fine_to_coarse = self._build_fine_to_coarse_tensor(self.label_info.get("fine_to_coarse"))
        self.coarse_to_fine = self._build_coarse_to_fine()
        self.superclass_smooth_alpha = float(self.config.get("superclass_smooth_alpha", 0.0))
        self.grad_clip_norm = float(self.config.get("grad_clip_norm", 0.0))

        optimizer_name = str(self.config.get('optimizer', 'sgd')).lower()
        if optimizer_name == 'sgd':
            self.optimizer = SGD(
                model.parameters(),
                lr=float(lr),
                momentum=float(self.config.get('momentum', 0.9)),
                weight_decay=float(weight_decay),
                nesterov=bool(self.config.get('nesterov', True)),
            )
        elif optimizer_name == 'adam':
            self.optimizer = Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        elif optimizer_name == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        else :
            raise ValueError(f"Unsupported Optimizer! : {optimizer_name}")
        
        label_smoothing = float(self.config.get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.augmentation_mode = str(self.config.get("augmentation_mode", "mixup")).lower()
        self.augmentation_prob = float(self.config.get("augmentation_prob", 1.0))
        self.mixup_scope = str(self.config.get("mixup_scope", "same_superclass")).lower()
        self.mixup_alpha = float(self.config.get("mixup_alpha", 0.0))
        self.cutmix_alpha = float(self.config.get("cutmix_alpha", 0.0))
        self.cutmix_prob = float(self.config.get("cutmix_prob", 0.5))

        self.lambda_coarse = float(self.config.get("lambda_coarse", 1.0))

        self.use_ema = bool(self.config.get("ema_enable", False))
        self.ema_use_for_eval = bool(self.config.get("ema_use_for_eval", True))
        self.ema_decay = float(self.config.get("ema_decay", 0.999))
        self.ema_update_after_step = int(self.config.get("ema_update_after_step", 0))
        self.global_step = 0
        self.ema_model = None
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False

        self.scheduler = self._build_scheduler()
        self.log_interval = log_interval

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / 'checkpoints'
        self.model.set_output_dir(str(ckpt_dir))

    def _update_ema(self):
        if not self.use_ema or self.ema_model is None:
            return
        if self.global_step < self.ema_update_after_step:
            return
        with torch.no_grad():
            ema_state = self.ema_model.state_dict()
            model_state = self.model.state_dict()
            for k, ema_v in ema_state.items():
                model_v = model_state[k].detach()
                # Integer buffers (e.g., num_batches_tracked) cannot use EMA arithmetic.
                if not torch.is_floating_point(ema_v):
                    ema_v.copy_(model_v)
                    continue
                if ema_v.dtype != model_v.dtype:
                    model_v = model_v.to(dtype=ema_v.dtype)
                ema_v.mul_(self.ema_decay).add_(model_v, alpha=1.0 - self.ema_decay)

    def get_eval_model(self):
        if self.use_ema and self.ema_use_for_eval and self.ema_model is not None:
            return self.ema_model
        return self.model

    def get_ema_state_dict(self):
        if not self.use_ema or self.ema_model is None:
            return None
        return {k: v.detach().cpu().clone() for k, v in self.ema_model.state_dict().items()}

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
        scheduler_name = str(self.config.get('scheduler', 'cosine')).lower()
        if scheduler_name in {'none', 'off', ''}:
            return None
        
        total_epochs = int(self.config.get("epochs", 1))
        warmup_epochs = int(self.config.get("warmup_epochs", 0))
        min_lr = float(self.config.get("min_lr", 0.0))

        if scheduler_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.get("plateau_mode", "max"),
                factor=float(self.config.get("plateau_factor", 0.5)),
                patience=int(self.config.get("plateau_patience", 10)),
                threshold=float(self.config.get("plateau_threshold", 1e-4)),
                cooldown=int(self.config.get("plateau_cooldown", 0)),
                min_lr=min_lr,
            )

        if scheduler_name == "step":
            step_size = int(self.config.get("lr_step_size", self.config.get("step_size", 30)))
            gamma = float(self.config.get("lr_gamma", self.config.get("step_gamma", 0.1)))
            if warmup_epochs > 0:
                warmup = LinearLR(
                    self.optimizer,
                    start_factor=1.0 / float(max(1, warmup_epochs)),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                step = StepLR(self.optimizer, step_size=max(1, step_size), gamma=gamma)
                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, step],
                    milestones=[warmup_epochs],
                )
            return StepLR(self.optimizer, step_size=max(1, step_size), gamma=gamma)

        if scheduler_name != "cosine":
            raise ValueError(f"Unsupported scheduler! : {scheduler_name}")
        
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

    def step_scheduler(self, metrics=None):
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metrics is None:
                    raise ValueError("Metrics is required for ReduceLROnPlateau scheduler.")
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()

    def _sample_pair_indices(self, y):
        if self.mixup_scope != "same_superclass" or self.fine_to_coarse is None:
            return torch.randperm(y.size(0), device=y.device)

        coarse_y = self.fine_to_coarse[y]
        batch_size = y.size(0)
        same_sc = (coarse_y.unsqueeze(0) == coarse_y.unsqueeze(1))
        same_sc.fill_diagonal_(False)
        rand_vals = torch.rand(batch_size, batch_size, device=y.device)
        rand_vals[~same_sc] = -1.0
        index = rand_vals.argmax(dim=1)
        has_pair = same_sc.any(dim=1)
        return torch.where(has_pair, index, torch.arange(batch_size, device=y.device))

    @staticmethod
    def _rand_bbox(size, lam, device):
        _, _, h, w = size
        cut_ratio = torch.sqrt(torch.tensor(1.0 - lam, device=device))
        cut_w = int(w * float(cut_ratio))
        cut_h = int(h * float(cut_ratio))

        cx = torch.randint(0, w, (1,), device=device).item()
        cy = torch.randint(0, h, (1,), device=device).item()

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)
        return x1, y1, x2, y2

    def _apply_batch_augmentation(self, x, y):
        mode = self.augmentation_mode
        if mode in {"none", "off", ""}:
            return x, y, y, 1.0

        if self.augmentation_prob < 1.0:
            if float(torch.rand(1, device=x.device).item()) > self.augmentation_prob:
                return x, y, y, 1.0

        can_mixup = self.mixup_alpha > 0.0
        can_cutmix = self.cutmix_alpha > 0.0 and x.ndim == 4

        if mode == "mixup":
            if not can_mixup:
                return x, y, y, 1.0
            aug_type = "mixup"
        elif mode == "cutmix":
            if not can_cutmix:
                return x, y, y, 1.0
            aug_type = "cutmix"
        elif mode in {"mixup_cutmix", "mixcut", "cutmix_mixup"}:
            if can_mixup and can_cutmix:
                aug_type = "cutmix" if float(torch.rand(1, device=x.device).item()) < self.cutmix_prob else "mixup"
            elif can_cutmix:
                aug_type = "cutmix"
            elif can_mixup:
                aug_type = "mixup"
            else:
                return x, y, y, 1.0
        else:
            return x, y, y, 1.0

        alpha = self.cutmix_alpha if aug_type == "cutmix" else self.mixup_alpha
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
        index = self._sample_pair_indices(y)

        if aug_type == "mixup":
            mixed_x = lam * x + (1.0 - lam) * x[index]
            return mixed_x, y, y[index], lam

        x1, y1, x2, y2 = self._rand_bbox(x.size(), lam, x.device)
        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        cut_area = max(1, (x2 - x1) * (y2 - y1))
        full_area = max(1, x.size(-1) * x.size(-2))
        lam = 1.0 - (cut_area / full_area)
        return mixed_x, y, y[index], float(lam)

    def train_one_epoch(self, loader, epoch: int, track_flips: bool = False):
        self.model.train()
        running_loss = 0.0
        running_fine_loss = 0.0
        running_coarse_loss = 0.0
        progress_bar = tqdm(loader, desc=f"train epoch {epoch}")
        for step, (x, y) in enumerate(progress_bar):            
                        
            x, y = x.to(self.device), y.to(self.device)
            x, y_a, y_b, lam = self._apply_batch_augmentation(x, y)
            outputs = self.model(x)
            fine_logits, coarse_logits = self._extract_logits(outputs)

            if self.cutmix_alpha > 0.0 and torch.rand(1, device=x.device).item() < self.cutmix_prob:
                x, y_a, y_b, lam = self._apply_cutmix(x, y)
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
            if self.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            self.global_step += 1
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
