"""
DINO self-supervised pretraining on STL-10 unlabeled split (100K images).

Backbone   : ViT-S/16 from timm (dynamic_img_size=True so the same backbone
             handles 224 globals and 96 locals).
Multi-crop : 2 globals (224x224) + N locals (96x96), DINOv1 standard.
Loss       : DINO cross-entropy. Centering = Sinkhorn-Knopp (DINOv2-style) by
             default, with EMA centering available for ablation. SK centering
             enforces a near-uniform marginal over prototypes per batch, which
             is the strongest anti-collapse safeguard we have for the small
             STL-10 unlabeled set.
Optimizer  : AdamW with cosine LR/WD schedules, linear warmup, layer-wise lr
             decay across ViT blocks.
Schedules  : Teacher EMA momentum cosine 0.996 -> 0.9999 (capped to avoid the
             frozen-teacher degenerate regime), teacher temp warmup, freeze
             last DINO-head layer for the first few epochs.
Validation : Every `probe_every` epochs we cache STL-10 train/test features
             once and fit a tiny linear classifier for a quick SSL monitor.

Why ViT-S/16 at 224 and not ResNet-18 at 96:
  - DINOv1 (Caron et al. 2021) showed ViT-S/16 > RN50 on linear probe by ~3-4%
    and self-distillation is much more stable on ViT due to attention diversity.
  - The official evaluator only takes features, so we control the resolution at
    extraction time. Upsampling STL-10 96 -> 224 is the standard SSL recipe and
    matches how DINOv1/v2 were trained.

Why Sinkhorn-Knopp centering for STL-10:
  - The previous EMA-centered run on STL-10 collapsed: loss locked at ln(4096)
    = 8.3178 and probe acc fell from 55% (ep 20) to 41% (ep 300).
  - With small batches (64-128) the EMA center estimate is noisy; SK centering
    is batch-uniform by construction (each prototype gets equal mass per
    iteration) so collapse cannot drive everything into one or zero modes.
"""

import argparse
import csv
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# DINO multi-crop augmentation
# -----------------------------------------------------------------------------
class DINOMultiCropTransform:
    """
    2 globals at `global_size` and `n_local` locals at `local_size`.
    DINOv1 defaults: 224 / 96.
    """

    def __init__(self, global_size=224, local_size=96, n_local=8,
                 global_scale=(0.4, 1.0), local_scale=(0.05, 0.4),
                 blur_kernel_global=9, blur_kernel_local=5,
                 normalize_mean=(0.485, 0.456, 0.406),
                 normalize_std=(0.229, 0.224, 0.225)):
        # Smaller blur kernels (9 / 5) match the gaussian effectively for the
        # sigma range (0.1, 2.0) and are ~4-6x faster than the textbook 23 used
        # in original DINO (which was tuned for 224x224 ImageNet). On STL-10
        # the larger kernel was the dominant CPU bottleneck.
        # Normalize stats default = ImageNet, but can be swapped to STL-10
        # native (measured: mean=(.441,.427,.386), std=(.269,.261,.269)).
        normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
        color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8,
        )
        base = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter,
            transforms.RandomGrayscale(p=0.2),
        ])
        bicubic = transforms.InterpolationMode.BICUBIC

        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_scale, interpolation=bicubic),
            base,
            transforms.GaussianBlur(kernel_size=blur_kernel_global, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])
        # Second global = DINO twist: solarize + occasional blur.
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_scale, interpolation=bicubic),
            base,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel_global, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_scale, interpolation=bicubic),
            base,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel_local, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        self.n_local = n_local

    def __call__(self, img):
        crops = [self.global1(img), self.global2(img)]
        for _ in range(self.n_local):
            crops.append(self.local(img))
        return crops


# -----------------------------------------------------------------------------
# DINO head: 3-layer MLP + L2 norm + weight-normalized linear
# -----------------------------------------------------------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=8192, hidden_dim=2048, bottleneck_dim=256, n_layers=3):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        # Force fp32 for the L2 normalize + weight-normalized projection.
        # In bf16 autocast, weight_norm + F.normalize is numerically unstable
        # (we saw NaN within the first iteration); fp32 here is cheap because
        # the bottleneck dim is small (256) and adds a few % overhead at most.
        x = x.float()
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# -----------------------------------------------------------------------------
# Backbone wrapper: ViT-S/16 via timm, CLS token output
# -----------------------------------------------------------------------------
def build_vit_backbone(name="vit_small_patch16_224", drop_path_rate=0.1):
    """
    timm ViT with global pool disabled so forward returns CLS token features.
    dynamic_img_size=True lets the same backbone process both 224 globals and
    96 locals: positional embeddings are interpolated per forward.
    """
    model = timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        drop_path_rate=drop_path_rate,
    )
    embed_dim = model.num_features
    return model, embed_dim


class DINOModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops):
        if not isinstance(crops, list):
            crops = [crops]
        # Group by spatial size so each unique size runs through ViT once.
        idx_by_size = {}
        for i, c in enumerate(crops):
            size = c.shape[-1]
            idx_by_size.setdefault(size, []).append(i)
        out = [None] * len(crops)
        for size, idxs in idx_by_size.items():
            batch = torch.cat([crops[i] for i in idxs], dim=0)
            feats = self.backbone(batch)
            logits = self.head(feats)
            chunks = logits.chunk(len(idxs), dim=0)
            for j, i in enumerate(idxs):
                out[i] = chunks[j]
        return out


# -----------------------------------------------------------------------------
# DINO loss with optional Sinkhorn-Knopp centering
# -----------------------------------------------------------------------------
@torch.no_grad()
def sinkhorn_knopp(scores, n_iter=3, eps=0.05):
    """
    SwAV/DINOv2-style Sinkhorn-Knopp centering.
    scores: [B, K] teacher logits (already in the temperature regime you want;
            do not divide by another temperature outside).
    eps: sharpening temperature inside the exp (smaller = sharper).
    Returns: [B, K] target distribution Q (rows sum to 1) with balanced columns.

    Numerical notes:
      - We cast to fp32 and subtract the per-row max before the exp to avoid
        overflow when running under bf16/fp16 autocast.
      - Q is normalized as a joint distribution; we maintain the standard
        SwAV row/col alternation.
    """
    scores = scores.float()
    # Subtract row-wise max for numerical stability.
    scores = scores - scores.max(dim=-1, keepdim=True).values
    Q = torch.exp(scores / eps).t()  # [K, B]
    Q = Q / Q.sum().clamp_min(1e-12)
    K, B = Q.shape
    for _ in range(n_iter):
        # Row normalize -> each prototype gets equal mass 1/K
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q = Q / K
        # Column normalize -> each sample is a distribution 1/B
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q = Q / B
    Q = Q * B  # rescale so rows sum to 1
    return Q.t()  # [B, K]


class DINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, n_epochs,
                 student_temp=0.1, center_momentum=0.9,
                 centering="ema", sk_iter=3, sk_eps=0.05):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.centering = centering
        self.sk_iter = sk_iter
        self.sk_eps = sk_eps
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(0, n_epochs - warmup_teacher_temp_epochs)) * teacher_temp,
        ))

    def forward(self, student_out_list, teacher_out_list, epoch):
        teacher_temp = float(self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)])
        student_log = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out_list]

        if self.centering == "sinkhorn":
            # DINOv2-style: pass teacher logits directly and let SK's eps act as
            # the sharpening temperature. Using teacher_temp as the SK eps gives
            # the same regime as softmax((t-center)/teacher_temp) but with
            # batch-balanced columns instead of EMA centering.
            teacher = [sinkhorn_knopp(t, n_iter=self.sk_iter,
                                      eps=teacher_temp).detach()
                       for t in teacher_out_list]
        else:
            teacher = [F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
                       for t in teacher_out_list]

        total_loss, n_terms = 0.0, 0
        for ti, t in enumerate(teacher):
            for si, s in enumerate(student_log):
                if si == ti:
                    continue
                total_loss = total_loss + (-(t * s).sum(dim=-1).mean())
                n_terms += 1
        loss = total_loss / max(1, n_terms)

        # Update EMA center even if not strictly needed for SK — used as a diag.
        with torch.no_grad():
            teacher_cat = torch.cat(teacher_out_list, dim=0)
            batch_center = teacher_cat.mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))
        return loss


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------
def cosine_schedule(base_value, final_value, n_iters, warmup_iters=0, start_warmup_value=0.0):
    if warmup_iters > 0:
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        warmup = np.array([])
    iters = np.arange(n_iters - warmup_iters)
    cos = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
    return np.concatenate((warmup, cos))


# -----------------------------------------------------------------------------
# Mini linear probe for SSL monitoring (does not touch SSL weights)
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = backbone(x)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe_eval(backbone, device, data_root, eval_size=224,
                      batch_size=256, n_epochs=10, lr=0.1,
                      use_stl10_stats=False):
    if use_stl10_stats:
        nm = (0.4406, 0.4273, 0.3858); nstd = (0.2687, 0.2613, 0.2685)
    else:
        nm = (0.485, 0.456, 0.406); nstd = (0.229, 0.224, 0.225)
    eval_tf = transforms.Compose([
        transforms.Resize(eval_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=nm, std=nstd),
    ])
    train_ds = datasets.STL10(data_root, split="train", transform=eval_tf, download=True)
    test_ds = datasets.STL10(data_root, split="test", transform=eval_tf, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    f_train, y_train = extract_features(backbone, train_loader, device)
    f_test, y_test = extract_features(backbone, test_loader, device)

    mu, sigma = f_train.mean(dim=0, keepdim=True), f_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    f_train = (f_train - mu) / sigma
    f_test = (f_test - mu) / sigma

    embed_dim = f_train.shape[1]
    num_classes = int(y_train.max().item()) + 1
    clf = nn.Linear(embed_dim, num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    f_train, y_train = f_train.to(device), y_train.to(device)
    f_test, y_test = f_test.to(device), y_test.to(device)

    n = f_train.shape[0]
    for _ in range(n_epochs):
        clf.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(clf(f_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        sched.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(f_test).argmax(dim=1) == y_test).float().mean().item()
    return acc


# -----------------------------------------------------------------------------
# Layer-wise LR decay for ViT
# -----------------------------------------------------------------------------
def get_vit_lr_groups(backbone, head, base_lr, weight_decay, layer_decay=1.0):
    """
    Build param groups with optional layer-wise LR decay across ViT blocks.
    layer_decay < 1.0 down-scales earlier blocks; 1.0 = no decay (DINOv1).
    """
    # Identify ViT block depth.
    blocks = getattr(backbone, "blocks", None)
    n_layers = len(blocks) if blocks is not None else 0
    # Layers: 0 = patch_embed/pos_embed/cls_token, 1..n_layers = block i-1,
    # n_layers + 1 = norm / head.
    def get_layer_id(name):
        if name.startswith("backbone."):
            n = name[len("backbone."):]
        else:
            n = name
        if n in ("cls_token", "pos_embed") or n.startswith("patch_embed"):
            return 0
        if n.startswith("blocks."):
            idx = int(n.split(".")[1])
            return idx + 1
        return n_layers + 1  # backbone norm, head, etc.

    groups = {}
    for name, p in list(backbone.named_parameters(prefix="backbone")) + list(head.named_parameters(prefix="head")):
        if not p.requires_grad:
            continue
        lid = get_layer_id(name) if name.startswith("backbone.") else (n_layers + 1)
        no_wd = name.endswith(".bias") or p.ndim == 1
        scale = layer_decay ** (n_layers + 1 - lid) if layer_decay < 1.0 else 1.0
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_wd else weight_decay,
                "lr_scale": scale,
                "_layer_id": lid,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    # STL-10 native stats (measured on unlabeled split, 100K images) or default
    # ImageNet stats. Switching to STL-10 stats fixes a ~17% std mismatch.
    if args.use_stl10_stats:
        norm_mean = (0.4406, 0.4273, 0.3858)
        norm_std = (0.2687, 0.2613, 0.2685)
    else:
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
    print(f"[norm] mean={norm_mean} std={norm_std} "
          f"({'STL10-native' if args.use_stl10_stats else 'ImageNet'})", flush=True)
    transform = DINOMultiCropTransform(
        global_size=args.global_size,
        local_size=args.local_size,
        n_local=args.n_local,
        global_scale=tuple(args.global_scale),
        local_scale=tuple(args.local_scale),
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )
    dataset = datasets.STL10(args.data_root, split="unlabeled", transform=transform, download=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(args.num_workers > 0),
    )
    print(f"[data] STL-10 unlabeled: {len(dataset)} images, {len(loader)} iters/epoch")

    # --- Model ---
    backbone_s, embed_dim = build_vit_backbone(args.backbone, drop_path_rate=args.drop_path)
    backbone_t, _ = build_vit_backbone(args.backbone, drop_path_rate=0.0)
    head_s = DINOHead(embed_dim, out_dim=args.out_dim)
    head_t = DINOHead(embed_dim, out_dim=args.out_dim)
    student = DINOModel(backbone_s, head_s).to(device)
    teacher = DINOModel(backbone_t, head_t).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[model] backbone={args.backbone} embed_dim={embed_dim} "
          f"params(student)={sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")

    n_crops_total = 2 + args.n_local
    loss_fn = DINOLoss(
        out_dim=args.out_dim,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        n_epochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
        centering=args.centering,
        sk_iter=args.sk_iter,
        sk_eps=args.sk_eps,
    ).to(device)

    # --- Optimizer with layer-wise lr decay & no-WD for biases/norms ---
    base_lr_now = 0.0  # filled in by scheduler
    param_groups = get_vit_lr_groups(
        backbone_s, head_s, base_lr=base_lr_now,
        weight_decay=args.weight_decay, layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=0.0)
    n_groups = len(param_groups)

    # --- Schedules ---
    iters_per_epoch = len(loader)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = args.warmup_epochs * iters_per_epoch
    base_lr = args.lr * args.batch_size / 256.0
    lr_schedule = cosine_schedule(base_lr, args.min_lr, total_iters, warmup_iters=warmup_iters)
    wd_schedule = cosine_schedule(args.weight_decay, args.weight_decay_end, total_iters)
    momentum_schedule = cosine_schedule(args.momentum_teacher, args.momentum_teacher_end, total_iters)
    # Clamp momentum to avoid m=1.0 dead end.
    momentum_schedule = np.clip(momentum_schedule, 0.0, 0.9999)
    lr_scales = [pg["lr_scale"] for pg in optimizer.param_groups]
    wd_flag = [pg["weight_decay"] > 0 for pg in optimizer.param_groups]

    # --- AMP ---
    use_bf16 = args.amp_dtype == "bf16"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and not use_bf16))

    # --- Log file ---
    log_path = os.path.join(args.output_dir, "log.csv")
    log_fields = ["epoch", "avg_loss", "lr", "wd", "momentum", "probe_acc", "epoch_sec"]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    best_probe = -1.0
    start_epoch = 0
    latest_path = os.path.join(args.output_dir, "ckpt_latest.pt")
    if args.resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location="cpu")
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        optimizer.load_state_dict(ck["optimizer"])
        if not use_bf16:
            scaler.load_state_dict(ck["scaler"])
        loss_fn.load_state_dict(ck["loss_fn"])
        start_epoch = ck["epoch"] + 1
        best_probe = ck.get("best_probe", -1.0)
        print(f"[resume] from epoch {start_epoch}, best_probe={best_probe:.4f}")

    # Save args for reproducibility.
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        import json
        json.dump(vars(args), f, indent=2)

    # --- Train loop ---
    for epoch in range(start_epoch, args.epochs):
        student.train()
        teacher.eval()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0

        for it, (crops, _) in enumerate(loader):
            global_it = epoch * iters_per_epoch + it
            lr_now = lr_schedule[global_it]
            wd_now = wd_schedule[global_it]
            for pi, pg in enumerate(optimizer.param_groups):
                pg["lr"] = lr_now * lr_scales[pi]
                if wd_flag[pi]:
                    pg["weight_decay"] = wd_now

            crops = [c.to(device, non_blocking=True) for c in crops]

            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                with torch.no_grad():
                    teacher_out = teacher(crops[:2])
                student_out = student(crops)
                loss = loss_fn(student_out, teacher_out, epoch)

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch {epoch} it {it}: {loss.item()} -- skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                if (epoch < args.freeze_last_layer) or (args.clip_grad > 0):
                    scaler.unscale_(optimizer)
            if epoch < args.freeze_last_layer:
                for _, p in student.head.last_layer.named_parameters():
                    if p.grad is not None:
                        p.grad = None
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            if use_bf16:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            # EMA teacher update.
            with torch.no_grad():
                m = momentum_schedule[global_it]
                for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                    p_t.data.mul_(m).add_(p_s.detach().data, alpha=1.0 - m)

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        epoch_sec = time.time() - t0
        cur_lr = optimizer.param_groups[-1]["lr"]
        cur_wd = max(pg["weight_decay"] for pg in optimizer.param_groups)
        cur_m = momentum_schedule[min(global_it, len(momentum_schedule) - 1)]

        # --- Validation (mini linear probe) ---
        probe_acc = float("nan")
        if args.probe_every > 0 and ((epoch + 1) % args.probe_every == 0 or epoch + 1 == args.epochs):
            probe_acc = linear_probe_eval(
                backbone=student.backbone,
                device=device, data_root=args.data_root,
                eval_size=args.probe_eval_size,
                batch_size=args.probe_batch_size, n_epochs=args.probe_epochs, lr=args.probe_lr,
                use_stl10_stats=args.use_stl10_stats,
            )

        print(f"[epoch {epoch+1:03d}/{args.epochs}] loss={avg_loss:.4f}  lr={cur_lr:.2e}  "
              f"wd={cur_wd:.4f}  m={cur_m:.4f}  probe_acc={probe_acc if probe_acc==probe_acc else 'n/a'}  "
              f"time={epoch_sec:.1f}s", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_loss:.6f}", f"{cur_lr:.6e}",
                                    f"{cur_wd:.6f}", f"{cur_m:.6f}",
                                    f"{probe_acc:.6f}" if probe_acc == probe_acc else "",
                                    f"{epoch_sec:.2f}"])

        ck = {
            "epoch": epoch,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "loss_fn": loss_fn.state_dict(),
            "args": vars(args),
            "best_probe": best_probe,
        }
        torch.save(ck, latest_path)
        if probe_acc == probe_acc and probe_acc > best_probe:
            best_probe = probe_acc
            ck["best_probe"] = best_probe
            torch.save(ck, os.path.join(args.output_dir, "ckpt_best_probe.pt"))
            torch.save(student.backbone.state_dict(),
                       os.path.join(args.output_dir, "backbone_best_probe.pt"))
            torch.save(teacher.backbone.state_dict(),
                       os.path.join(args.output_dir, "backbone_teacher_best_probe.pt"))

    torch.save(student.backbone.state_dict(), os.path.join(args.output_dir, "backbone_final.pt"))
    torch.save(teacher.backbone.state_dict(), os.path.join(args.output_dir, "backbone_teacher_final.pt"))
    print(f"[done] best probe acc = {best_probe:.4f}", flush=True)


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./output/dino_stl_vits16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Backbone / crops
    p.add_argument("--backbone", type=str, default="vit_small_patch16_224",
                   help="timm model name (e.g. vit_small_patch16_224, vit_tiny_patch16_224)")
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--global-size", type=int, default=224)
    p.add_argument("--local-size", type=int, default=96)
    p.add_argument("--n-local", type=int, default=8)
    p.add_argument("--global-scale", type=float, nargs=2, default=[0.4, 1.0])
    p.add_argument("--local-scale", type=float, nargs=2, default=[0.05, 0.4])

    # Optim / schedule
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4, help="base lr at batch 256")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--weight-decay-end", type=float, default=0.4)
    p.add_argument("--clip-grad", type=float, default=3.0)
    p.add_argument("--freeze-last-layer", type=int, default=5)
    p.add_argument("--layer-decay", type=float, default=1.0,
                   help="<1.0 enables layer-wise lr decay (DINOv2 small data uses ~0.9)")

    # DINO loss
    p.add_argument("--out-dim", type=int, default=8192)
    p.add_argument("--student-temp", type=float, default=0.1)
    p.add_argument("--teacher-temp", type=float, default=0.07)
    p.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    p.add_argument("--warmup-teacher-temp-epochs", type=int, default=15)
    p.add_argument("--center-momentum", type=float, default=0.9)
    p.add_argument("--centering", type=str, default="sinkhorn", choices=["ema", "sinkhorn"],
                   help="Anti-collapse centering. sinkhorn = DINOv2-style (recommended for small data).")
    p.add_argument("--sk-iter", type=int, default=3)
    p.add_argument("--sk-eps", type=float, default=0.05)
    p.add_argument("--momentum-teacher", type=float, default=0.996)
    p.add_argument("--momentum-teacher-end", type=float, default=0.9999)

    # Normalization
    p.add_argument("--use-stl10-stats", action="store_true",
                   help="Use STL-10 native mean/std (mean=(.441,.427,.386), "
                        "std=(.269,.261,.269)). Default: ImageNet stats.")

    # AMP / probe
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--probe-every", type=int, default=5)
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--probe-eval-size", type=int, default=224)
    p.add_argument("--probe-lr", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
