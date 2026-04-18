#!/usr/bin/env python
"""Official CIFAR-100 evaluation script.

Loads a checkpoint, wraps the model for single-tensor output,
then calls src.eval.official_eval.evaluate().

Usage:
  python scripts/run_official_eval.py --ckpt logs/run/best.pt --config configs/cfg.yaml
  python scripts/run_official_eval.py --ckpt logs/run/best_ema.pt --config configs/cfg.yaml --use-ema
  python scripts/run_official_eval.py --ckpt logs/run/best.pt --config configs/cfg.yaml --split val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# resolve imports from project root regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.builders import build_model
from src.eval.official_eval import evaluate as official_evaluate
from src.models.net.wrappers import FineLogitOnlyWrapper


# ---------------------------------------------------------------------------
# Loader helpers — eval_transform only, no augmentation
# ---------------------------------------------------------------------------

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# Conditions verified:
#   1. Normalize: CIFAR-100 standard mean/std — same as training
#   2. No augmentation: RandomCrop / RandomHorizontalFlip / RandAugment all absent
#   3. Mixup / CutMix: not applied (inference-only transform)
_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
])


def _build_test_loader(cfg: dict) -> DataLoader:
    """CIFAR-100 test set. shuffle=False."""
    ds = datasets.CIFAR100(
        root=cfg["data"]["root"], train=False,
        transform=_EVAL_TRANSFORM, download=True,
    )
    return DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,          # condition 3: no shuffle
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )


def _build_val_loader(cfg: dict) -> DataLoader:
    """Recreates the same 10 % validation split used during training."""
    base = datasets.CIFAR100(
        root=cfg["data"]["root"], train=True,
        transform=_EVAL_TRANSFORM, download=True,
    )
    indices = np.arange(len(base))
    _, valid_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=base.targets
    )
    return DataLoader(
        Subset(base, valid_idx),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Official CIFAR-100 evaluation")
    parser.add_argument("--ckpt",    required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--config",  required=True,  help="Config yaml used for training")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_state instead of model_state")
    parser.add_argument("--split",   choices=["test", "val"], default="test",
                        help="Dataset split to evaluate (default: test)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device(
        cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    # Build model (CIFAR-100 input_dim is fixed)
    input_dim = (3, 32, 32)
    cfg["model"]["input_dim"] = input_dim
    cfg["model"]["num_classes"] = int(cfg["model"]["num_classes"])
    cfg["model"]["num_coarse_classes"] = int(cfg["model"].get("num_coarse_classes", 20))

    model = build_model(cfg, device, cfg["model"].get("name", "base"), input_dim)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    if args.use_ema:
        if not ckpt.get("ema_state"):
            raise ValueError("--use-ema specified but checkpoint has no ema_state")
        model.load_state_dict(ckpt["ema_state"])
        weight_label = "EMA weights"
    else:
        model.load_state_dict(ckpt["model_state"])
        weight_label = "model weights"
    print(f"Loaded {weight_label} from {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # Wrap: official evaluate() expects model(x) → Tensor
    wrapped = FineLogitOnlyWrapper(model)

    # Build loader (no augmentation, shuffle=False)
    loader = _build_test_loader(cfg) if args.split == "test" else _build_val_loader(cfg)
    print(f"Split: {args.split}  |  samples: {len(loader.dataset)}")

    criterion = torch.nn.CrossEntropyLoss()
    loss, top1_acc, super_class_acc = official_evaluate(wrapped, loader, criterion, device)

    result_lines = [
        f"split:           {args.split}",
        f"checkpoint:      {ckpt_path}",
        f"use_ema:         {args.use_ema}",
        f"epoch:           {ckpt.get('epoch', 'unknown')}",
        f"loss:            {loss:.4f}",
        f"top1_acc:        {top1_acc:.4f}",
        f"super_class_acc: {super_class_acc:.4f}",
    ]
    print("\n=== Official Evaluation Results ===")
    print("\n".join(result_lines))

    out_txt = ckpt_path.with_name(f"{ckpt_path.stem}_official_{args.split}.txt")
    out_txt.write_text("\n".join(result_lines))
    print(f"\nSaved to {out_txt}")


if __name__ == "__main__":
    main()
