"""
Extract features for evaluate.py from a trained DINO backbone.

Outputs (as .npy files) inside --output-dir:
  stl10_train_features.npy / stl10_train_labels.npy
  stl10_test_features.npy  / stl10_test_labels.npy
  cifar10_train_features.npy / cifar10_train_labels.npy
  cifar10_test_features.npy  / cifar10_test_labels.npy

The backbone is the timm ViT chosen at training time. We use the CLS token
output (num_features-dim vector) as the linear-probing feature. For CIFAR-10
(32x32) we upsample to the same eval size as STL-10 since DINO's backbone was
trained at 224 — keeping resolution consistent gives the most faithful probe.
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm


def build_backbone(name, weights_path, device):
    model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
    state = torch.load(weights_path, map_location="cpu")
    # Accept either a raw state_dict or a dict that contains one.
    if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("blocks.") for k in state):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval().to(device)
    return model


def get_transform(eval_size, use_stl10_stats=False):
    if use_stl10_stats:
        nm = (0.4406, 0.4273, 0.3858); nstd = (0.2687, 0.2613, 0.2685)
    else:
        nm = (0.485, 0.456, 0.406); nstd = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(eval_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=nm, std=nstd),
    ])


@torch.no_grad()
def extract(model, loader, device, amp_dtype):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            f = model(x)
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model = build_backbone(args.backbone, args.weights, device)
    tf = get_transform(args.eval_size, use_stl10_stats=args.use_stl10_stats)
    print(f"[norm] {'STL10-native' if args.use_stl10_stats else 'ImageNet'} stats", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # STL-10: labeled train (5K) and test (8K).
    stl_tr = datasets.STL10(args.data_root, split="train", transform=tf, download=True)
    stl_te = datasets.STL10(args.data_root, split="test", transform=tf, download=True)
    # CIFAR-10: full train (50K) and test (10K).
    cif_tr = datasets.CIFAR10(args.data_root, train=True, transform=tf, download=True)
    cif_te = datasets.CIFAR10(args.data_root, train=False, transform=tf, download=True)

    sets = [
        ("stl10_train", stl_tr),
        ("stl10_test", stl_te),
        ("cifar10_train", cif_tr),
        ("cifar10_test", cif_te),
    ]
    for name, ds in sets:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        feats, labels = extract(model, loader, device, amp_dtype)
        f_path = os.path.join(args.output_dir, f"{name}_features.npy")
        l_path = os.path.join(args.output_dir, f"{name}_labels.npy")
        np.save(f_path, feats)
        np.save(l_path, labels)
        print(f"[saved] {name}: features={feats.shape} -> {f_path}")
        print(f"[saved] {name}: labels  ={labels.shape} -> {l_path}")

    # Manifest.
    with open(os.path.join(args.output_dir, "feature_manifest.json"), "w") as f:
        json.dump({
            "backbone": args.backbone,
            "weights": os.path.abspath(args.weights),
            "eval_size": args.eval_size,
            "amp_dtype": args.amp_dtype,
            "datasets": [name for name, _ in sets],
        }, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--backbone", type=str, default="vit_small_patch16_224")
    p.add_argument("--weights", type=str, required=True,
                   help="path to backbone state_dict (.pt)")
    p.add_argument("--eval-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--use-stl10-stats", action="store_true",
                   help="Use STL-10 native mean/std for normalization (must "
                        "match the stats used at pretraining time).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
