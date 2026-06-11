"""
Fixed linear-probing evaluator for pre-extracted features.

Students may use any SSL method, backbone, feature dimension, or image
resolution. They are responsible for extracting features and labels:

    train_features: (N_train, D)
    train_labels:   (N_train,)
    test_features:  (N_test, D)
    test_labels:    (N_test,)

This script only fixes the linear probing recipe:
  - linear head: Linear(D, num_classes)
  - SGD, lr=0.1, momentum=0.9, weight_decay=0.0
  - cosine annealing, 100 epochs, batch size 128
  - final epoch evaluation; no validation/model selection
  - output: Top-1 Accuracy

The --seed argument is used only inside this linear probing script, for linear
head initialization and train-feature shuffling. It is not a pretraining seed.

Linear probing data:
  - STL10:   train on labeled train split (5k),  test on test split (8k)
  - CIFAR10: train on full train split (50k),    test on test split (10k)

DO NOT modify any hyperparameters. They are fixed for fair comparison.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#      Fixed hyperparameters (DO NOT change)     
EPOCHS = 100
BATCH_SIZE = 128


def feature_parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed linear probing from pre-extracted features."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for linear head initialization and feature shuffling only.",
    )

    # Single-dataset mode.
    parser.add_argument("--name", default="features")
    parser.add_argument("--train-features")
    parser.add_argument("--train-labels")
    parser.add_argument("--test-features")
    parser.add_argument("--test-labels")

    # Two-dataset convenience mode.
    parser.add_argument("--stl10-train-features")
    parser.add_argument("--stl10-train-labels")
    parser.add_argument("--stl10-test-features")
    parser.add_argument("--stl10-test-labels")
    parser.add_argument("--cifar10-train-features")
    parser.add_argument("--cifar10-train-labels")
    parser.add_argument("--cifar10-test-features")
    parser.add_argument("--cifar10-test-labels")
    return parser.parse_args()


def feature_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _first_npz_array(obj, path):
    keys = list(obj.keys())
    if not keys:
        raise RuntimeError(f"No arrays found in {path}")
    preferred = ("features", "labels", "arr_0", "x", "y")
    for key in preferred:
        if key in obj:
            return obj[key]
    return obj[keys[0]]


def load_array(path):
    if path is None:
        raise RuntimeError("Missing feature or label path.")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        arr = _first_npz_array(np.load(path), path)
    elif ext in (".pt", ".pth"):
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, dict):
            for key in ("features", "labels", "arr_0", "x", "y"):
                if key in arr:
                    arr = arr[key]
                    break
            else:
                raise RuntimeError(f"Could not find an array-like key in {path}")
    else:
        raise RuntimeError(f"Unsupported file extension for {path}")

    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu()
    return torch.from_numpy(np.asarray(arr))


def load_feature_set(feature_path, label_path):
    features = load_array(feature_path).float()
    labels = load_array(label_path).long().view(-1)
    if features.ndim != 2:
        raise RuntimeError(f"features must have shape (N, D), got {tuple(features.shape)}")
    if labels.ndim != 1:
        raise RuntimeError(f"labels must have shape (N,), got {tuple(labels.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"feature/label length mismatch: {features.shape[0]} vs {labels.shape[0]}"
        )
    return features, labels


def infer_num_classes(*labels):
    max_label = max(int(y.max().item()) for y in labels)
    min_label = min(int(y.min().item()) for y in labels)
    if min_label < 0:
        raise RuntimeError("Labels must be non-negative class indices.")
    return max_label + 1


def run_feature_linear_probe(
    name,
    train_features,
    train_labels,
    test_features,
    test_labels,
    device,
    epochs,
    batch_size,
    num_workers,
    seed,
):
    if train_features.shape[1] != test_features.shape[1]:
        raise RuntimeError(
            f"{name}: train/test feature dim mismatch "
            f"{train_features.shape[1]} vs {test_features.shape[1]}"
        )

    feature_dim = train_features.shape[1]
    num_classes = infer_num_classes(train_labels, test_labels)
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        scheduler.step()
        print(
            f"[{name}] epoch {epoch + 1:03d}/{epochs} "
            f"loss={total_loss / max(total, 1):.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = head(features).argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(
        f"[{name}] Top-1 Accuracy: {acc:.2f}% "
        f"(train={len(train_set)}, test={len(test_set)}, dim={feature_dim})"
    )
    return acc


def _maybe_add_dataset(args, prefix, name):
    paths = [
        getattr(args, f"{prefix}_train_features"),
        getattr(args, f"{prefix}_train_labels"),
        getattr(args, f"{prefix}_test_features"),
        getattr(args, f"{prefix}_test_labels"),
    ]
    if any(p is not None for p in paths):
        if not all(paths):
            raise RuntimeError(f"{name}: all four feature/label paths are required.")
        return [(name, *paths)]
    return []


def feature_main():
    args = feature_parse_args()
    feature_set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    jobs = []
    if any(
        p is not None
        for p in (args.train_features, args.train_labels, args.test_features, args.test_labels)
    ):
        if not all((args.train_features, args.train_labels, args.test_features, args.test_labels)):
            raise RuntimeError(
                "Single-dataset mode requires --train-features, --train-labels, "
                "--test-features, and --test-labels."
            )
        jobs.append(
            (
                args.name,
                args.train_features,
                args.train_labels,
                args.test_features,
                args.test_labels,
            )
        )

    jobs += _maybe_add_dataset(args, "stl10", "stl10")
    jobs += _maybe_add_dataset(args, "cifar10", "cifar10")
    if not jobs:
        raise RuntimeError("No feature files were provided. Run evaluate.py --help.")

    print("Fixed linear probing recipe:")
    print("  SGD lr=0.1 momentum=0.9 weight_decay=0.0")
    print(f"  cosine annealing, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    print("  final epoch evaluation, no validation/model selection")
    print(f"  device={device}")

    results = {}
    for name, train_x, train_y, test_x, test_y in jobs:
        print(f"\n== {name} ==")
        tr_x, tr_y = load_feature_set(train_x, train_y)
        te_x, te_y = load_feature_set(test_x, test_y)
        results[name] = run_feature_linear_probe(
            name,
            tr_x,
            tr_y,
            te_x,
            te_y,
            device,
            EPOCHS,
            BATCH_SIZE,
            args.num_workers,
            args.seed,
        )

    print("\nFinal Results")
    for name, acc in results.items():
        print(f"{name:10s} Top-1: {acc:.2f}%")


if __name__ == "__main__":
    feature_main()