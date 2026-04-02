from __future__ import annotations

import os
import pickle
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def worker_init_fn(worker_id):
    """Keep worker RNG deterministic across dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_targets_from_dataset(ds):
    if hasattr(ds, "targets"):
        targets = ds.targets
        return targets if torch.is_tensor(targets) else torch.tensor(targets)
    if hasattr(ds, "labels"):
        labels = ds.labels
        return labels if torch.is_tensor(labels) else torch.tensor(labels)
    raise AttributeError("Dataset has no `targets`/`labels`. Provide a custom target getter.")


def _make_balanced_subset(base_dataset, k_per_class, num_classes):
    targets = _get_targets_from_dataset(base_dataset)
    indices = []
    counts = []
    for class_idx in range(num_classes):
        class_indices = (targets == class_idx).nonzero(as_tuple=True)[0][:k_per_class]
        indices.append(class_indices)
        counts.append(len(class_indices))

    all_indices = torch.cat(indices).tolist() if indices else []
    subset = Subset(base_dataset, all_indices)
    return subset, counts


def _load_cifar100_split_entry(dataset, key: str):
    split_list = dataset.train_list if dataset.train else dataset.test_list
    split_file_name = split_list[0][0]
    split_file_path = os.path.join(dataset.root, dataset.base_folder, split_file_name)
    with open(split_file_path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")

    if key in entry:
        return entry[key]

    key_bytes = key.encode("utf-8")
    if key_bytes in entry:
        return entry[key_bytes]

    raise KeyError(f"{key} not found in {split_file_path}")


def _load_cifar100_coarse_targets(dataset):
    return list(_load_cifar100_split_entry(dataset, "coarse_labels"))


def _load_cifar100_coarse_classes(dataset):
    meta_file_path = os.path.join(dataset.root, dataset.base_folder, dataset.meta["filename"])
    with open(meta_file_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")

    names = meta.get("coarse_label_names")
    if names is None:
        names = meta.get(b"coarse_label_names")
    if names is None:
        raise KeyError(f"coarse_label_names not found in {meta_file_path}")

    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]


def _build_fine_to_coarse_map(fine_targets, coarse_targets, num_fine: int = 100):
    mapping = [-1] * num_fine
    for fine_label, coarse_label in zip(fine_targets, coarse_targets):
        fine_idx = int(fine_label)
        coarse_idx = int(coarse_label)
        if mapping[fine_idx] == -1:
            mapping[fine_idx] = coarse_idx

    if any(idx < 0 for idx in mapping):
        raise ValueError("Failed to build complete fine_to_coarse mapping.")
    return mapping


def get_dataset_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool = True,
    seed: int = 42,
    k_train: int | None = 1000,
    num_classes: int = 100,
    dataset_name: str = "CIFAR100",
    label_level: str = "coarse",
    randaugment_enable: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    no_validation: bool = False,
):
    if dataset_name == "CIFAR100":
        train_ops = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if randaugment_enable:
            train_ops.append(
                transforms.RandAugment(
                    num_ops=int(randaugment_num_ops),
                    magnitude=int(randaugment_magnitude),
                )
            )
        train_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_transform = transforms.Compose(train_ops)
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_base = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download)
        valid_base = datasets.CIFAR100(root=root, train=True, transform=eval_transform, download=download)
        test_base = datasets.CIFAR100(root=root, train=False, transform=eval_transform, download=download)

        fine_classes = train_base.classes
        train_fine_targets = list(train_base.targets)
        train_coarse_targets = _load_cifar100_coarse_targets(train_base)
        coarse_classes = _load_cifar100_coarse_classes(train_base)
        fine_to_coarse = _build_fine_to_coarse_map(train_fine_targets, train_coarse_targets, num_fine=len(fine_classes))

        if label_level == "coarse":
            train_base.targets = train_coarse_targets
            valid_base.targets = _load_cifar100_coarse_targets(valid_base)
            test_base.targets = _load_cifar100_coarse_targets(test_base)
            classes = coarse_classes
        elif label_level == "fine":
            classes = fine_classes
        else:
            raise ValueError(f"Invalid label_level: {label_level}. Use 'coarse' or 'fine'.")

        indices = np.arange(len(train_base))
        targets = train_base.targets
        if no_validation:
            train_idx = indices
            valid_idx = np.array([], dtype=np.int64)
        else:
            train_idx, valid_idx = train_test_split(
                indices,
                test_size=0.1,
                random_state=42,
                stratify=targets,
            )
        input_dim = (3, 32, 32)
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")

    if k_train is None or k_train < 0:
        train_set = Subset(train_base, train_idx)
        valid_set = None if no_validation else Subset(valid_base, valid_idx)
        test_set = test_base
    else:
        effective_num_classes = len(classes)
        k_per_class_train = max(1, k_train // effective_num_classes)
        k_per_class_test = max(1, k_per_class_train // 10)

        train_set, train_counts = _make_balanced_subset(train_base, k_per_class_train, effective_num_classes)
        valid_set = None if no_validation else Subset(valid_base, valid_idx)
        test_set, test_counts = _make_balanced_subset(test_base, k_per_class_test, effective_num_classes)

        print(f"[{dataset_name}] TRAIN - total: {len(train_set)}, per-class: {dict(enumerate(train_counts))}")
        print(f"[{dataset_name}] TEST  - total: {len(test_set)}, per-class: {dict(enumerate(test_counts))}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_init_fn,
    )

    valid_loader = None
    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed),
            worker_init_fn=worker_init_fn,
        )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    label_info = {
        "label_level": label_level,
        "fine_classes": fine_classes if dataset_name == "CIFAR100" else classes,
        "coarse_classes": coarse_classes if dataset_name == "CIFAR100" else classes,
        "fine_to_coarse": fine_to_coarse if dataset_name == "CIFAR100" else None,
    }

    return train_loader, valid_loader, test_loader, input_dim, classes, label_info
