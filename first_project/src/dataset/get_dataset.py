from __future__ import annotations
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
import numpy as np
import random

def worker_init_fn(worker_id):
    """각 worker process의 random seed를 고정합니다."""
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

def get_dataset_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool = True,
    seed: int = 42,
    k_train: int | None = 1000,
    num_classes: int = 100,
    dataset_name: str = "CIFAR100",
):  
    batch_size = batch_size        
        
    if dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_base = datasets.CIFAR100(root=root, train=True, transform=transform, download=download)
        test_base  = datasets.CIFAR100(root=root, train=False, transform=transform, download=download)
        input_dim = "3*32*32"            
        
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")

    if k_train is None or k_train < 0:
        train_set, test_set = train_base, test_base
    else:
        k_per_class_train = max(1, k_train // num_classes)
        k_per_class_test = max(1, k_per_class_train // 10)

        train_set, train_counts = _make_balanced_subset(train_base, k_per_class_train, num_classes)
        test_set, test_counts = _make_balanced_subset(test_base, k_per_class_test, num_classes)

        print(f"[{dataset_name}] TRAIN - total: {len(train_set)}, per-class: {dict(enumerate(train_counts))}")
        print(f"[{dataset_name}] TEST  - total: {len(test_set)}, per-class: {dict(enumerate(test_counts))}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(seed),
                              worker_init_fn=worker_init_fn)

    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    
    return train_loader, test_loader, input_dim
