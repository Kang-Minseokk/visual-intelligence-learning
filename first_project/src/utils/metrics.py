import torch

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def multiclass_prf(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_per_class: bool = False,
    eps: float = 1e-12,
) -> dict:
    """Compute multiclass precision/recall/F1 for single-label classification."""
    if preds.ndim != 1 or targets.ndim != 1:
        raise ValueError("preds and targets must be 1D tensors.")
    if preds.numel() != targets.numel():
        raise ValueError("preds and targets must have the same length.")
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    preds = preds.to(torch.long)
    targets = targets.to(torch.long)

    valid_mask = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    if preds.numel() == 0:
        base = {
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_micro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
        }
        if include_per_class:
            base["per_class"] = []
        return base

    index = targets * num_classes + preds
    cm = torch.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes).to(torch.float32)

    tp = torch.diag(cm)
    predicted_positive = cm.sum(dim=0)
    actual_positive = cm.sum(dim=1)

    precision_per_class = tp / (predicted_positive + eps)
    recall_per_class = tp / (actual_positive + eps)
    f1_per_class = 2.0 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()

    total_tp = tp.sum()
    micro_precision = (total_tp / (predicted_positive.sum() + eps)).item()
    micro_recall = (total_tp / (actual_positive.sum() + eps)).item()
    micro_f1 = (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall + eps)

    support = actual_positive
    support_sum = support.sum().item()
    if support_sum <= 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        weighted_precision = (precision_per_class * support).sum().item() / support_sum
        weighted_recall = (recall_per_class * support).sum().item() / support_sum
        weighted_f1 = (f1_per_class * support).sum().item() / support_sum

    result = {
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "precision_micro": float(micro_precision),
        "recall_micro": float(micro_recall),
        "f1_micro": float(micro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
    }

    if include_per_class:
        result["per_class"] = [
            {
                "class_idx": int(i),
                "support": int(actual_positive[i].item()),
                "precision": float(precision_per_class[i].item()),
                "recall": float(recall_per_class[i].item()),
                "f1": float(f1_per_class[i].item()),
            }
            for i in range(num_classes)
        ]

    return result
