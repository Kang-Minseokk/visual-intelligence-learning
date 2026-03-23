from __future__ import annotations
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, loader, classes=None, label_info=None, topk: int = 5, log_sample_topk: bool = False, split_name: str = "eval"):
        self.model.eval()
        total_loss, total_top1, total_topk, total_superclass_match, n = 0.0, 0.0, 0.0, 0.0, 0
        has_superclass_metric = False
        logged_sample = False

        fine_to_coarse = (label_info or {}).get("fine_to_coarse")
        fine_classes = (label_info or {}).get("fine_classes")
        label_level = (label_info or {}).get("label_level")

        for x, y in tqdm(loader, desc='eval'):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            probs = logits.softmax(dim=1)
            k = min(int(topk), probs.shape[1])
            topk_labels = probs.topk(k=k, dim=1).indices

            top1_preds = logits.argmax(dim=1)
            top1_batch = (top1_preds == y).float().mean().item()

            in_topk = topk_labels.eq(y.unsqueeze(1)).any(dim=1).float().mean().item()

            superclass_match_batch = None
            if fine_to_coarse is not None and fine_classes is not None and logits.shape[1] == len(fine_classes):
                has_superclass_metric = True
                mapped_pred = torch.tensor(
                    [[int(fine_to_coarse[idx]) for idx in row.tolist()] for row in topk_labels],
                    device=y.device,
                    dtype=torch.long,
                )
                if label_level == "fine":
                    mapped_true = torch.tensor([int(fine_to_coarse[idx]) for idx in y.tolist()], device=y.device, dtype=torch.long)
                else:
                    mapped_true = y

                superclass_match_batch = mapped_pred.eq(mapped_true.unsqueeze(1)).float().mean().item()

            if log_sample_topk and (not logged_sample):
                sample_probs, sample_labels = probs[0].topk(k=k, dim=0)
                names_source = fine_classes if (fine_classes is not None and logits.shape[1] == len(fine_classes)) else classes
                if names_source is None:
                    names = [str(idx) for idx in sample_labels.tolist()]
                else:
                    names = [names_source[idx] for idx in sample_labels.tolist()]
                pairs = [f"{name}: {prob:.4f}" for name, prob in zip(names, sample_probs.tolist())]
                msg = f"[{split_name}] top-{k} probs -> " + ", ".join(pairs)
                if superclass_match_batch is not None:
                    msg += f" | superclass_match@{k}: {superclass_match_batch:.3f}"
                print(msg)
                logged_sample = True

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_top1 += top1_batch * bsz
            total_topk += in_topk * bsz
            if superclass_match_batch is not None:
                total_superclass_match += superclass_match_batch * bsz
            n += bsz

        metrics = {
            "loss": total_loss / n,
            "top1": total_top1 / n,
            "top5": total_topk / n,
        }
        if has_superclass_metric:
            metrics["superclass_match@5"] = total_superclass_match / n
        return metrics
