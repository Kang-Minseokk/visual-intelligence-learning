from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = CrossEntropyLoss()

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, dict):
            fine_logits = outputs.get("fine_logits")
            coarse_logits = outputs.get("coarse_logits")
            if fine_logits is None:
                raise ValueError("Model output dict must include 'fine_logits'.")
            return fine_logits, coarse_logits
        return outputs, None

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        classes=None,
        label_info=None,
        topk: int = 5,
        log_sample_topk: bool = False,
        split_name: str = "eval",
        coarse_head_type: str = "separate",
    ):
        self.model.eval()
        total_loss = total_top1 = total_topk = total_superclass_match = total_coarse_top1 = 0.0
        n = 0
        has_superclass_metric = False
        has_coarse_top1 = False
        logged_sample = False

        fine_to_coarse = (label_info or {}).get("fine_to_coarse")
        fine_classes = (label_info or {}).get("fine_classes")
        label_level = (label_info or {}).get("label_level")
        fine_to_coarse_tensor = None
        if fine_to_coarse is not None:
            fine_to_coarse_tensor = torch.tensor(
                [int(v) for v in fine_to_coarse], device=self.device, dtype=torch.long
            )

        for x, y in tqdm(loader, desc="eval"):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            logits, coarse_logits = self._extract_logits(outputs)
            loss = self.criterion(logits, y)

            probs = logits.softmax(dim=1)
            k = min(int(topk), probs.shape[1])
            topk_labels = probs.topk(k=k, dim=1).indices

            top1_preds = logits.argmax(dim=1)
            top1_batch = (top1_preds == y).float().mean().item()
            in_topk = topk_labels.eq(y.unsqueeze(1)).any(dim=1).float().mean().item()

            # superclass_match@5
            superclass_match_batch = None
            if fine_to_coarse is not None and fine_classes is not None and logits.shape[1] == len(fine_classes):
                has_superclass_metric = True
                mapped_pred = torch.tensor(
                    [[int(fine_to_coarse[idx]) for idx in row.tolist()] for row in topk_labels],
                    device=y.device,
                    dtype=torch.long,
                )
                if label_level == "fine":
                    mapped_true = torch.tensor(
                        [int(fine_to_coarse[idx]) for idx in y.tolist()],
                        device=y.device, dtype=torch.long,
                    )
                else:
                    mapped_true = y
                superclass_match_batch = mapped_pred.eq(mapped_true.unsqueeze(1)).float().mean().item()

            # coarse_top1 — branched on coarse_head_type
            coarse_top1_batch = None
            if fine_to_coarse_tensor is not None:
                coarse_targets = fine_to_coarse_tensor[y] if label_level == "fine" else y
                if coarse_head_type == "aggregate":
                    # derive coarse prediction by aggregating fine softmax over each superclass
                    num_coarse = int(fine_to_coarse_tensor.max().item()) + 1
                    fine_probs = torch.softmax(logits, dim=1)                       # [B, 100]
                    onehot = F.one_hot(fine_to_coarse_tensor, num_classes=num_coarse).float()  # [100, 20]
                    coarse_probs = fine_probs @ onehot                              # [B, 20]
                    coarse_preds = coarse_probs.argmax(dim=1)
                    coarse_top1_batch = (coarse_preds == coarse_targets).float().mean().item()
                    has_coarse_top1 = True
                elif coarse_logits is not None:
                    coarse_preds = coarse_logits.argmax(dim=1)
                    coarse_top1_batch = (coarse_preds == coarse_targets).float().mean().item()
                    has_coarse_top1 = True

            if log_sample_topk and not logged_sample:
                names_src = fine_classes if (fine_classes is not None and logits.shape[1] == len(fine_classes)) else classes
                names = [names_src[i] for i in topk_labels[0].tolist()] if names_src else [str(i) for i in topk_labels[0].tolist()]
                pairs = [f"{n}: {p:.4f}" for n, p in zip(names, probs[0].topk(k).values.tolist())]
                msg = f"[{split_name}] top-{k} -> " + ", ".join(pairs)
                if superclass_match_batch is not None:
                    msg += f" | sc@{k}: {superclass_match_batch:.3f}"
                print(msg)
                logged_sample = True

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_top1 += top1_batch * bsz
            total_topk += in_topk * bsz
            if superclass_match_batch is not None:
                total_superclass_match += superclass_match_batch * bsz
            if coarse_top1_batch is not None:
                total_coarse_top1 += coarse_top1_batch * bsz
            n += bsz

        metrics = {
            "loss": total_loss / n,
            "top1": total_top1 / n,
            "top5": total_topk / n,
        }
        if has_superclass_metric:
            metrics["superclass_match@5"] = total_superclass_match / n
        if has_coarse_top1:
            metrics["coarse_top1"] = total_coarse_top1 / n
        return metrics
