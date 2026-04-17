from __future__ import annotations
from pathlib import Path
import time

import torch

from src.config import load_config
from src.engine.builders import (
    build_device,
    build_dataset_loaders,
    build_model,
    build_trainer,
    build_writer,
)
from src.engine.evaluator import Evaluator
from src.utils.seed import set_seed


def _compute_early_stop_score(valid_metrics, train_cfg):
    metric_name = str(train_cfg.get("early_stop_metric", "valid_top1")).lower()
    top1 = float(valid_metrics.get("top1", 0.0))
    superclass = valid_metrics.get("superclass_match@5")
    if superclass is not None:
        superclass = float(superclass)

    if metric_name in {"valid_top1", "top1"}:
        return top1, "valid_top1"

    if metric_name in {"valid_superclass_match@5", "superclass_match@5", "superclass"}:
        if superclass is None:
            return top1, "valid_top1(fallback)"
        return superclass, "valid_superclass_match@5"

    if metric_name in {"composite", "top1_superclass", "valid_top1+superclass"}:
        if superclass is None:
            return top1, "valid_top1(fallback)"
        w_top1 = float(train_cfg.get("early_stop_top1_weight", 0.5))
        w_super = float(train_cfg.get("early_stop_superclass_weight", 0.5))
        weight_sum = w_top1 + w_super
        if abs(weight_sum) < 1e-12:
            w_top1, w_super = 0.5, 0.5
            weight_sum = 1.0
        w_top1 /= weight_sum
        w_super /= weight_sum
        score = (w_top1 * top1) + (w_super * superclass)
        return score, f"composite(top1={w_top1:.2f},superclass={w_super:.2f})"

    return top1, "valid_top1(fallback)"


def _log_epoch_metrics(
    writer,
    epoch,
    train_loss,
    train_top1,
    train_top5,
    train_superclass,
    train_coarse_top1,
    valid_loss,
    valid_top1,
    valid_top5,
    valid_superclass,
    valid_coarse_top1,
    train_precision_macro=None,
    train_recall_macro=None,
    train_f1_macro=None,
    valid_precision_macro=None,
    valid_recall_macro=None,
    valid_f1_macro=None,
):
    writer.add_scalar('Train/Loss', float(train_loss), epoch)
    writer.add_scalar('Train/Top1', float(train_top1), epoch)
    writer.add_scalar('Train/Top5', float(train_top5), epoch)
    if train_superclass is not None:
        writer.add_scalar("Train/SuperclassMatchAt5", float(train_superclass), epoch)
    if train_coarse_top1 is not None:
        writer.add_scalar("Train/CoarseTop1", float(train_coarse_top1), epoch)
    if train_precision_macro is not None:
        writer.add_scalar("Train/PrecisionMacro", float(train_precision_macro), epoch)
    if train_recall_macro is not None:
        writer.add_scalar("Train/RecallMacro", float(train_recall_macro), epoch)
    if train_f1_macro is not None:
        writer.add_scalar("Train/F1Macro", float(train_f1_macro), epoch)
    writer.add_scalar("Valid/Loss", float(valid_loss), epoch)
    writer.add_scalar("Valid/Top1", float(valid_top1), epoch)
    writer.add_scalar("Valid/Top5", float(valid_top5), epoch)
    if valid_superclass is not None:
        writer.add_scalar("Valid/SuperclassMatchAt5", float(valid_superclass), epoch)
    if valid_coarse_top1 is not None:
        writer.add_scalar("Valid/CoarseTop1", float(valid_coarse_top1), epoch)
    if valid_precision_macro is not None:
        writer.add_scalar("Valid/PrecisionMacro", float(valid_precision_macro), epoch)
    if valid_recall_macro is not None:
        writer.add_scalar("Valid/RecallMacro", float(valid_recall_macro), epoch)
    if valid_f1_macro is not None:
        writer.add_scalar("Valid/F1Macro", float(valid_f1_macro), epoch)


def _log_model_graph(writer, model, cfg, device, input_dim):
    try:
        configured_input_dim = cfg['model'].get('input_dim', input_dim)
        dummy = torch.randn(1, *eval(str(configured_input_dim)), device=device)
        writer.add_graph(model, dummy)
    except Exception as e:
        print(f'[TB] add_graph skipped: {e}')


def main():
    cfg = load_config()
    set_seed(42)

    eval_cfg = cfg.get("eval", {})
    compute_prf = bool(eval_cfg.get("compute_prf", True))
    include_per_class_prf = bool(eval_cfg.get("include_per_class_prf", False))
    
    out_dir = Path(cfg['output_dir'])

    out_dir = Path(cfg['output_dir'])
    device = build_device(cfg)

    train_loader, valid_loader, test_loader, input_dim, classes, label_info = build_dataset_loaders(cfg)
    cfg['model']['input_dim'] = input_dim
    cfg['model']['num_classes'] = len(classes)

    model_name = cfg['model'].get('name', 'base')
    model = build_model(cfg, device, model_name, input_dim)
    trainer = build_trainer(model, device, out_dir, cfg)
    train_evaluator = Evaluator(model=model, device=device)

    writer = build_writer(cfg)
    _log_model_graph(writer, model, cfg, device, input_dim)

    early_stop_mode = str(cfg["train"].get("early_stop_mode", "max")).lower()
    if early_stop_mode not in {"max", "min"}:
        early_stop_mode = "max"

    if early_stop_mode == "max":
        best_valid_score = float("-inf")
    else:
        best_valid_score = float("inf")

    best_valid_top1 = float("-inf")
    best_epoch = 0
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
    early_stop_min_delta = float(cfg["train"].get("early_stop_min_delta", 0.0))
    stale_epochs = 0
    best_ckpt_path = out_dir / "best_valid_top1.pt"

    for epoch in range(1, cfg['train']['epochs'] + 1):            
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / 'best_valid_top1.pt'
    best_valid_top1 = -1.0
    best_epoch = 0
    train_loop_start = time.time()

    for epoch in range(1, cfg['train']['epochs'] + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        train_step_start = time.time()
        trainer.train_one_epoch(train_loader, epoch)
        
        torch.cuda.synchronize()
        end = time.time()

        avg_step_time = (train_step_end - train_step_start) / len(train_loader)
        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
        batch_size = cfg['data']['batch_size']
        throughput = batch_size / avg_step_time
        print(
            f'[Epoch {epoch:3d}] Train Time: {train_step_end - train_step_start:.2f}s | '
            f'Avg Step Time: {avg_step_time:.4f}s | Peak Mem: {peak_mem_gb:.2f} GB | '
            f'Throughput: {throughput:.2f} samples/s'
        )

        print(f"[Epoch {epoch:3d}] Time: {end - start:.2f}s | Avg Step Time: {avg_step_time:.4f}s | Peak Mem: {peak_mem_gb:.2f} GB | Throughput: {throughput:.2f} samples/s")

        evaluator.model = trainer.get_eval_model()
        
        train_metrics = evaluator.evaluate(
            train_loader,
            classes=classes,
            label_info=label_info,
            topk=5,
            log_sample_topk=True,
            split_name="train",
            compute_prf=compute_prf,
            include_per_class_prf=include_per_class_prf,
        )
        valid_metrics = evaluator.evaluate(
            valid_loader,
            classes=classes,
            label_info=label_info,
            topk=5,
            log_sample_topk=True,
            split_name="valid",
            compute_prf=compute_prf,
            include_per_class_prf=include_per_class_prf,
        )

        train_loss = train_metrics["loss"]
        train_top1 = train_metrics["top1"]
        train_top5 = train_metrics["top5"]
        train_super = train_metrics.get("superclass_match@5")
        train_coarse_top1 = train_metrics.get("coarse_top1")
        train_precision_macro = train_metrics.get("precision_macro")
        train_recall_macro = train_metrics.get("recall_macro")
        train_f1_macro = train_metrics.get("f1_macro")
        valid_loss = valid_metrics["loss"]
        valid_top1 = valid_metrics["top1"]
        valid_top5 = valid_metrics["top5"]
        valid_super = valid_metrics.get("superclass_match@5")
        valid_coarse_top1 = valid_metrics.get("coarse_top1")
        valid_precision_macro = valid_metrics.get("precision_macro")
        valid_recall_macro = valid_metrics.get("recall_macro")
        valid_f1_macro = valid_metrics.get("f1_macro")

        train_loss = train_metrics['loss']
        train_top1 = train_metrics['top1']
        train_top5 = train_metrics['top5']
        train_super = train_metrics.get('superclass_match@5')

        train_line = f'train_loss: {train_loss:.4f} | train_top1: {train_top1:.4f} | train_top5: {train_top5:.4f}'
        if train_super is not None:
            train_line += f" | train_superclass_match@5: {train_super:.4f}"
        if train_coarse_top1 is not None:
            train_line += f" | train_coarse_top1: {train_coarse_top1:.4f}"
        if train_precision_macro is not None and train_recall_macro is not None and train_f1_macro is not None:
            train_line += (
                f" | train_precision_macro: {train_precision_macro:.4f}"
                f" | train_recall_macro: {train_recall_macro:.4f}"
                f" | train_f1_macro: {train_f1_macro:.4f}"
            )
        valid_line = f"valid_loss: {valid_loss:.4f} | valid_top1: {valid_top1:.4f} | valid_top5: {valid_top5:.4f}"
        if valid_super is not None:
            valid_line += f" | valid_superclass_match@5: {valid_super:.4f}"
        if valid_coarse_top1 is not None:
            valid_line += f" | valid_coarse_top1: {valid_coarse_top1:.4f}"
        if valid_precision_macro is not None and valid_recall_macro is not None and valid_f1_macro is not None:
            valid_line += (
                f" | valid_precision_macro: {valid_precision_macro:.4f}"
                f" | valid_recall_macro: {valid_recall_macro:.4f}"
                f" | valid_f1_macro: {valid_f1_macro:.4f}"
            )
        print(train_line)

        valid_loss = None
        valid_top1 = None
        valid_top5 = None
        valid_super = None
        if valid_metrics is not None:
            valid_loss = valid_metrics['loss']
            valid_top1 = valid_metrics['top1']
            valid_top5 = valid_metrics['top5']
            valid_super = valid_metrics.get('superclass_match@5')

            valid_line = f'valid_loss: {valid_loss:.4f} | valid_top1: {valid_top1:.4f} | valid_top5: {valid_top5:.4f}'
            if valid_super is not None:
                valid_line += f' | valid_superclass_match@5: {valid_super:.4f}'
            print(valid_line)

        _log_epoch_metrics(
            writer,
            epoch,
            train_loss,
            train_top1,
            train_top5,
            train_super,
            valid_loss,
            valid_top1,
            valid_top5,
            valid_super,
            valid_coarse_top1,
            train_precision_macro,
            train_recall_macro,
            train_f1_macro,
            valid_precision_macro,
            valid_recall_macro,
            valid_f1_macro,
        )

        early_stop_score, early_stop_metric_label = _compute_early_stop_score(valid_metrics, cfg["train"])
        if early_stop_mode == "max":
            improved = early_stop_score > (best_valid_score + early_stop_min_delta)
        else:
            improved = early_stop_score < (best_valid_score - early_stop_min_delta)

        if improved:
            best_valid_score = float(early_stop_score)
            best_valid_top1 = float(valid_top1)
            best_epoch = int(epoch)
            stale_epochs = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "valid_score": best_valid_score,
                    "early_stop_metric": early_stop_metric_label,
                    "valid_top1": best_valid_top1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": trainer.get_ema_state_dict(),
                    "use_ema_for_eval": bool(trainer.use_ema and trainer.ema_use_for_eval),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                },
                best_ckpt_path,
            )
            print(
                f"[BEST] epoch={best_epoch} | {early_stop_metric_label}={best_valid_score:.4f} "
                f"| valid_top1={best_valid_top1:.4f} | saved={best_ckpt_path}"
            )
        else:
            stale_epochs += 1

        trainer.step_scheduler(early_stop_score)

        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            print(
                f"[EARLY STOP] no improvement for {stale_epochs} epochs "
                f"(patience={early_stop_patience}). Best epoch={best_epoch}, "
                f"{early_stop_metric_label}={best_valid_score:.4f}, valid_top1={best_valid_top1:.4f}"
            )
            break

    # === TB: 종료 ===
    writer.flush()
    writer.close()

    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        if bool(best_ckpt.get("use_ema_for_eval", False)) and best_ckpt.get("ema_state_dict") is not None:
            model.load_state_dict(best_ckpt["ema_state_dict"])
        else:
            model.load_state_dict(best_ckpt["model_state_dict"])
        evaluator.model = model
        print(
            f"[LOAD BEST] epoch={best_ckpt.get('epoch', best_epoch)} "
            f"| {best_ckpt.get('early_stop_metric', 'valid_score')}={best_ckpt.get('valid_score', best_valid_score):.4f} "
            f"| valid_top1={best_ckpt.get('valid_top1', best_valid_top1):.4f}"
        )
    
    test_metrics = evaluator.evaluate(
        test_loader,
        classes=classes,
        label_info=label_info,
        topk=5,
        log_sample_topk=True,
        split_name="test",
        compute_prf=compute_prf,
        include_per_class_prf=include_per_class_prf,
    )
    test_line = (
        f"test_loss: {test_metrics['loss']:.4f} | "
        f"test_top1: {test_metrics['top1']:.4f} | "
        f"test_top5: {test_metrics['top5']:.4f}"
    )
    if 'superclass_match@5' in test_metrics:
        test_line += f" | test_superclass_match@5: {test_metrics['superclass_match@5']:.4f}"
    if "coarse_top1" in test_metrics:
        test_line += f" | test_coarse_top1: {test_metrics['coarse_top1']:.4f}"
    if all(key in test_metrics for key in ["precision_macro", "recall_macro", "f1_macro"]):
        test_line += (
            f" | test_precision_macro: {test_metrics['precision_macro']:.4f}"
            f" | test_recall_macro: {test_metrics['recall_macro']:.4f}"
            f" | test_f1_macro: {test_metrics['f1_macro']:.4f}"
        )
    print(test_line)


if __name__ == '__main__':
    main()

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f'[GPU] max memory allocated: {max_mem:.2f} MB')
