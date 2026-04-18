from __future__ import annotations
from pathlib import Path
import platform, subprocess, sys, time

import torch
import yaml

from src.config import load_config
from src.engine.builders import (
    build_dataset_loaders,
    build_device,
    build_model,
    build_trainer,
    build_writer,
)
from src.engine.evaluator import Evaluator
from src.utils.seed import set_seed


def _save_metadata(out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        rev = "no git"
    (out_dir / "git_info.txt").write_text(rev)
    (out_dir / "env_info.txt").write_text(
        f"python: {sys.version}\n"
        f"torch: {torch.__version__}\n"
        f"cuda: {torch.version.cuda}\n"
        f"platform: {platform.platform()}\n"
    )


def _log_to_tb(writer, epoch, prefix, metrics):
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", float(v), epoch)


def _metrics_str(prefix, m):
    return " | ".join(f"{prefix}_{k}: {v:.4f}" for k, v in m.items())


def _log_model_graph(writer, model, cfg, device, input_dim):
    try:
        dummy = torch.randn(1, *eval(str(cfg["model"].get("input_dim", input_dim))), device=device)
        writer.add_graph(model, dummy)
    except Exception as e:
        print(f"[TB] add_graph skipped: {e}")


def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_metadata(out_dir, cfg)

    device = build_device(cfg)
    train_cfg = cfg["train"]
    coarse_head_type = str(train_cfg.get("coarse_head_type", "separate"))
    eval_train = bool(train_cfg.get("eval_train_every_epoch", False))

    # test_loader is intentionally unused here — use scripts/run_official_eval.py
    train_loader, valid_loader, _, input_dim, classes, label_info = build_dataset_loaders(cfg)
    cfg["model"]["input_dim"] = input_dim
    cfg["model"]["num_classes"] = len(classes)
    cfg["model"]["num_coarse_classes"] = int(
        cfg["model"].get(
            "num_coarse_classes",
            len((label_info or {}).get("coarse_classes", [])) or 20,
        )
    )

    model = build_model(cfg, device, cfg["model"].get("name", "base"), input_dim)
    trainer = build_trainer(model, device, out_dir, cfg, label_info=label_info)

    evaluator = Evaluator(model=model, device=device)
    ema_evaluator = (
        Evaluator(model=trainer.ema_model, device=device) if trainer.ema_enable else None
    )
    ema_start = trainer.ema_eval_start_epoch

    writer = build_writer(cfg)
    _log_model_graph(writer, model, cfg, device, input_dim)

    eval_kw = dict(
        classes=classes,
        label_info=label_info,
        topk=5,
        coarse_head_type=coarse_head_type,
    )
    best_valid_top1 = 0.0
    best_valid_ema_top1 = 0.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()

        train_loss = trainer.train_one_epoch(train_loader, epoch)
        trainer.step_scheduler()

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        throughput = cfg["data"]["batch_size"] / (elapsed / len(train_loader))
        print(
            f"[Epoch {epoch:3d}] {elapsed:.1f}s | "
            f"step {elapsed/len(train_loader):.4f}s | "
            f"mem {peak_mem:.2f}GB | {throughput:.0f} samp/s | "
            f"train_loss {train_loss:.4f}"
        )

        # --- Train eval (off by default — halves GPU time) ---
        train_metrics = {}
        if eval_train:
            train_metrics = evaluator.evaluate(
                train_loader, split_name="train", log_sample_topk=True, **eval_kw
            )
            print(_metrics_str("train", train_metrics))
        _log_to_tb(writer, epoch, "Train", {"loss": train_loss, **train_metrics})

        # --- Valid eval (always) ---
        valid_metrics = evaluator.evaluate(
            valid_loader, split_name="valid", log_sample_topk=True, **eval_kw
        )
        print(_metrics_str("valid", valid_metrics))
        _log_to_tb(writer, epoch, "Valid", valid_metrics)

        # --- EMA valid eval (after warmup epoch) ---
        ema_metrics: dict = {}
        if ema_evaluator is not None and epoch >= ema_start:
            ema_metrics = ema_evaluator.evaluate(
                valid_loader, split_name="valid_ema", **eval_kw
            )
            print(_metrics_str("valid_ema", ema_metrics))
            _log_to_tb(writer, epoch, "Valid_EMA", ema_metrics)

        # --- Checkpoints ---
        ckpt_kw = dict(
            epoch=epoch,
            best_valid_top1=best_valid_top1,
            best_valid_ema_top1=best_valid_ema_top1,
            cfg=cfg,
        )
        trainer.save_checkpoint(out_dir / "last.pt", **ckpt_kw)

        v_top1 = valid_metrics.get("top1", 0.0)
        if v_top1 > best_valid_top1:
            best_valid_top1 = v_top1
            trainer.save_checkpoint(
                out_dir / "best.pt", **dict(ckpt_kw, best_valid_top1=best_valid_top1)
            )
            print(f"  → best.pt saved  (valid_top1={best_valid_top1:.4f})")

        if ema_metrics and epoch >= ema_start:
            e_top1 = ema_metrics.get("top1", 0.0)
            if e_top1 > best_valid_ema_top1:
                best_valid_ema_top1 = e_top1
                trainer.save_checkpoint(
                    out_dir / "best_ema.pt",
                    **dict(ckpt_kw, best_valid_ema_top1=best_valid_ema_top1),
                )
                print(f"  → best_ema.pt saved (valid_ema_top1={best_valid_ema_top1:.4f})")

    writer.flush()
    writer.close()

    print(f"\n=== Training complete ===")
    print(f"best valid_top1:     {best_valid_top1:.4f}  →  {out_dir}/best.pt")
    if ema_evaluator:
        print(f"best valid_ema_top1: {best_valid_ema_top1:.4f}  →  {out_dir}/best_ema.pt")
    print(f"\nRun official eval:")
    print(f"  python scripts/run_official_eval.py --ckpt {out_dir}/best.pt --config <cfg>")
    if ema_evaluator:
        print(f"  python scripts/run_official_eval.py --ckpt {out_dir}/best_ema.pt --config <cfg> --use-ema")


if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        print(f"[GPU] peak mem: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
