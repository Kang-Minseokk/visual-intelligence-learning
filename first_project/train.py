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


def _log_epoch_metrics(
    writer,
    epoch,
    train_loss,
    train_top1,
    train_top5,
    train_superclass,
    valid_loss,
    valid_top1,
    valid_top5,
    valid_superclass,
):
    writer.add_scalar("Train/Loss", float(train_loss), epoch)
    writer.add_scalar("Train/Top1", float(train_top1), epoch)
    writer.add_scalar("Train/Top5", float(train_top5), epoch)
    if train_superclass is not None:
        writer.add_scalar("Train/SuperclassMatchAt5", float(train_superclass), epoch)
    writer.add_scalar("Valid/Loss", float(valid_loss), epoch)
    writer.add_scalar("Valid/Top1", float(valid_top1), epoch)
    writer.add_scalar("Valid/Top5", float(valid_top5), epoch)
    if valid_superclass is not None:
        writer.add_scalar("Valid/SuperclassMatchAt5", float(valid_superclass), epoch)


def _log_model_graph(writer, model, cfg, device, input_dim):
    try:
        configured_input_dim = cfg['model'].get('input_dim', input_dim)
        dummy = torch.randn(1, *eval(str(configured_input_dim)), device=device)
        writer.add_graph(model, dummy)
    except Exception as e:
        print(f"[TB] add_graph 생략: {e}")
        

def main():
    cfg = load_config()
    set_seed(42)
    
    out_dir = Path(cfg['output_dir'])

    device = build_device(cfg)

    train_loader, valid_loader, test_loader, input_dim, classes, label_info = build_dataset_loaders(cfg)
    cfg['model']['input_dim'] = input_dim
    cfg['model']['num_classes'] = len(classes)

    model_name = cfg['model'].get('name', 'base')
    model = build_model(cfg, device, model_name, input_dim)     
    trainer = build_trainer(model, device, out_dir, cfg)
    evaluator = Evaluator(model=model, device=device)

    writer = build_writer(cfg)
    _log_model_graph(writer, model, cfg, device, input_dim)

    for epoch in range(1, cfg['train']['epochs'] + 1):            
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()

        trainer.train_one_epoch(train_loader, epoch)
        trainer.step_scheduler()
        
        torch.cuda.synchronize()
        end = time.time()

        avg_step_time = (end - start) / len(train_loader)
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        batch_size = cfg['data']['batch_size']
        throughput = batch_size / avg_step_time

        print(f"[Epoch {epoch:3d}] Time: {end - start:.2f}s | Avg Step Time: {avg_step_time:.4f}s | Peak Mem: {peak_mem_gb:.2f} GB | Throughput: {throughput:.2f} samples/s")
        
        train_metrics = evaluator.evaluate(
            train_loader,
            classes=classes,
            label_info=label_info,
            topk=5,
            log_sample_topk=True,
            split_name="train",
        )
        valid_metrics = evaluator.evaluate(
            valid_loader,
            classes=classes,
            label_info=label_info,
            topk=5,
            log_sample_topk=True,
            split_name="valid",
        )

        train_loss = train_metrics["loss"]
        train_top1 = train_metrics["top1"]
        train_top5 = train_metrics["top5"]
        train_super = train_metrics.get("superclass_match@5")
        valid_loss = valid_metrics["loss"]
        valid_top1 = valid_metrics["top1"]
        valid_top5 = valid_metrics["top5"]
        valid_super = valid_metrics.get("superclass_match@5")

        train_line = f"train_loss: {train_loss:.4f} | train_top1: {train_top1:.4f} | train_top5: {train_top5:.4f}"
        if train_super is not None:
            train_line += f" | train_superclass_match@5: {train_super:.4f}"
        valid_line = f"valid_loss: {valid_loss:.4f} | valid_top1: {valid_top1:.4f} | valid_top5: {valid_top5:.4f}"
        if valid_super is not None:
            valid_line += f" | valid_superclass_match@5: {valid_super:.4f}"
        print(train_line)
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
        )            
 
    # === TB: 종료 ===
    writer.flush()
    writer.close()
    
    test_metrics = evaluator.evaluate(
        test_loader,
        classes=classes,
        label_info=label_info,
        topk=5,
        log_sample_topk=True,
        split_name="test",
    )
    test_line = (
        f"test_loss: {test_metrics['loss']:.4f} | "
        f"test_top1: {test_metrics['top1']:.4f} | "
        f"test_top5: {test_metrics['top5']:.4f}"
    )
    if "superclass_match@5" in test_metrics:
        test_line += f" | test_superclass_match@5: {test_metrics['superclass_match@5']:.4f}"
    print(test_line)


if __name__ == '__main__':
    main()

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[GPU] 최대 메모리 사용량: {max_mem:.2f} MB")
