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


class _FineLogitsEvalModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, dict):
            if "fine_logits" not in outputs:
                raise ValueError("Model output dict must include 'fine_logits' for evaluation.")
            return outputs["fine_logits"]
        return outputs


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
):
    writer.add_scalar("Train/Loss", float(train_loss), epoch)
    writer.add_scalar("Train/Top1", float(train_top1), epoch)
    if train_top5 is not None:
        writer.add_scalar("Train/Top5", float(train_top5), epoch)
    if train_superclass is not None:
        writer.add_scalar("Train/SuperclassMatchAt5", float(train_superclass), epoch)
    if train_coarse_top1 is not None:
        writer.add_scalar("Train/CoarseTop1", float(train_coarse_top1), epoch)
    writer.add_scalar("Valid/Loss", float(valid_loss), epoch)
    writer.add_scalar("Valid/Top1", float(valid_top1), epoch)
    if valid_top5 is not None:
        writer.add_scalar("Valid/Top5", float(valid_top5), epoch)
    if valid_superclass is not None:
        writer.add_scalar("Valid/SuperclassMatchAt5", float(valid_superclass), epoch)
    if valid_coarse_top1 is not None:
        writer.add_scalar("Valid/CoarseTop1", float(valid_coarse_top1), epoch)


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
    cfg['model']['num_coarse_classes'] = int(cfg['model'].get('num_coarse_classes', len((label_info or {}).get('coarse_classes', [])) or 20))

    model_name = cfg['model'].get('name', 'base')
    model = build_model(cfg, device, model_name, input_dim)     
    trainer = build_trainer(model, device, out_dir, cfg, label_info=label_info)
    eval_model = _FineLogitsEvalModel(model)
    evaluator = Evaluator(model=eval_model, device=device)

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
        
        train_loss, train_top1, train_super = evaluator.evaluate(
            eval_model,
            train_loader,
            evaluator.criterion,
            device,
        )
        valid_loss, valid_top1, valid_super = evaluator.evaluate(
            eval_model,
            valid_loader,
            evaluator.criterion,
            device,
        )

        train_top5 = None
        train_coarse_top1 = None
        valid_top5 = None
        valid_coarse_top1 = None

        train_line = f"train_loss: {train_loss:.4f} | train_top1: {train_top1:.4f}"
        if train_super is not None:
            train_line += f" | train_superclass_match@5: {train_super:.4f}"
        if train_coarse_top1 is not None:
            train_line += f" | train_coarse_top1: {train_coarse_top1:.4f}"
        valid_line = f"valid_loss: {valid_loss:.4f} | valid_top1: {valid_top1:.4f}"
        if valid_super is not None:
            valid_line += f" | valid_superclass_match@5: {valid_super:.4f}"
        if valid_coarse_top1 is not None:
            valid_line += f" | valid_coarse_top1: {valid_coarse_top1:.4f}"
        print(train_line)
        print(valid_line)

        _log_epoch_metrics(
            writer,
            epoch,
            train_loss,
            train_top1,
            train_top5,
            train_super,
            train_coarse_top1,
            valid_loss,
            valid_top1,
            valid_top5,
            valid_super,
            valid_coarse_top1,
        )            
 
    # === TB: 종료 ===
    writer.flush()
    writer.close()
    
    test_loss, test_top1, test_super = evaluator.evaluate(
        eval_model,
        test_loader,
        evaluator.criterion,
        device,
    )
    test_line = (
        f"test_loss: {test_loss:.4f} | "
        f"test_top1: {test_top1:.4f}"
    )
    if test_super is not None:
        test_line += f" | test_superclass_match@5: {test_super:.4f}"
    print(test_line)


if __name__ == '__main__':
    main()

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[GPU] 최대 메모리 사용량: {max_mem:.2f} MB")
