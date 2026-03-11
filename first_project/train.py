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
    train_acc,
    test_loss,
    test_acc, 
):
    writer.add_scalar("Train/Loss", float(train_loss), epoch)
    writer.add_scalar("Train/Acc", float(train_acc), epoch)
    writer.add_scalar("Test/Loss", float(test_loss), epoch)
    writer.add_scalar("Test/Acc", float(test_acc), epoch)    


def _log_model_graph(writer, model, cfg, device):
    try:
        dummy = torch.randn(1, *eval(str(cfg['model']['input_dim'])), device=device)
        writer.add_graph(model, dummy)
    except Exception as e:
        print(f"[TB] add_graph 생략: {e}")
        

def main():
    cfg = load_config()
    set_seed(42)
    
    out_dir = Path(cfg['output_dir'])

    device = build_device(cfg)

    train_loader, test_loader, input_dim = build_dataset_loaders(cfg)

    model_name = cfg['model'].get('name', 'base')
    model = build_model(cfg, device, model_name, input_dim)     
    trainer = build_trainer(model, device, out_dir, cfg)
    evaluator = Evaluator(model=model, device=device)

    writer = build_writer(cfg)
    _log_model_graph(writer, model, cfg, device)

    for epoch in range(1, cfg['train']['epochs'] + 1):            
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()

        trainer.train_one_epoch(train_loader, epoch)
        
        torch.cuda.synchronize()
        end = time.time()

        avg_step_time = (end - start) / len(train_loader)
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        batch_size = cfg['data']['batch_size']
        throughput = batch_size / avg_step_time

        print(f"[Epoch {epoch:3d}] Time: {end - start:.2f}s | Avg Step Time: {avg_step_time:.4f}s | Peak Mem: {peak_mem_gb:.2f} GB | Throughput: {throughput:.2f} samples/s")
        
        train_loss, train_acc = evaluator.evaluate(train_loader)  
        test_loss, test_acc = evaluator.evaluate(test_loader)
         
        print(f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        print(f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}")

        _log_epoch_metrics(
            writer,
            epoch,
            train_loss,
            train_acc,
            test_loss,
            test_acc, 
        )
 
    # === TB: 종료 ===
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[GPU] 최대 메모리 사용량: {max_mem:.2f} MB")
