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
    valid_loss=None,
    valid_top1=None,
    valid_top5=None,
    valid_superclass=None,
):
    writer.add_scalar('Train/Loss', float(train_loss), epoch)
    writer.add_scalar('Train/Top1', float(train_top1), epoch)
    writer.add_scalar('Train/Top5', float(train_top5), epoch)
    if train_superclass is not None:
        writer.add_scalar('Train/SuperclassMatchAt5', float(train_superclass), epoch)

    if valid_loss is not None:
        writer.add_scalar('Valid/Loss', float(valid_loss), epoch)
    if valid_top1 is not None:
        writer.add_scalar('Valid/Top1', float(valid_top1), epoch)
    if valid_top5 is not None:
        writer.add_scalar('Valid/Top5', float(valid_top5), epoch)
    if valid_superclass is not None:
        writer.add_scalar('Valid/SuperclassMatchAt5', float(valid_superclass), epoch)


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

    save_best = bool(cfg['train'].get('save_best', True))
    test_use_best = bool(cfg['train'].get('test_use_best', True))
    if valid_loader is None:
        save_best = False
        test_use_best = False
        print('[Info] no_validation=True: validation metrics and best-checkpoint tracking are disabled.')

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
        trainer.step_scheduler()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_step_end = time.time()

        avg_step_time = (train_step_end - train_step_start) / len(train_loader)
        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
        batch_size = cfg['data']['batch_size']
        throughput = batch_size / avg_step_time
        print(
            f'[Epoch {epoch:3d}] Train Time: {train_step_end - train_step_start:.2f}s | '
            f'Avg Step Time: {avg_step_time:.4f}s | Peak Mem: {peak_mem_gb:.2f} GB | '
            f'Throughput: {throughput:.2f} samples/s'
        )

        eval_evaluator = Evaluator(model=trainer.get_eval_model(), device=device)
        train_metrics = train_evaluator.evaluate(
            train_loader,
            classes=classes,
            label_info=label_info,
            topk=5,
            log_sample_topk=True,
            split_name='train',
        )

        valid_metrics = None
        if valid_loader is not None:
            valid_metrics = eval_evaluator.evaluate(
                valid_loader,
                classes=classes,
                label_info=label_info,
                topk=5,
                log_sample_topk=True,
                split_name='valid',
            )

        train_loss = train_metrics['loss']
        train_top1 = train_metrics['top1']
        train_top5 = train_metrics['top5']
        train_super = train_metrics.get('superclass_match@5')

        train_line = f'train_loss: {train_loss:.4f} | train_top1: {train_top1:.4f} | train_top5: {train_top5:.4f}'
        if train_super is not None:
            train_line += f' | train_superclass_match@5: {train_super:.4f}'
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
        )

        if save_best and valid_top1 is not None and valid_top1 > best_valid_top1:
            best_valid_top1 = valid_top1
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'valid_metrics': valid_metrics,
                'config': cfg,
            }
            if trainer.ema_model is not None:
                checkpoint['ema_state_dict'] = trainer.ema_model.state_dict()
            torch.save(checkpoint, best_ckpt_path)
            print(f'[Checkpoint] best valid_top1 updated: {best_valid_top1:.4f} at epoch {best_epoch}')

        epoch_end = time.time()
        elapsed_since_start = epoch_end - train_loop_start
        print(
            f'[Epoch {epoch:3d}] Total Epoch Wall Time: {epoch_end - epoch_start:.2f}s | '
            f'Elapsed Training Time: {elapsed_since_start / 3600.0:.2f}h'
        )

    total_train_time = time.time() - train_loop_start
    print(f"[Training] Completed {cfg['train']['epochs']} epochs in {total_train_time / 3600.0:.2f}h ({total_train_time:.1f}s)")

    writer.flush()
    writer.close()

    if test_use_best and best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if trainer.ema_model is not None and 'ema_state_dict' in checkpoint:
            trainer.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"[Checkpoint] loaded best model from epoch {checkpoint['epoch']} for final test")

    final_evaluator = Evaluator(model=trainer.get_eval_model(), device=device)
    test_metrics = final_evaluator.evaluate(
        test_loader,
        classes=classes,
        label_info=label_info,
        topk=5,
        log_sample_topk=True,
        split_name='test',
    )
    test_line = (
        f"test_loss: {test_metrics['loss']:.4f} | "
        f"test_top1: {test_metrics['top1']:.4f} | "
        f"test_top5: {test_metrics['top5']:.4f}"
    )
    if 'superclass_match@5' in test_metrics:
        test_line += f" | test_superclass_match@5: {test_metrics['superclass_match@5']:.4f}"
    print(test_line)


if __name__ == '__main__':
    main()

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f'[GPU] max memory allocated: {max_mem:.2f} MB')
