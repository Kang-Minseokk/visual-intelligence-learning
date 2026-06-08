# Reproduction Guide — DINO ViT-S/8 on STL-10 (best result: STL10 93.22% / CIFAR10 89.07%)

Self-contained instructions for a third party to reproduce our best result with
`evaluate.py` (the TA-provided fixed linear-probe evaluator, **never modified**).

## 1. System requirements

| | Tested with | Acceptable range |
|---|---|---|
| OS | Ubuntu 22.04 (Linux 6.8) | Any Linux with CUDA |
| Python | 3.9.25 | 3.9–3.12 |
| CUDA | 12.4 | 11.8+ |
| GPU | RTX 4090 (24 GB) | Any Ampere+ NVIDIA GPU with ≥14 GB and bf16 support (RTX 30xx/40xx, A100, H100, A6000, etc.) |
| Disk | ~10 GB free | 5 GB data + checkpoints |
| Network | required (first run) | downloads STL-10 (~2.6 GB) + CIFAR-10 (~170 MB) |

> **bf16 caveat**: pre-Ampere GPUs (RTX 20xx and older) do not support bf16. On
> those, change `--amp-dtype bf16` to `--amp-dtype fp16` everywhere.

## 2. Setup

```bash
# 1. (Optional but recommended) create an isolated env
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 3. Reproducing the best result

The single command below trains from scratch, extracts features, and runs the
fixed `evaluate.py` end-to-end. Total time: ~14 hours on a single RTX 4090.

```bash
USE_STL10_STATS=1 EVAL_SIZE=96 BACKBONE=vit_small_patch8_224 \
  bash run_full_pipeline.sh dino_vits8_sk_v2_fresh_stl_stats \
    --epochs 300 \
    --warmup-epochs 10 \
    --warmup-teacher-temp 0.07 \
    --warmup-teacher-temp-epochs 5 \
    --teacher-temp 0.04 \
    --batch-size 128 \
    --num-workers 12 \
    --global-size 96 \
    --local-size 48 \
    --n-local 8 \
    --global-scale 0.32 1.0 \
    --local-scale 0.05 0.32 \
    --out-dim 8192 \
    --centering sinkhorn \
    --freeze-last-layer 3 \
    --lr 5e-4 \
    --probe-every 5 \
    --probe-epochs 10 \
    --probe-eval-size 96 \
    --amp-dtype bf16 \
    --use-stl10-stats
```

Output files:
- `output/dino_vits8_sk_v2_fresh_stl_stats/log.csv` — per-epoch metrics
- `output/dino_vits8_sk_v2_fresh_stl_stats/backbone_best_probe.pt` — final backbone
- `output/dino_vits8_sk_v2_fresh_stl_stats/eval_results.txt` — final `evaluate.py` Top-1

Expected final line:
```
Final Results
stl10      Top-1: ~93.0–93.5%
cifar10    Top-1: ~88.7–89.4%
```

## 4. If you already have the trained checkpoint (skip training)

Drop `backbone_best_probe.pt` into a directory and run only Stage 2+3:

```bash
# Stage 2: extract features (must use --use-stl10-stats to match training)
python3 extract_features.py \
  --data-root ./data \
  --output-dir ./output/<your-dir>/features \
  --backbone vit_small_patch8_224 \
  --weights <path-to-backbone_best_probe.pt> \
  --eval-size 96 \
  --amp-dtype bf16 \
  --batch-size 256 \
  --num-workers 8 \
  --use-stl10-stats

# Stage 3: official evaluate.py (DO NOT modify — sha256 should be unchanged)
F=./output/<your-dir>/features
python3 evaluate.py \
  --stl10-train-features   $F/stl10_train_features.npy \
  --stl10-train-labels     $F/stl10_train_labels.npy \
  --stl10-test-features    $F/stl10_test_features.npy \
  --stl10-test-labels      $F/stl10_test_labels.npy \
  --cifar10-train-features $F/cifar10_train_features.npy \
  --cifar10-train-labels   $F/cifar10_train_labels.npy \
  --cifar10-test-features  $F/cifar10_test_features.npy \
  --cifar10-test-labels    $F/cifar10_test_labels.npy
```

## 5. Sanity check — verify `evaluate.py` was not modified

```bash
sha256sum evaluate.py
# Expected: 3eed54f38f87d2597783725d94b70640eb6eed95c391b1303b84e6eabddff0d1
```

## 6. Reproducibility caveats (honest)

1. **GPU non-determinism**: torch CUDA kernels are not fully deterministic by
   default. Run-to-run variance on the same hardware is typically ±0.3 %p on
   STL-10 / CIFAR-10 linear-probe.
2. **bf16 numerics**: different GPU architectures may produce slightly different
   bf16 rounding. Numbers within ±0.5 %p across Ampere/Hopper variants are
   expected.
3. **STL-10 download**: from `cs.toronto.edu` — if blocked, manually place
   `stl10_binary.tar.gz` in `./data/` first (torchvision will skip the
   download).
4. **CIFAR-10 download**: from `cs.toronto.edu` — same as above
   (`cifar-10-python.tar.gz` in `./data/`).

## 7. Configuration that produced the reported numbers

Saved verbatim in `output/dino_vits8_sk_v2_fresh_stl_stats/args.json`:

| Arg | Value |
|---|---|
| backbone | vit_small_patch8_224 |
| epochs | 300 |
| batch_size | 128 |
| n_local | 8 |
| global_size / local_size | 96 / 48 |
| global_scale | (0.32, 1.0) |
| local_scale | (0.05, 0.32) |
| out_dim | 8192 |
| centering | sinkhorn (3 iter, eps=0.05) |
| use_stl10_stats | True |
| lr | 5e-4 (effective 2.5e-4 at bs=128) |
| weight_decay | 0.04 → 0.4 (cosine) |
| teacher_temp | 0.04 (after 5-ep warmup from 0.07) |
| student_temp | 0.1 |
| momentum_teacher | 0.996 → 0.9999 (cosine, clamp 0.9999) |
| freeze_last_layer | 3 epochs |
| clip_grad | 3.0 |
| amp_dtype | bf16 |

## 8. Files in this directory

| File | Purpose |
|---|---|
| `train_dino_vit.py` | DINO + Sinkhorn-Knopp pretraining (the **production** trainer) |
| `extract_features.py` | Frozen-backbone feature extraction → `.npy` |
| `evaluate.py` | **TA-provided official evaluator** (do not edit) |
| `run_full_pipeline.sh` | End-to-end orchestrator (train → extract → evaluate) |
| `queue_experiments.sh` | Sequential multi-experiment runner |
| `make_tensorboard.py` | Convert `log.csv` → TensorBoard event files |
| `train_dino_ibot.py` | DINO + iBOT experiment (kept for reference — **failed** in our setup) |
| `smoke_test.py` | Fast loss-curve sanity check |
| `requirements.txt` | Pinned dependency versions |
| `REPRODUCE.md` | This file |
| `FINAL_REPORT.md` (under `output/`) | All experiment results table |
