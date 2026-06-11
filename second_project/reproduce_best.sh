#!/usr/bin/env bash
# reproduce_best.sh — one command to reproduce our best DINO result.
#
# Trains ViT-S/8 with DINO self-distillation + Sinkhorn-Knopp centering on
# STL-10 unlabeled (300 epochs), then runs the fixed evaluate.py linear probe
# on STL-10 and CIFAR-10. End-to-end on a single RTX 4090 takes ~14 hours.
#
# Expected final result (within ±1 %p due to GPU non-determinism):
#   STL-10  Top-1 ≈ 93.2 %
#   CIFAR-10 Top-1 ≈ 89.0 %
#
# Usage:
#   pip install -r requirements.txt
#   bash reproduce_best.sh

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="dino_vits8_sk_v2_fresh_stl_stats"

USE_STL10_STATS=1 \
EVAL_SIZE=96 \
BACKBONE=vit_small_patch8_224 \
  bash run_full_pipeline.sh "$RUN_NAME" \
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

echo ""
echo "=========================================================="
echo "FINAL Top-1 (output/$RUN_NAME/eval_results.txt):"
grep -A 3 "Final Results" "output/$RUN_NAME/eval_results.txt"
echo "=========================================================="
