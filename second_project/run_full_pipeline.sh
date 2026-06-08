#!/usr/bin/env bash
# End-to-end: train DINO -> extract features -> run evaluate.py.
#
# Usage:
#   ./run_full_pipeline.sh <run-name> [extra train args...]
#
# Example:
#   ./run_full_pipeline.sh vits16_sk_v1 --epochs 80 --batch-size 128
#
# All paths are relative to second_project/. The script:
#   1. Trains via train_dino_vit.py into output/<run-name>/
#   2. Extracts features (CLS) for STL10 and CIFAR10 train/test
#   3. Runs the fixed evaluate.py linear probe and tees the results.
#
# Logs:
#   nohup/<run-name>.log               (training output)
#   output/<run-name>/log.csv          (per-epoch metrics)
#   output/<run-name>/features/        (extracted .npy features)
#   output/<run-name>/eval_results.txt (final evaluate.py output)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run-name> [extra train args...]"
  exit 1
fi

RUN_NAME="$1"; shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="./output/${RUN_NAME}"
FEATURES_DIR="${OUTPUT_DIR}/features"
LOG_DIR="./nohup"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"
EVAL_LOG="${OUTPUT_DIR}/eval_results.txt"

mkdir -p "$OUTPUT_DIR" "$FEATURES_DIR" "$LOG_DIR"

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
BACKBONE="${BACKBONE:-vit_small_patch16_224}"
EVAL_SIZE="${EVAL_SIZE:-224}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
USE_STL10_STATS="${USE_STL10_STATS:-0}"  # set to 1 to use STL10 native stats
EXTRACT_STATS_FLAG=""
if [[ "$USE_STL10_STATS" == "1" ]]; then EXTRACT_STATS_FLAG="--use-stl10-stats"; fi

echo "[$(date '+%F %T')] Stage 1/3: Training ($BACKBONE on GPU $CUDA_DEVICE) -> $OUTPUT_DIR" | tee -a "$TRAIN_LOG"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python3 -u train_dino_vit.py \
  --data-root ./data \
  --output-dir "$OUTPUT_DIR" \
  --backbone "$BACKBONE" \
  --amp-dtype "$AMP_DTYPE" \
  "$@" 2>&1 | tee -a "$TRAIN_LOG"

# Pick the best checkpoint: prefer best_probe, fall back to final.
if [[ -f "${OUTPUT_DIR}/backbone_best_probe.pt" ]]; then
  WEIGHTS="${OUTPUT_DIR}/backbone_best_probe.pt"
else
  WEIGHTS="${OUTPUT_DIR}/backbone_final.pt"
fi
echo "[$(date '+%F %T')] Stage 2/3: Feature extraction from $WEIGHTS" | tee -a "$TRAIN_LOG"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python3 -u extract_features.py \
  --data-root ./data \
  --output-dir "$FEATURES_DIR" \
  --backbone "$BACKBONE" \
  --weights "$WEIGHTS" \
  --eval-size "$EVAL_SIZE" \
  --amp-dtype "$AMP_DTYPE" \
  $EXTRACT_STATS_FLAG 2>&1 | tee -a "$TRAIN_LOG"

echo "[$(date '+%F %T')] Stage 3/3: evaluate.py (DO NOT modify)" | tee -a "$TRAIN_LOG"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python3 -u evaluate.py \
  --stl10-train-features "${FEATURES_DIR}/stl10_train_features.npy" \
  --stl10-train-labels   "${FEATURES_DIR}/stl10_train_labels.npy" \
  --stl10-test-features  "${FEATURES_DIR}/stl10_test_features.npy" \
  --stl10-test-labels    "${FEATURES_DIR}/stl10_test_labels.npy" \
  --cifar10-train-features "${FEATURES_DIR}/cifar10_train_features.npy" \
  --cifar10-train-labels   "${FEATURES_DIR}/cifar10_train_labels.npy" \
  --cifar10-test-features  "${FEATURES_DIR}/cifar10_test_features.npy" \
  --cifar10-test-labels    "${FEATURES_DIR}/cifar10_test_labels.npy" \
  2>&1 | tee "$EVAL_LOG"

echo "[$(date '+%F %T')] DONE. Final results saved to $EVAL_LOG" | tee -a "$TRAIN_LOG"
