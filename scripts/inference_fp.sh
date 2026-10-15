#!/usr/bin/env bash
# Full-precision (W16A16) inference of the one-step VSR backbone.
#
# Usage:
#   bash scripts/inference_fp.sh <dataset> [output_dir]
#
# Environment variables:
#   PYTHON          python interpreter (default: python)
#   GPU             CUDA_VISIBLE_DEVICES (optional)
#   PRETRAINED_DIR  WAN2.1 weights (default: pretrained/Wan2.1-T2V-1.3B)
#   CKPT_PATH       FP VSR weights (default: pretrained/lsgquant_fp.pth)
#   DATA_ROOT       test datasets (default: data/VSR_test)
#   UPSCALE         override the per-dataset upscaling factor
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained/Wan2.1-T2V-1.3B}"
CKPT_PATH="${CKPT_PATH:-pretrained/lsgquant_fp.pth}"
DATA_ROOT="${DATA_ROOT:-data/VSR_test}"

DATASET="${1:?usage: bash scripts/inference_fp.sh <dataset> [output_dir]}"
OUTPUT_PATH="${2:-results/fp/${DATASET}}"

if [[ -z "${UPSCALE:-}" ]]; then
  case "${DATASET}" in
    RealVSR|MVSR4x) UPSCALE=1 ;;
    *) UPSCALE=4 ;;
  esac
fi

MODEL_PATHS='[
    [
        "'"${PRETRAINED_DIR}"'/diffusion_pytorch_model.safetensors"
    ],
    "'"${PRETRAINED_DIR}"'/Wan2.1_VAE.pth"
]'

echo "FP inference | dataset=${DATASET} upscale=${UPSCALE} -> ${OUTPUT_PATH}"

"${PYTHON}" main_test.py \
  --prompt_cache prompt_cache \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --val_path "${DATA_ROOT}/${DATASET}" \
  --sr_step 799 \
  --upscale "${UPSCALE}"
