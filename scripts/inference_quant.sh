#!/usr/bin/env bash
# Quantized inference with one of the released quantization configs.
#
# Usage:
#   bash scripts/inference_quant.sh <config> <dataset> [output_dir]
#   bash scripts/inference_quant.sh lsgquant_w4a4_rank_32 UDM10
#
# Environment variables: PYTHON / GPU / PRETRAINED_DIR / CKPT_PATH / DATA_ROOT /
#   QUANT_DIR (default: quant_params) / UPSCALE / NPROC (torchrun, optional)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained/Wan2.1-T2V-1.3B}"
CKPT_PATH="${CKPT_PATH:-pretrained/lsgquant_fp.pth}"
DATA_ROOT="${DATA_ROOT:-data/VSR_test}"
QUANT_DIR="${QUANT_DIR:-quant_params}"

CONFIG_NAME="${1:?usage: bash scripts/inference_quant.sh <config> <dataset> [output_dir]}"
DATASET="${2:?usage: bash scripts/inference_quant.sh <config> <dataset> [output_dir]}"
OUTPUT_PATH="${3:-results/${CONFIG_NAME}/${DATASET}}"
CONFIG_FILE="quant_configs/${CONFIG_NAME}.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "config not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -f "${QUANT_DIR}/${CONFIG_NAME}.pth" ]]; then
  echo "quantization parameters not found: ${QUANT_DIR}/${CONFIG_NAME}.pth" >&2
  echo "download the released weights (see README) or run scripts/ptq.sh ${CONFIG_NAME}" >&2
  exit 1
fi

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

echo "quantized inference | config=${CONFIG_NAME} dataset=${DATASET} upscale=${UPSCALE}"

"${PYTHON}" quant_main_test.py \
  --prompt_cache prompt_cache \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --val_path "${DATA_ROOT}/${DATASET}" \
  --sr_step 799 \
  --upscale "${UPSCALE}" \
  --log "${QUANT_DIR}" \
  --quant_config "${CONFIG_FILE}"
