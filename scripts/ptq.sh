#!/usr/bin/env bash
# Post-training quantization: build the quantization parameters of one config.
#
# Usage:
#   bash scripts/ptq.sh <config>
#   bash scripts/ptq.sh lsgquant_w4a4_rank_32
#
# Reads the calibration data from `calib/` (scripts/calibrate.sh) and writes
# <config>.pth into `quant_params/`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained/Wan2.1-T2V-1.3B}"
CKPT_PATH="${CKPT_PATH:-pretrained/lsgquant_fp.pth}"
CALIB_DIR="${CALIB_DIR:-calib}"
QUANT_DIR="${QUANT_DIR:-quant_params}"
SEED="${SEED:-42}"

CONFIG_NAME="${1:?usage: bash scripts/ptq.sh <config>}"
CONFIG_FILE="quant_configs/${CONFIG_NAME}.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

MODEL_PATHS='[
    [
        "'"${PRETRAINED_DIR}"'/diffusion_pytorch_model.safetensors"
    ],
    "'"${PRETRAINED_DIR}"'/Wan2.1_VAE.pth"
]'

echo "PTQ | config=${CONFIG_NAME} (calib data: ${CALIB_DIR}) -> ${QUANT_DIR}/${CONFIG_NAME}.pth"

"${PYTHON}" ptq.py \
  --prompt_cache prompt_cache \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --sr_step 799 \
  --log "${QUANT_DIR}" \
  --calib_dir "${CALIB_DIR}" \
  --quant_config "${CONFIG_FILE}" \
  --seed "${SEED}"
