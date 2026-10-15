#!/usr/bin/env bash
# Collect the calibration statistics (see README "Calibration").
#
# Usage:
#   bash scripts/calibrate.sh <config> [calib_mode] [num_videos]
#   bash scripts/calibrate.sh lsgquant_w4a4_rank_32 var 50
#   bash scripts/calibrate.sh smoothquant_w4a4 act_max 50
#
# The output file name comes from `calib_data.save_path` of the config and is
# written into `calib/`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained/Wan2.1-T2V-1.3B}"
CKPT_PATH="${CKPT_PATH:-pretrained/lsgquant_fp.pth}"
CALIB_DATASET="${CALIB_DATASET:-data/VideoSR/HQ-VSR}"

CONFIG_NAME="${1:?usage: bash scripts/calibrate.sh <config> [calib_mode] [num_videos]}"
CALIB_MODE="${2:-var}"
NUM_VIDEOS="${3:-50}"
CONFIG_FILE="quant_configs/${CONFIG_NAME}.yaml"
LOG_DIR="${LOG_DIR:-calib}"

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

echo "calibration | config=${CONFIG_NAME} calib_mode=${CALIB_MODE} num_videos=${NUM_VIDEOS}"

"${PYTHON}" gen_calib_data.py \
  --calib_mode "${CALIB_MODE}" \
  --num_videos "${NUM_VIDEOS}" \
  --prompt_cache prompt_cache \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --degradation_config datasets/degradation.yaml \
  --dataset_base_path "${CALIB_DATASET}" \
  --num_frames 25 \
  --height 288 \
  --width 576 \
  --sr_step 799 \
  --upscale 4 \
  --batch_size 1 \
  --log "${LOG_DIR}" \
  --quant_config "${CONFIG_FILE}"
