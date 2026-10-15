#!/usr/bin/env bash
# Evaluate one inference output directory.
#
# Usage:
#   bash scripts/evaluate.sh <pred_dir> [gt_dir]
#   bash scripts/evaluate.sh results/lsgquant_w4a4_rank_32/UDM10 data/VSR_test/UDM10/GT
#
# Without `gt_dir` only the no-reference metrics are computed. The DOVER and
# EWARP weights have to be downloaded beforehand (see README).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"

PRED_DIR="${1:?usage: bash scripts/evaluate.sh <pred_dir> [gt_dir]}"
GT_DIR="${2:-}"

if [[ -n "${GT_DIR}" ]]; then
  echo "evaluate ${PRED_DIR} against ${GT_DIR}"
  "${PYTHON}" eval_metrics.py \
    --gt "${GT_DIR}" \
    --pred "${PRED_DIR}" \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,maniqa
else
  echo "evaluate ${PRED_DIR} (no-reference metrics only)"
  "${PYTHON}" eval_metrics.py \
    --pred "${PRED_DIR}" \
    --metrics clipiqa,musiq,maniqa
fi

"${PYTHON}" eval_dover.py --pred "${PRED_DIR}"
"${PYTHON}" eval_ewarp.py --pred "${PRED_DIR}"
