#!/usr/bin/env bash
# Build and install the CUDA extension of the vendored RAFT (Resample2d), which
# the E*_warp metric needs. Run once per environment (see README "Measure").
#
# Usage:
#   bash scripts/setup_eval_raft.sh
#
# Environment variables:
#   PYTHON  python interpreter (default: python)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/eval_metric/RAFT/networks/resample2d_package"

PYTHON="${PYTHON:-python}"

echo "building the Resample2d CUDA extension with ${PYTHON}"
"${PYTHON}" setup.py build
"${PYTHON}" setup.py install
