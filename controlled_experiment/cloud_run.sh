#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${CHECKPOINT138:?Set CHECKPOINT138}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
mode="${1:?Use check, smoke or train}"
branch="${2:?Use A or B}"
case "$branch" in
  A) config='controlled_experiment/configs/A_beta1.json' ;;
  B) config='controlled_experiment/configs/B_beta1p25.json' ;;
  *) exit 2 ;;
esac
case "$mode" in check|smoke|train) ;; *) exit 2 ;; esac
shift 2
exec "$PYTHON_BIN" -u controlled_experiment/run.py "$mode" \
  --config "$config" --dataset-root "$DATASET_ROOT" \
  --parent-checkpoint "$CHECKPOINT138" --output "$OUTPUT_ROOT/${mode}_${branch}" \
  --device cuda "$@"
