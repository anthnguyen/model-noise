#!/usr/bin/env bash
# Run a single experiment, resumable, with logging.
#   ./scripts/run_experiment.sh configs/mvp/self_a.yaml [gpu_id]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?usage: run_experiment.sh <config.yaml> [gpu_id]}"
GPU="${2:-0}"
mkdir -p outputs/logs
LOG="outputs/logs/$(basename "${CONFIG%.yaml}")_$(date +%Y%m%d_%H%M%S).log"

echo "[run_experiment] config=$CONFIG gpu=$GPU log=$LOG"
python run.py --config "$CONFIG" --gpu "$GPU" --skip-completed 2>&1 | tee "$LOG"
