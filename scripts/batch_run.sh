#!/usr/bin/env bash
# Run every config in a directory, sequentially, skipping completed runs.
# Safe to re-launch after a crash / preemption — it resumes where it stopped.
#   ./scripts/batch_run.sh configs/sweep [gpu_id]
# Parallel across N gpus (one shell per GPU):
#   ./scripts/batch_run.sh configs/sweep 0 4 &   # shard 0 of 4
#   ./scripts/batch_run.sh configs/sweep 1 4 &   # shard 1 of 4 ...
set -uo pipefail
cd "$(dirname "$0")/.."

DIR="${1:?usage: batch_run.sh <config_dir> [gpu_id] [n_shards]}"
GPU="${2:-0}"
SHARDS="${3:-1}"
mkdir -p outputs/logs

# portable array fill (macOS ships bash 3.2, which lacks mapfile)
CONFIGS=()
while IFS= read -r line; do CONFIGS+=("$line"); done < <(ls "$DIR"/*.yaml | sort)
TOTAL=${#CONFIGS[@]}
FAILED=0

for i in "${!CONFIGS[@]}"; do
    # shard by index so N parallel invocations partition the sweep
    if (( i % SHARDS != GPU % SHARDS )); then continue; fi
    cfg="${CONFIGS[$i]}"
    echo "=== [$((i+1))/$TOTAL] $cfg (gpu $GPU) ==="
    if ! python run.py --config "$cfg" --gpu "$GPU" --skip-completed \
            2>&1 | tee -a "outputs/logs/batch_gpu${GPU}.log"; then
        code=$?
        if [ "$code" -eq 3 ]; then
            echo "    interrupted — stopping batch (rerun to resume)"
            exit 3
        fi
        echo "    FAILED (exit $code) — continuing with next config"
        FAILED=$((FAILED+1))
    fi
done

echo "=== batch done: $FAILED failures out of shard of $TOTAL ==="
[ "$FAILED" -eq 0 ]
