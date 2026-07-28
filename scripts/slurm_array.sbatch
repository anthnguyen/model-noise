#!/usr/bin/env bash
#SBATCH --job-name=attractors
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm_%A_%a.out
# Launch:
#   N=$(ls configs/sweep/*.yaml | wc -l)
#   sbatch --array=0-$((N-1))%16 scripts/slurm_array.sbatch configs/sweep
# %16 caps concurrent tasks; requeued/preempted tasks resume automatically
# because run.py picks up its checkpoint.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

DIR="${1:?usage: sbatch --array=0-N scripts/slurm_array.sbatch <config_dir>}"
CONFIGS=()
while IFS= read -r line; do CONFIGS+=("$line"); done < <(ls "$DIR"/*.yaml | sort)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "task $SLURM_ARRAY_TASK_ID -> $CONFIG on $(hostname)"
python run.py --config "$CONFIG" --gpu 0 --skip-completed
