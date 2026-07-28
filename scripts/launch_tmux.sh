#!/usr/bin/env bash
# Launch a batch run inside tmux so it survives SSH disconnects / closing
# your laptop. Use this on a raw GPU box with no job scheduler; if the
# cluster has SLURM, prefer scripts/slurm_array.sbatch instead (a submitted
# sbatch job already runs independent of your session).
#
#   ./scripts/launch_tmux.sh configs/sweep 0            # session "attractors"
#   ./scripts/launch_tmux.sh configs/sweep 0 my-session # custom session name
#
# Reattach later, from anywhere:
#   tmux attach -t attractors
# Detach without stopping the job: Ctrl-b then d
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="${1:?usage: launch_tmux.sh <config_dir> [gpu_id] [session_name]}"
GPU="${2:-0}"
SESSION="${3:-attractors}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists — reattaching."
    echo "  (if it's not running a batch, start one manually inside)"
    exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
    "source .venv/bin/activate && ./scripts/batch_run.sh '$DIR' '$GPU'; \
     echo; echo '[batch finished — press any key to close]'; read -n 1"

echo "started tmux session '$SESSION' running batch_run.sh on $DIR (gpu $GPU)"
echo "  attach : tmux attach -t $SESSION"
echo "  detach : Ctrl-b then d (job keeps running)"
echo "  status : python scripts/status.py"
