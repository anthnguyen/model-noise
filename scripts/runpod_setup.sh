#!/usr/bin/env bash
# One-shot setup for a fresh RunPod pod. Run once after the pod boots:
#
#   git clone <this repo> /workspace/attractors
#   cd /workspace/attractors && ./scripts/runpod_setup.sh
#
# Everything that must survive a pod stop/restart/terminate (repo, venv,
# HF model cache, run outputs) is placed under /workspace, which is the
# RunPod Network Volume mount point IF you attached one when creating the
# pod. Container disk (everywhere else, e.g. plain /root) is wiped on
# terminate and, on many templates, even on stop. If /workspace isn't a
# real mounted volume this script still works, it just warns you loudly.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

echo "[runpod_setup] repo at: $REPO_DIR"

if [ "$REPO_DIR" != /workspace/* ] && [ -d /workspace ]; then
    echo "[warn] repo is not under /workspace — if /workspace is your"
    echo "       network volume, your repo/venv/outputs will NOT survive"
    echo "       a pod terminate. Consider re-cloning into /workspace."
fi
if mountpoint -q /workspace 2>/dev/null; then
    echo "[ok] /workspace is a mounted volume (persists across restarts)"
else
    echo "[warn] /workspace is not a separate mount — if this pod has no"
    echo "       Network Volume attached, ALL local data is lost on"
    echo "       terminate (stopping may also wipe it, depending on"
    echo "       template). Attach a network volume before long sweeps,"
    echo "       or push outputs/ out (e.g. rsync/rclone) as you go."
fi

# --- HF cache on the persistent volume: model weights (several GB each)
# survive pod restarts instead of re-downloading every time. -------------
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"
grep -q HF_HOME ~/.bashrc 2>/dev/null || \
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo "[ok] HF_HOME=$HF_HOME (add HF_TOKEN via RunPod's pod env vars for"
echo "     gated models like gemma-2-9b-it, or: huggingface-cli login)"

# --- Python env. Most RunPod PyTorch templates already have a working
# torch+CUDA install system-wide; --system-site-packages reuses it instead
# of re-downloading multi-GB wheels. -------------------------------------
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    echo "[ok] system torch found — reusing via --system-site-packages"
    uv venv --system-site-packages
else
    echo "[info] no system torch — will install the full gpu stack"
    uv venv
fi
source .venv/bin/activate
uv pip install -q -r requirements.txt
python -c "import torch" 2>/dev/null || \
    uv pip install -q torch transformers accelerate bitsandbytes sentencepiece
python -c "import transformers, accelerate, bitsandbytes" 2>/dev/null || \
    uv pip install -q transformers accelerate bitsandbytes sentencepiece

echo "[runpod_setup] done. Next:"
echo "  python scripts/preflight.py --gpu --models"
echo "  ./scripts/launch_tmux.sh configs/mvp 0     # survives disconnects"
