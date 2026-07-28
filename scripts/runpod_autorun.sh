#!/usr/bin/env bash
# Run a sweep to completion, analyze it, then stop or terminate the pod so an
# unattended overnight run doesn't keep billing after it finishes.
#
#   ./scripts/runpod_autorun.sh configs/sweep 0 terminate
#   ./scripts/runpod_autorun.sh configs/sweep 0 stop        # default
#   ./scripts/runpod_autorun.sh configs/sweep 0 none        # just run+analyze
#
# Launch it detached so it survives your SSH session closing:
#   tmux new -s auto -d './scripts/runpod_autorun.sh configs/sweep 0 terminate'
#
# stop vs terminate (from RunPod docs):
#   stop      - GPU released, /workspace retained, still billed for volume
#               storage. Reversible: `runpodctl pod start <id>`.
#   terminate - pod deleted, no further charges, ALL DATA DELETED unless it
#               lives on an attached Network Volume.
# Because terminate is destructive, this script refuses to terminate unless
# it can verify your outputs are on a mounted network volume. Override that
# check only if you have copied results off-pod yourself:
#   ALLOW_UNSAFE_TERMINATE=1 ./scripts/runpod_autorun.sh ... terminate
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

CONFIG_DIR="${1:?usage: runpod_autorun.sh <config_dir> [gpu_id] [stop|terminate|none]}"
GPU="${2:-0}"
ON_DONE="${3:-stop}"

case "$ON_DONE" in
    stop|terminate|none) ;;
    *) echo "[fatal] third arg must be stop|terminate|none (got '$ON_DONE')"
       exit 2 ;;
esac

if ! ls "$CONFIG_DIR"/*.yaml >/dev/null 2>&1; then
    echo "[fatal] no .yaml configs in '$CONFIG_DIR' - nothing to run."
    echo "        generate them first: python scripts/gen_configs.py --out $CONFIG_DIR ..."
    exit 2
fi

mkdir -p outputs/logs
AUTORUN_LOG="outputs/logs/autorun_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$AUTORUN_LOG") 2>&1
echo "[autorun] start $(date -u +%Y-%m-%dT%H:%M:%SZ) config_dir=$CONFIG_DIR gpu=$GPU on_done=$ON_DONE"

# --- persistence guard --------------------------------------------------- #
# Terminate deletes everything not on a network volume, so verify before
# arming it. Downgrade to stop rather than silently destroying results.
outputs_persistent() {
    # outputs/ must live on a mounted RunPod network volume. Only the known
    # volume mount points count: a generic "not on /" test gives false
    # positives on non-RunPod hosts, and a wrong "safe" verdict here is the
    # one mistake that destroys results.
    local out_path mnt
    mkdir -p outputs 2>/dev/null
    out_path="$(cd outputs 2>/dev/null && pwd -P)" || return 1
    case "$out_path" in
        /workspace|/workspace/*|/runpod-volume|/runpod-volume/*) ;;
        *) return 1 ;;
    esac
    mnt="$(df -P "$out_path" 2>/dev/null | awk 'NR==2 {print $6}')"
    case "$mnt" in
        /workspace|/runpod-volume) return 0 ;;
    esac
    # df may report the overlay for bind mounts; accept an explicit mountpoint
    mountpoint -q /workspace 2>/dev/null && return 0
    mountpoint -q /runpod-volume 2>/dev/null && return 0
    return 1
}

if [ "$ON_DONE" = "terminate" ]; then
    if outputs_persistent; then
        echo "[autorun] outputs/ is on a persistent mount, terminate is safe"
    elif [ "${ALLOW_UNSAFE_TERMINATE:-0}" = "1" ]; then
        echo "[autorun] WARNING: outputs/ is NOT on a network volume."
        echo "[autorun] ALLOW_UNSAFE_TERMINATE=1 set, terminating anyway."
        echo "[autorun] All results on this pod will be destroyed."
    else
        echo "[autorun] WARNING: outputs/ is NOT on a mounted network volume"
        echo "[autorun]   ($REPO_DIR/outputs). Terminating would delete every"
        echo "[autorun]   result this run produces. Downgrading to 'stop' so"
        echo "[autorun]   your data survives; start the pod later to collect it."
        echo "[autorun]   To terminate anyway: ALLOW_UNSAFE_TERMINATE=1"
        ON_DONE="stop"
    fi
fi

# --- the actual work ----------------------------------------------------- #
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

./scripts/batch_run.sh "$CONFIG_DIR" "$GPU"
BATCH_RC=$?
echo "[autorun] batch_run.sh exited $BATCH_RC"

if [ "$BATCH_RC" -eq 3 ]; then
    # exit 3 = SIGINT/SIGTERM. Someone (or the platform) interrupted us on
    # purpose; leave the pod up so the run can be resumed.
    echo "[autorun] batch was interrupted, leaving pod running so you can resume"
    exit 3
fi

echo "[autorun] running analysis"
python analyze.py || echo "[autorun] analyze.py failed, continuing to shutdown"
python scripts/status.py || true

# --- shut the pod down --------------------------------------------------- #
if [ "$ON_DONE" = "none" ]; then
    echo "[autorun] on_done=none, leaving pod running"
    exit "$BATCH_RC"
fi

POD_ID="${RUNPOD_POD_ID:-}"
if [ -z "$POD_ID" ]; then
    echo "[autorun] RUNPOD_POD_ID not set (not a RunPod pod?), cannot $ON_DONE"
    exit "$BATCH_RC"
fi

echo "[autorun] work finished, requesting pod $ON_DONE for $POD_ID in 60s"
echo "[autorun] cancel with Ctrl-C or: pkill -f runpod_autorun"
sleep 60

shutdown_via_cli() {
    command -v runpodctl >/dev/null 2>&1 || return 1
    # current syntax; fall back to the legacy form on older runpodctl builds
    if [ "$ON_DONE" = "terminate" ]; then
        runpodctl pod delete "$POD_ID" && return 0
        runpodctl remove pod "$POD_ID" && return 0
    else
        runpodctl pod stop "$POD_ID" && return 0
        runpodctl stop pod "$POD_ID" && return 0
    fi
    return 1
}

shutdown_via_api() {
    # fallback for pods where runpodctl hits TLS/cert issues; needs a real
    # API key exported as RUNPOD_API_KEY (pod env var)
    [ -n "${RUNPOD_API_KEY:-}" ] || return 1
    if [ "$ON_DONE" = "terminate" ]; then
        curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" && return 0
    else
        curl -fsS -X POST "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" && return 0
    fi
    return 1
}

if shutdown_via_cli || shutdown_via_api; then
    echo "[autorun] $ON_DONE requested successfully for pod $POD_ID"
else
    echo "[autorun] FAILED to $ON_DONE pod $POD_ID automatically."
    echo "[autorun] The pod is still billing. Stop it from the RunPod console,"
    echo "[autorun] or set RUNPOD_API_KEY as a pod env var for the API fallback."
    exit 1
fi

exit "$BATCH_RC"
