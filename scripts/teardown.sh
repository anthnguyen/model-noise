#!/usr/bin/env bash
# Clean up after a sweep. Never deletes run data unless --purge-runs.
#   ./scripts/teardown.sh              # remove logs + __pycache__ only
#   ./scripts/teardown.sh --purge-runs # ALSO delete outputs/runs (asks first)
set -euo pipefail
cd "$(dirname "$0")/.."

find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf outputs/logs
echo "removed logs and __pycache__"

if [ "${1:-}" = "--purge-runs" ]; then
    n=$(ls -d outputs/runs/*/ 2>/dev/null | wc -l | tr -d ' ')
    read -r -p "Delete ALL $n run directories under outputs/runs? [y/N] " ans
    if [ "$ans" = "y" ]; then
        rm -rf outputs/runs
        echo "purged outputs/runs"
    else
        echo "kept outputs/runs"
    fi
fi
