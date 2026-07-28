#!/usr/bin/env python
"""Sweep progress at a glance (run from anywhere over SSH).

    python scripts/status.py
    watch -n 30 python scripts/status.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list non-completed runs")
    args = ap.parse_args()

    runs = sorted(Path(args.runs_dir).glob("*/manifest.json"))
    if not runs:
        print(f"no runs under {args.runs_dir}")
        return
    counts: Counter = Counter()
    details = []
    for m in runs:
        try:
            with open(m) as f:
                status = json.load(f).get("status", "unknown")
        except (json.JSONDecodeError, OSError):
            status = "corrupt"
        counts[status] += 1
        if status != "completed":
            ck = m.parent / "checkpoint.json"
            turn = "?"
            if ck.exists():
                try:
                    with open(ck) as f:
                        turn = json.load(f).get("next_turn", "?")
                except (json.JSONDecodeError, OSError):
                    pass
            details.append(f"  {status:12s} round {turn:>4}  {m.parent.name}")

    total = sum(counts.values())
    done = counts.get("completed", 0)
    bar = "#" * int(30 * done / total) + "-" * (30 - int(30 * done / total))
    print(f"[{bar}] {done}/{total} completed")
    for status, n in counts.most_common():
        print(f"  {status:12s} {n}")
    if args.verbose and details:
        print("\nnon-completed runs:")
        print("\n".join(details))


if __name__ == "__main__":
    main()
