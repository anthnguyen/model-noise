#!/usr/bin/env python
"""Run one experiment from a YAML config.

    python run.py --config configs/mvp/self_a.yaml
    python run.py --config configs/sweep/xxx.yaml --gpu 1
    python run.py --config ... --override conversation.max_turns=20

Exit codes: 0 completed, 3 interrupted (resumable), 1 failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_config
from src.storage import run_status


def parse_overrides(pairs: list[str]) -> dict:
    """key.path=value pairs; values parsed as YAML scalars."""
    import yaml
    out: dict = {}
    for pair in pairs:
        key, _, val = pair.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=None,
                    help="shortcut for models.*.device=cuda:N")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dot.path=value config overrides")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoint (run.resume=false)")
    ap.add_argument("--skip-completed", action="store_true",
                    help="exit 0 immediately if this run already completed")
    args = ap.parse_args()

    overrides = parse_overrides(args.override)
    if args.gpu is not None:
        overrides.setdefault("models", {})
        for agent in ("agent_a", "agent_b"):
            overrides["models"].setdefault(agent, {})["device"] = f"cuda:{args.gpu}"
    if args.fresh:
        overrides.setdefault("run", {})["resume"] = False

    cfg = load_config(args.config, overrides)

    if args.skip_completed and run_status(cfg.run_dir) == "completed":
        print(f"[skip] {cfg.run_id} already completed")
        return 0

    from src.conversation import ConversationRunner
    runner = ConversationRunner(cfg)
    summary = runner.run()

    # verbalization on final centroids (best-effort: never lose a run to it)
    if cfg["verbalize"]["enabled"] and summary["status"] == "completed":
        try:
            import numpy as np
            from src.storage import atomic_write_json
            from src.verbalize import verbalize_run
            pre = []
            for agent in range(2):
                acts = runner._agent_acts(agent)
                w = cfg["convergence"]["window"]
                pre.append(acts[:w].mean(axis=0) if len(acts) >= w else None)
            verb = verbalize_run(cfg, runner.backends, runner.centroids, pre)
            atomic_write_json(cfg.run_dir / "verbalizations.json", verb)
        except Exception as e:
            print(f"[warn] verbalization failed: {e}", file=sys.stderr)

    if cfg["figures"]["per_run"] and summary["status"] == "completed":
        try:
            from src.plotting import trajectory_plot
            trajectory_plot(cfg.run_dir, dpi=cfg["figures"]["dpi"])
        except Exception as e:
            print(f"[warn] per-run figure failed: {e}", file=sys.stderr)

    print(json.dumps(
        {k: summary.get(k) for k in
         ("run_id", "status", "joint_converged", "agent0_time_to_attractor",
          "agent1_time_to_attractor", "basin_depth_turns", "wall_seconds")},
        indent=2))
    return {"completed": 0, "interrupted": 3}.get(summary["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
