#!/usr/bin/env python
"""Generate the full sweep of experiment configs.

    python scripts/gen_configs.py \
        --out configs/sweep \
        --seeds 10 --topics t01 t02 t03 t04 t05 t06 t07 t08 t09 t10 \
        --conditions self_a self_b mixed \
        --pert-scales 0 0.1 0.3 0.5 \
        --max-turns 100

Writes one YAML per run + sweep_manifest.csv listing them all.
Default matrix: 10 seeds x 10 topics x 3 conditions x 4 pert levels = 1200
(the plan's 900 = 3 pert levels; scale 0 rows double as unperturbed baseline).

For a mock sweep (laptop rehearsal of the whole batch machinery):
    python scripts/gen_configs.py --out configs/sweep_mock --backend mock \
        --seeds 3 --topics t01 t02 --pert-scales 0 0.6 --max-turns 40
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = {
    "A": {"name": "Qwen/Qwen2.5-7B-Instruct", "quantization": "4bit"},
    "B": {"name": "google/gemma-2-9b-it", "quantization": "4bit"},
}
MOCK_MODELS = {"A": {"name": "mock-alpha"}, "B": {"name": "mock-beta"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="configs/sweep")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--topics", nargs="+", default=None,
                    help="topic ids from data/seed_strings.yaml (default all)")
    ap.add_argument("--conditions", nargs="+",
                    default=["self_a", "self_b", "mixed"])
    ap.add_argument("--pert-scales", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 0.5])
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--seed-file", default="data/seed_strings.yaml")
    args = ap.parse_args()

    with open(args.seed_file) as f:
        seed_data = yaml.safe_load(f)
    topics = args.topics or sorted(seed_data["topics"].keys())
    models = MOCK_MODELS if args.backend == "mock" else MODELS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in args.conditions:
        pair = {"self_a": ("A", "A"), "self_b": ("B", "B"),
                "mixed": ("A", "B")}[cond]
        for topic in topics:
            for seed in range(args.seeds):
                for scale in args.pert_scales:
                    cfg = {
                        "condition": cond,
                        "topic_id": topic,
                        "backend": args.backend,
                        "run": {"seed": seed},
                        "models": {
                            "agent_a": dict(models[pair[0]]),
                            "agent_b": dict(models[pair[1]]),
                        },
                        "conversation": {
                            "max_turns": args.max_turns,
                            "seed_string": seed_data["topics"][topic],
                            "framing": seed_data["framing"],
                        },
                        "perturbation": {
                            "enabled": scale > 0,
                            "scale": scale,
                        },
                    }
                    if args.backend == "mock":
                        cfg["convergence"] = {"threshold": 0.05, "min_turns": 8}
                    pert_tag = f"p{scale:g}".replace(".", "")
                    name = f"{cond}_{topic}_s{seed}_{pert_tag}.yaml"
                    with open(out / name, "w") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)
                    rows.append({"config": str(out / name), "condition": cond,
                                 "topic": topic, "seed": seed,
                                 "pert_scale": scale})

    with open(out / "sweep_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} configs to {out}/ (+ sweep_manifest.csv)")


if __name__ == "__main__":
    main()
