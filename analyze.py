#!/usr/bin/env python
"""Aggregate analysis + figures over all completed runs.

    python analyze.py                          # analyze outputs/runs
    python analyze.py --runs-dir outputs/runs --analysis-id demo1
    python analyze.py --skip-permutation       # faster, skips null check
"""

from __future__ import annotations

import argparse

from src.analysis import analyze


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--out-root", default="outputs/analysis")
    ap.add_argument("--analysis-id", default=None,
                    help="default: timestamped, never overwrites")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--min-turns", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--fdr-q", type=float, default=0.1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()

    analyze(
        runs_dir=args.runs_dir, out_root=args.out_root,
        analysis_id=args.analysis_id,
        convergence={"window": args.window, "threshold": args.threshold,
                     "min_turns": args.min_turns},
        dpi=args.dpi, n_perm=args.n_perm, fdr_q=args.fdr_q,
        skip_permutation=args.skip_permutation,
    )


if __name__ == "__main__":
    main()
