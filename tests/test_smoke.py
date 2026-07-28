"""End-to-end smoke tests on the mock backend (no GPU, no torch).

    python -m pytest tests/ -q          # or: python tests/test_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_config(tmp: Path, name: str, extra: dict) -> dict:
    import json
    import yaml
    cfg = {
        "condition": "self_a", "topic_id": "t01", "backend": "mock",
        "run": {"seed": 1, "output_dir": str(tmp / "runs")},
        "models": {"agent_a": {"name": "mock-alpha"},
                   "agent_b": {"name": "mock-alpha"}},
        "conversation": {"max_turns": 40, "seed_string": "x7m3"},
        "convergence": {"threshold": 0.05, "min_turns": 8},
    }
    for k, v in extra.items():
        cfg.setdefault(k, {}).update(v) if isinstance(v, dict) else cfg.update({k: v})
    cfg_path = tmp / f"{name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    proc = subprocess.run(
        [sys.executable, "run.py", "--config", str(cfg_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    run_dirs = list((tmp / "runs").iterdir())
    summaries = [d for d in run_dirs if (d / "summary.json").exists()]
    assert summaries, f"no summary written; dirs={run_dirs}"
    latest = max(summaries, key=lambda d: (d / "summary.json").stat().st_mtime)
    with open(latest / "summary.json") as f:
        return json.load(f) | {"_run_dir": str(latest)}


def test_convergence_and_artifacts(tmp_path):
    s = _run_config(tmp_path, "basic", {})
    assert s["status"] == "completed"
    assert s["joint_converged"], "mock dynamics should converge"
    assert s["agent0_time_to_attractor"] is not None
    d = Path(s["_run_dir"])
    for fname in ("config.yaml", "manifest.json", "transcript.jsonl",
                  "metrics.parquet", "acts_agent0.zarr", "summary.json",
                  "verbalizations.json", "figures/trajectory.png"):
        assert (d / fname).exists(), f"missing {fname}"
    acts = __import__("zarr").open(str(d / "acts_agent0.zarr"), mode="r")[:]
    valid = ~np.isnan(np.asarray(acts, dtype=np.float32)).any(axis=1)
    assert valid.sum() == 40, "one activation per round expected"


def test_perturbation_and_springback(tmp_path):
    s = _run_config(tmp_path, "pert", {
        "perturbation": {"enabled": True, "scale": 0.6, "patience": 2},
        "conversation": {"max_turns": 60, "seed_string": "x7m3"},
    })
    assert s["perturb_injected_at"] is not None, "perturbation never fired"
    assert "springback" in s
    assert s["springback"]["max_displacement"] > 0.05, \
        "perturbation should displace the state"
    assert s["springback"]["recovered"], \
        "mock contraction should spring back"


def test_resume_determinism(tmp_path):
    """Interrupting at round k and resuming must equal an uninterrupted run."""
    import json
    import yaml
    cfg = {
        "condition": "self_a", "topic_id": "t01", "backend": "mock",
        "run": {"seed": 7, "output_dir": str(tmp_path / "runs_a")},
        "models": {"agent_a": {"name": "mock-alpha"},
                   "agent_b": {"name": "mock-alpha"}},
        "conversation": {"max_turns": 20, "seed_string": "qq4z"},
        "convergence": {"threshold": 0.05, "min_turns": 8},
    }
    # run A: straight through
    p = tmp_path / "a.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    subprocess.run([sys.executable, "run.py", "--config", str(p)],
                   cwd=ROOT, check=True, capture_output=True, timeout=300)
    # run B: 10 rounds, then resume to 20
    cfg["run"]["output_dir"] = str(tmp_path / "runs_b")
    cfg["conversation"]["max_turns"] = 10
    p2 = tmp_path / "b1.yaml"
    with open(p2, "w") as f:
        yaml.safe_dump(cfg, f)
    subprocess.run([sys.executable, "run.py", "--config", str(p2)],
                   cwd=ROOT, check=True, capture_output=True, timeout=300)
    cfg["conversation"]["max_turns"] = 20
    cfg["run"]["run_id"] = next(
        (tmp_path / "runs_b").iterdir()).name  # same run dir
    p3 = tmp_path / "b2.yaml"
    with open(p3, "w") as f:
        yaml.safe_dump(cfg, f)
    subprocess.run([sys.executable, "run.py", "--config", str(p3)],
                   cwd=ROOT, check=True, capture_output=True, timeout=300)

    ta = [json.loads(l)["text"] for l in open(
        next((tmp_path / "runs_a").iterdir()) / "transcript.jsonl")]
    tb = [json.loads(l)["text"] for l in open(
        next((tmp_path / "runs_b").iterdir()) / "transcript.jsonl")]
    assert ta == tb, "resumed transcript diverged from uninterrupted run"


def test_analysis_pipeline(tmp_path):
    for seed in range(3):
        _run_config(tmp_path, f"an{seed}", {"run": {
            "seed": seed, "output_dir": str(tmp_path / "runs")}})
    proc = subprocess.run(
        [sys.executable, "analyze.py", "--runs-dir", str(tmp_path / "runs"),
         "--out-root", str(tmp_path / "analysis"), "--analysis-id", "t",
         "--threshold", "0.05", "--min-turns", "8", "--n-perm", "50"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = tmp_path / "analysis" / "t"
    for fname in ("run_table.csv", "stats_tests.csv", "analysis_summary.json",
                  "figures/convergence_heatmap.png"):
        assert (out / fname).exists(), f"missing {fname}"


if __name__ == "__main__":
    import tempfile
    for fn in (test_convergence_and_artifacts, test_perturbation_and_springback,
               test_resume_determinism, test_analysis_pipeline):
        with tempfile.TemporaryDirectory() as td:
            print(f"-- {fn.__name__}")
            fn(Path(td))
    print("ALL SMOKE TESTS PASSED")
