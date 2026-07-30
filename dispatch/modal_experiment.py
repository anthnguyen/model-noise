"""Modal experiment: Gemma-2-9B self-play attractor sweep.

Deploy once:
    modal deploy modal_experiment.py

Then dispatch from anywhere:
    python -c "import modal_experiment as m; m.dispatch({'models': ['google/gemma-2-9b-it', 'google/gemma-2-9b-it'], 'conditions': ['self_b'], 'seeds': 10, 'topics': ['t01','t02','t03','t04','t05','t06','t07','t08','t09','t10'], 'pert_scales': [0,0.1,0.3,0.5], 'max_turns': 100})"

Architecture:
- Half 1 (top): Remote function. Lives on Modal. Runs the experiment, uploads to HF.
- Half 2 (bottom): Local dispatch/poll/fetch_logs. Agent calls these.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# -------- Config -------------------------------------------------------------

APP_NAME = "attractor-sweep"
GPU_PREFERENCE = ["L40S", "A100", "T4"]  # L40S = best value (48GB, ~$1.90/hr), A100 = fast, T4 = cheap fallback
TIMEOUT_HOURS = 12
HF_REPO = "metametal/model-noise-results"
COMMIT_SHA = "79283a087fbb102a6705b80e5fbf0cc882f13111"

# -------- Half 1: Remote function -------------------------------------------

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "accelerate>=0.30",
        "bitsandbytes>=0.43",
        "sentencepiece",
        "numpy>=1.26",
        "scipy>=1.11",
        "pandas>=2.1",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "huggingface_hub",
        "zarr<3",
    )
)

app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name("attractor-results", create_if_missing=True)


@app.function(
    image=IMAGE,
    gpu=GPU_PREFERENCE,
    volumes={"/results": results_vol},
    timeout=TIMEOUT_HOURS * 3600,
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_sweep(cfg: dict) -> dict:
    """Clone model-noise repo, run Gemma self-play sweep, upload results to HF.

    cfg keys:
        models: [model_a_name, model_b_name]
        conditions: ["self_a", "self_b", "mixed"]
        seeds: int
        topics: list[str]  e.g. ["t01", "t02", ...]
        pert_scales: list[float]  e.g. [0, 0.1, 0.3, 0.5]
        max_turns: int
    """
    import subprocess as sp
    from datetime import datetime, timezone

    jid = cfg.get("job_id", "unknown")
    repo_dir = Path("/workspace/model-noise")

    # Clone and install
    if not repo_dir.exists():
        sp.run(["git", "clone", "https://github.com/anthnguyen/model-noise.git", str(repo_dir)], check=True)
    sp.run(["git", "-C", str(repo_dir), "checkout", COMMIT_SHA], check=True)
    sp.run(["pip", "install", "-q", "-e", str(repo_dir)], check=True)

    # HF login for gated Gemma
    token = os.environ.get("HF_TOKEN", "")
    if token:
        sp.run(["huggingface-cli", "login", "--token", token], check=False)

    # Generate sweep configs
    conditions = cfg.get("conditions", ["self_b"])
    seeds = cfg.get("seeds", 10)
    topics = cfg.get("topics", [f"t{i:02d}" for i in range(1, 11)])
    pert_scales = cfg.get("pert_scales", [0, 0.1, 0.3, 0.5])
    max_turns = cfg.get("max_turns", 100)

    config_dir = repo_dir / "configs" / "sweep"
    gen_args = [
        "python", str(repo_dir / "scripts" / "gen_configs.py"),
        "--out", str(config_dir),
        "--seeds", str(seeds),
        "--topics", *topics,
        "--conditions", *conditions,
        "--pert-scales", *(str(p) for p in pert_scales),
        "--max-turns", str(max_turns),
    ]
    print(f"[modal] Generating configs: {seeds}s x {len(topics)}t x {len(conditions)}c x {len(pert_scales)}p")
    sp.run(gen_args, cwd=str(repo_dir), check=True)

    n_configs = len(list(config_dir.glob("*.yaml")))
    print(f"[modal] Generated {n_configs} configs")

    # Run batch
    print(f"[modal] Running batch sweep...")
    batch_result = sp.run(
        ["bash", str(repo_dir / "scripts" / "batch_run.sh"), str(config_dir), "0"],
        cwd=str(repo_dir),
    )

    # Analyze
    print("[modal] Running analysis...")
    sp.run(["python", str(repo_dir / "analyze.py")], cwd=str(repo_dir), check=False)

    # Upload to HF
    print("[modal] Uploading results to HuggingFace...")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=str(repo_dir / "outputs"),
        repo_id=HF_REPO,
        repo_type="dataset",
        path_in_repo=f"gemma-self_b-{ts}",
    )
    results_vol.commit()

    # Summary
    run_dirs = list((repo_dir / "outputs" / "runs").glob("*"))
    completed = len(run_dirs)
    summary = {
        "job_id": jid,
        "completed_runs": completed,
        "total_configs": n_configs,
        "batch_exit": batch_result.returncode,
        "huggingface_path": f"{HF_REPO}/gemma-self_b-{ts}",
    }
    print(json.dumps(summary, indent=2))
    return summary


# -------- Half 2: Local dispatch --------------------------------------------

def dispatch(cfg: dict, gpu: str | None = None, est_minutes: int = 120) -> dict:
    """Agent-facing. Returns immediately with a call_id. Never raises.

    Example:
        >>> cfg = {"conditions": ["self_b"], "seeds": 10, "topics": ["t01","t02","t03"], "pert_scales": [0,0.3], "max_turns": 100}
        >>> result = dispatch(cfg)
    """
    import hashlib

    blob = json.dumps(cfg, sort_keys=True)
    jid = f"ge-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"
    cfg["job_id"] = jid

    try:
        fn = modal.Function.from_name(APP_NAME, "run_sweep")
        call = fn.spawn(cfg)
    except Exception as e:
        return {"ok": False, "job_id": jid, "error": repr(e)}

    return {
        "ok": True,
        "job_id": jid,
        "call_id": call.object_id,
        "est_minutes": est_minutes,
    }


def poll(call_id: str) -> dict:
    """Check status of a Modal function call."""
    try:
        call = modal.FunctionCall.from_id(call_id)
    except Exception as e:
        return {"status": "unknown", "error": repr(e)}
    try:
        result = call.get(timeout=0)
        return {"status": "done", "result": result}
    except TimeoutError:
        return {"status": "running"}
    except Exception as e:
        return {"status": "failed", "error": repr(e)}


def fetch_logs(lines: int = 50) -> str:
    """Return recent app logs."""
    try:
        out = subprocess.run(
            ["modal", "app", "logs", APP_NAME],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        return f"[log fetch failed: {e!r}]"
    text = out.stdout or out.stderr or ""
    return "\n".join(text.splitlines()[-lines:]) or "[no logs]"


# -------- CLI entrypoint ----------------------------------------------------

@app.local_entrypoint()
def main(
    condition: str = "self_b",
    seeds: int = 10,
    topics: str = "t01,t02,t03",  # comma-separated
    pert_scales: str = "0,0.3",    # comma-separated
    max_turns: int = 100,
):
    """Deploy and run the sweep directly."""
    cfg = {
        "conditions": [condition],
        "seeds": seeds,
        "topics": topics.split(","),
        "pert_scales": [float(p) for p in pert_scales.split(",")],
        "max_turns": max_turns,
    }
    result = dispatch(cfg)
    print(json.dumps(result, indent=2))
    if result["ok"]:
        print(f"\nPoll: modal_experiment.poll('{result['call_id']}')")