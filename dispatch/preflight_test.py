"""Preflight: run a single config on Modal and capture full output."""
import modal
import subprocess
import sys
from pathlib import Path

APP_NAME = "attractor-sweep-preflight"
COMMIT_SHA = "79283a087fbb102a6705b80e5fbf0cc882f13111"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "transformers>=4.44", "accelerate>=0.30",
        "bitsandbytes>=0.43", "sentencepiece",
        "numpy>=1.26", "scipy>=1.11", "pandas>=2.1",
        "pyarrow>=14.0", "pyyaml>=6.0", "scikit-learn>=1.3",
        "matplotlib>=3.8", "tqdm>=4.66", "huggingface_hub", "zarr<3",
    )
)

app = modal.App(APP_NAME)

@app.function(
    image=IMAGE,
    gpu="L40S",
    timeout=1800,
    secrets=[modal.Secret.from_name("hf-token")],
)
def preflight():
    import os, subprocess as sp
    
    repo_dir = Path("/workspace/model-noise")
    
    # Clone
    sp.run(["git", "clone", "https://github.com/anthnguyen/model-noise.git", str(repo_dir)], check=True)
    sp.run(["git", "-C", str(repo_dir), "checkout", COMMIT_SHA], check=True)
    
    # Install
    result = sp.run(["pip", "install", "-e", str(repo_dir)], capture_output=True, text=True)
    print("=== PIP INSTALL ===")
    print(result.stdout[-500:])
    if result.returncode != 0:
        print("PIP STDERR:", result.stderr[-500:])
    
    # Generate a single config
    config_dir = repo_dir / "configs" / "preflight"
    config_dir.mkdir(parents=True, exist_ok=True)
    gen_args = [
        "python", str(repo_dir / "scripts" / "gen_configs.py"),
        "--out", str(config_dir),
        "--seeds", "1",
        "--topics", "t01",
        "--conditions", "self_b",
        "--pert-scales", "0",
        "--max-turns", "10",
    ]
    print("\n=== GEN CONFIGS ===")
    sp.run(gen_args, cwd=str(repo_dir), check=True)
    
    configs = list(config_dir.glob("*.yaml"))
    print(f"Generated {len(configs)} config(s): {[c.name for c in configs]}")
    
    if not configs:
        print("NO CONFIGS GENERATED")
        return
    
    # Run a single config
    cfg = configs[0]
    print(f"\n=== RUNNING {cfg.name} ===")
    result = sp.run(
        ["python", "run.py", "--config", str(cfg), "--gpu", "0"],
        cwd=str(repo_dir),
        capture_output=True, text=True, timeout=600,
    )
    
    print(f"EXIT CODE: {result.returncode}")
    print("=== STDOUT ===")
    print(result.stdout)
    if result.stderr:
        print("=== STDERR ===")
        print(result.stderr[-2000:])
    
    # Check outputs
    outputs = repo_dir / "outputs"
    if outputs.exists():
        print("\n=== OUTPUTS ===")
        for p in sorted(outputs.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(repo_dir)} ({p.stat().st_size} bytes)")
    
    # Try HF login
    token = os.environ.get("HF_TOKEN", "")
    print(f"\nHF_TOKEN present: {bool(token)} (length: {len(token)})")
