#!/usr/bin/env python3
"""Sweep driver: closes the tail-crash holes in batch_run.sh.

  * per-run timeout  -> a hung generation is killed, not waited on forever
  * retry with a FRESH process -> transient OOM / network blips self-heal
  * heavy-first ordering -> Gemma/perturbed configs run while the pod is
    freshest; only cheap Qwen/pert0 runs remain at the tail
  * --gpus N -> one worker per GPU, each worker runs its configs serially
    with a fresh CUDA context per config (same isolation batch_run.sh gives)

Usage:
  python scripts/sweep_driver.py configs/sweep --gpus 0 1 \
      --timeout 1800 --retries 1 --order heavy-first

Resumes naturally: completed runs are skipped (run.py --skip-completed),
partial runs resume from their checkpoint. Exit codes: 0 = all configs done,
1 = some failed (report printed), 3 = interrupted (re-run to resume).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEAVY_HINTS = ("gemma", "mixed", "self_b")  # cost-descending sort keys


def heavy_key(cfg_path: Path) -> tuple:
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return (0, 0, 0)
    names = []
    for a in ("agent_a", "agent_b"):
        names.append(str((cfg.get("models", {}).get(a, {}) or {}).get("name", "")))
    is_gemma = any("gemma" in n.lower() for n in names)
    pert = float((cfg.get("perturbation", {}) or {}).get("scale", 0.0) or 0.0)
    turns = int((cfg.get("conversation", {}) or {}).get("max_turns", 0) or 0)
    return (1 if is_gemma else 0, pert, turns)


def is_completed(cfg_path: Path) -> bool:
    """Completed = summary.json says completed (not just 'directory exists')."""
    run_id = Path(cfg_path).stem
    summary = ROOT / "outputs" / "runs" / run_id / "summary.json"
    try:
        data = yaml.safe_load(summary.read_text()) or {}
        return data.get("status") == "completed"
    except Exception:
        return False


def run_one(cfg: Path, gpu: int, timeout: int, log_fh, retries: int) -> int:
    """Run one config with retries. Returns 0 on success, 1 on failure."""
    cmd = [sys.executable, "run.py", "--config", str(cfg),
           "--gpu", str(gpu), "--skip-completed"]
    for attempt in range(retries + 1):
        tag = "fresh" if attempt == 0 else f"retry {attempt}/{retries}"
        print(f"=== {cfg.name} (gpu {gpu}, {tag}, timeout {timeout}s) ===",
              flush=True)
        try:
            rc = subprocess.run(
                cmd, cwd=ROOT, timeout=timeout,
                stdout=log_fh, stderr=subprocess.STDOUT,
            ).returncode
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {timeout}s — killing {cfg.name}", flush=True)
            rc = -1
        if rc == 0:
            return 0
        if rc == 3:  # SIGINT/SIGTERM: stop the whole sweep, resume later
            return 3
        print(f"    FAILED (rc {rc}) {cfg.name} — attempt {attempt + 1}/"
              f"{retries + 1}", flush=True)
        time.sleep(10)
    return 1


def worker(gpu: int, configs: list[Path], timeout: int, retries: int,
           results: dict) -> None:
    log = open(LOG_DIR / f"sweep_gpu{gpu}.log", "a")
    try:
        for cfg in configs:
            if is_completed(cfg):
                print(f"=== {cfg.name} (gpu {gpu}) — already completed, skip ===",
                      flush=True)
                results[cfg.name] = 0
                continue
            rc = run_one(cfg, gpu, timeout, log, retries)
            results[cfg.name] = rc
            if rc == 3:
                return  # interrupted: stop this worker; others get told below
    finally:
        log.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_dir", help="directory of *.yaml run configs")
    ap.add_argument("--gpus", type=int, nargs="+", required=True,
                    help="CUDA device ids to use (one worker each)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-run timeout seconds (default 1800)")
    ap.add_argument("--retries", type=int, default=1,
                    help="fresh-process retries per config (default 1)")
    ap.add_argument("--order", choices=["heavy-first", "lexical", "light-first"],
                    default="heavy-first",
                    help="run order (default heavy-first: Gemma/perturbed first)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the summary.json sanity check after each run")
    args = ap.parse_args()

    cfg_dir = Path(args.config_dir)
    configs = sorted(cfg_dir.glob("*.yaml"))
    if not configs:
        print(f"[fatal] no .yaml configs in {cfg_dir}")
        return 2
    if args.order == "heavy-first":
        configs.sort(key=heavy_key, reverse=True)
    elif args.order == "light-first":
        configs.sort(key=heavy_key, reverse=False)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if Path("/workspace").exists():  # RunPod: HF weights belong on the volume
        os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

    print(f"[sweep_driver] {len(configs)} configs, gpus={args.gpus}, "
          f"order={args.order}, timeout={args.timeout}s, retries={args.retries}")
    if args.order == "heavy-first":
        print("[sweep_driver] first configs (heaviest):")
        for c in configs[:3]:
            print(f"    {c.name}")
        print("    ...")
        print("[sweep_driver] last configs (lightest):")
        for c in configs[-3:]:
            print(f"    {c.name}")

    results: dict = {}
    threads = []
    for i, gpu in enumerate(args.gpus):
        gpu_configs = configs[i::len(args.gpus)]  # shard round-robin
        t = threading.Thread(target=worker, args=(gpu, gpu_configs,
                                                  args.timeout, args.retries,
                                                  results), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    failed = [name for name, rc in results.items() if rc not in (0,)]
    interrupted = any(rc == 3 for rc in results.values())
    print(f"\n[sweep_driver] done: {len(results) - len(failed)}/{len(results)} "
          f"configs OK, {len(failed)} failed")
    if interrupted:
        print("[sweep_driver] interrupted (SIGINT/SIGTERM) — rerun to resume")
        return 3
    if failed:
        print("[sweep_driver] failed configs (re-run to resume; these are "
              "retried fresh):")
        for name in failed:
            print(f"    {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
