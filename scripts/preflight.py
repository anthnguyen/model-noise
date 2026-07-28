#!/usr/bin/env python
"""Preflight check before launching real runs on a cluster.

    python scripts/preflight.py                 # core-only check (mock runs)
    python scripts/preflight.py --gpu           # also check the GPU stack
    python scripts/preflight.py --gpu --models  # also try loading tokenizers

Exits non-zero if anything required is missing.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

CORE = ["numpy", "scipy", "pandas", "pyarrow", "zarr", "yaml", "sklearn",
        "matplotlib", "tqdm"]
GPU = ["torch", "transformers", "accelerate", "bitsandbytes"]


def check(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"  ok    {name}")
        return True
    except ImportError as e:
        print(f"  MISS  {name}  ({e})")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--models", action="store_true",
                    help="try loading tokenizers for the default models")
    args = ap.parse_args()
    ok = True

    print("[core packages]")
    for p in CORE:
        ok &= check(p)

    if args.gpu:
        print("[gpu stack]")
        for p in GPU:
            ok &= check(p)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  ok    cuda:{i} {props.name} "
                          f"{props.total_memory / 1e9:.0f} GB")
            else:
                print("  MISS  CUDA not available")
                ok = False
        except ImportError:
            pass

    if args.models:
        print("[model access]")
        try:
            from transformers import AutoTokenizer
            for m in ("Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"):
                try:
                    AutoTokenizer.from_pretrained(m)
                    print(f"  ok    {m}")
                except Exception as e:
                    print(f"  MISS  {m}: {type(e).__name__}: {e}")
                    print("        (gemma is gated: huggingface-cli login "
                          "+ accept license)")
                    ok = False
        except ImportError:
            ok = False

    print("[disk]")
    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    print(f"  {'ok  ' if free_gb > 20 else 'WARN'}  {free_gb:.0f} GB free "
          "(sweep needs ~15 GB)")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
