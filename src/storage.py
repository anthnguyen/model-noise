"""Run storage: incremental, crash-safe, resumable.

Layout of one run directory (outputs/runs/<run_id>/):

    config.yaml         frozen resolved config (reproduce the run from this)
    manifest.json       status, env info, timestamps, config hash
    transcript.jsonl    one line per message (append-only, flushed per turn)
    acts_agent0.zarr    (max_turns, hidden_a) fp16, chunked per turn
    acts_agent1.zarr    (max_turns, hidden_b) fp16
    metrics.parquet     per-turn metrics table (rewritten atomically on flush)
    checkpoint.json     resume state (next turn index)
    summary.json        run-level results, written at the end
    figures/            per-run auto-generated plots

JSON writes go through a temp file + os.replace (atomic on POSIX), so a kill
at any moment leaves the run resumable.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import zarr


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def env_info() -> dict:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    for pkg in ("transformers", "zarr", "numpy"):
        try:
            info[pkg] = __import__(pkg).__version__
        except ImportError:
            info[pkg] = None
    return info


class RunStore:
    def __init__(self, run_dir: Path, max_turns: int,
                 hidden_sizes: tuple[int, int], config_yaml_dump, config_hash: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        self.max_turns = max_turns

        config_yaml_dump(self.run_dir / "config.yaml")

        self.acts = []
        for i, h in enumerate(hidden_sizes):
            path = str(self.run_dir / f"acts_agent{i}.zarr")
            arr = zarr.open(
                path, mode="a", shape=(max_turns, h), chunks=(1, h),
                dtype="f2", fill_value=np.nan,
            )
            # zarr.open(mode="a") keeps the on-disk shape; grow it if a
            # resumed run was reconfigured with more turns
            if arr.shape[0] < max_turns:
                arr.resize(max_turns, h)
            self.acts.append(arr)

        self._transcript_path = self.run_dir / "transcript.jsonl"
        self._metrics_rows: list[dict] = []
        self._metrics_path = self.run_dir / "metrics.parquet"
        if self._metrics_path.exists():
            self._metrics_rows = pd.read_parquet(self._metrics_path).to_dict("records")

        self._manifest_path = self.run_dir / "manifest.json"
        if not self._manifest_path.exists():
            atomic_write_json(self._manifest_path, {
                "status": "running",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config_hash": config_hash,
                "env": env_info(),
            })
        else:
            self.update_manifest(status="running",
                                 resumed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    # -- manifest / checkpoint ---------------------------------------------- #

    def update_manifest(self, **fields) -> None:
        with open(self._manifest_path) as f:
            m = json.load(f)
        m.update(fields)
        atomic_write_json(self._manifest_path, m)

    def save_checkpoint(self, next_turn: int, extra: Optional[dict] = None) -> None:
        atomic_write_json(self.run_dir / "checkpoint.json",
                          {"next_turn": next_turn, **(extra or {})})

    def load_checkpoint(self) -> Optional[dict]:
        p = self.run_dir / "checkpoint.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    # -- per-turn data ------------------------------------------------------ #

    def append_message(self, record: dict) -> None:
        with open(self._transcript_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_transcript(self) -> list[dict]:
        if not self._transcript_path.exists():
            return []
        out = []
        with open(self._transcript_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def write_activation(self, turn: int, agent: int, vec: np.ndarray) -> None:
        self.acts[agent][turn, :] = vec.astype(np.float16)

    def read_activations(self, agent: int) -> np.ndarray:
        return np.asarray(self.acts[agent][:], dtype=np.float32)

    def add_metrics_row(self, row: dict) -> None:
        self._metrics_rows.append(row)

    def flush_metrics(self) -> None:
        if self._metrics_rows:
            df = pd.DataFrame(self._metrics_rows)
            tmp = self._metrics_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, self._metrics_path)

    # -- finalization ------------------------------------------------------- #

    def write_summary(self, summary: dict) -> None:
        atomic_write_json(self.run_dir / "summary.json", summary)

    def finalize(self, status: str = "completed") -> None:
        self.flush_metrics()
        self.update_manifest(
            status=status, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run_status(run_dir: Path) -> Optional[str]:
    p = Path(run_dir) / "manifest.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get("status")
    except (json.JSONDecodeError, OSError):
        return "corrupt"
