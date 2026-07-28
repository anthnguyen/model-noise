"""Config loading, validation, and run-id construction.

Every experiment is fully described by one YAML file. The resolved config
(defaults filled in) is frozen into the run directory so a run can always be
reproduced from its own artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULTS: dict[str, Any] = {
    "run": {
        "run_id": None,          # auto-built from condition/seed/topic/pert if null
        "output_dir": "outputs/runs",
        "seed": 0,               # master RNG seed; per-turn seeds derived from it
        "save_every_turns": 5,   # flush zarr/parquet at least this often
        "resume": True,          # pick up an interrupted run from its checkpoint
    },
    "backend": "hf",             # "hf" (GPU) or "mock" (CPU dry-run)
    "models": {
        "agent_a": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "4bit",     # "4bit" | "8bit" | "none"
            "device": "cuda:0",
            "capture_layer": None,      # null => int(0.75 * num_layers)
            "dtype": "bfloat16",
        },
        "agent_b": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "4bit",
            "device": "cuda:0",
            "capture_layer": None,
            "dtype": "bfloat16",
        },
    },
    "conversation": {
        "max_turns": 100,        # one turn = one message by one agent
        "seed_string": "x7m3_k9p2_v4n1",
        "framing": "This is a cipher. Discuss what it encodes.",
        "max_new_tokens": 200,
        "history_window": None,  # int => keep only last N messages in context
        "generation": {"temperature": 0.7, "top_p": 0.9},
    },
    "convergence": {
        "window": 5,             # sliding window (turns per agent)
        "threshold": 0.01,       # mean consecutive cosine distance below this
        "min_turns": 10,         # don't test before this many turns per agent
        "converge_on": "both",   # "both" | "either" | "agent_a" | "agent_b"
    },
    "perturbation": {
        "enabled": False,
        "scale": 0.3,            # fraction of the running activation norm
        "turn": None,            # fixed turn index; null => attractor + patience
        "patience": 3,           # turns after detection before injecting
        "target_agent": 0,       # 0 = agent_a, 1 = agent_b
        "duration_turns": 1,     # how many of the target's turns to perturb
    },
    "metrics": {
        "compute_cross_surprise": True,  # partner scores generator's tokens
    },
    "verbalize": {
        "enabled": True,
        "methods": ["logit_lens", "patchscope"],
        "max_new_tokens": 60,
        "top_k_tokens": 15,      # for logit lens
    },
    "figures": {
        "per_run": True,         # auto-generate trajectory (+recovery) plots
        "dpi": 200,
    },
    "wandb": {"enabled": False, "project": "attractors"},
    # Labels used for grouping in analysis; no effect on execution
    "condition": "self_a",       # "self_a" | "self_b" | "mixed"
    "topic_id": "t01",
}


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _slug(s: str, max_len: int = 24) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "-", str(s)).strip("-").lower()
    return s[:max_len]


@dataclass
class Config:
    raw: dict[str, Any]
    path: Optional[Path] = None

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def run_id(self) -> str:
        rid = self.raw["run"].get("run_id")
        if rid:
            return rid
        pert = self.raw["perturbation"]
        pert_tag = f"pert{pert['scale']:g}".replace(".", "") if pert["enabled"] else "pert0"
        rid = "_".join(
            [
                _slug(self.raw["condition"]),
                f"seed{self.raw['run']['seed']}",
                _slug(self.raw["topic_id"]),
                _slug(self.raw["conversation"]["seed_string"], 16),
                pert_tag,
            ]
        )
        self.raw["run"]["run_id"] = rid
        return rid

    @property
    def run_dir(self) -> Path:
        return Path(self.raw["run"]["output_dir"]) / self.run_id

    def config_hash(self) -> str:
        canon = yaml.safe_dump(self.raw, sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)


def validate(cfg: Config) -> None:
    errors = []
    if cfg["backend"] not in ("hf", "mock"):
        errors.append(f"backend must be 'hf' or 'mock', got {cfg['backend']!r}")
    if cfg["convergence"]["converge_on"] not in ("both", "either", "agent_a", "agent_b"):
        errors.append("convergence.converge_on must be both|either|agent_a|agent_b")
    if cfg["condition"] not in ("self_a", "self_b", "mixed"):
        errors.append("condition must be self_a|self_b|mixed")
    if cfg["conversation"]["max_turns"] < 2:
        errors.append("conversation.max_turns must be >= 2")
    if cfg["perturbation"]["target_agent"] not in (0, 1):
        errors.append("perturbation.target_agent must be 0 or 1")
    w = cfg["convergence"]["window"]
    if w < 2:
        errors.append("convergence.window must be >= 2")
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))


def load_config(path: str | Path, overrides: Optional[dict] = None) -> Config:
    path = Path(path)
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    merged = _deep_update(DEFAULTS, user_cfg)
    if overrides:
        merged = _deep_update(merged, overrides)
    cfg = Config(raw=merged, path=path)
    validate(cfg)
    return cfg
