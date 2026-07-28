"""Convergence detection, permutation null, and springback measurement.

Convergence criterion (Ko & Geiping style): within a sliding window of W
turns per agent, the mean cosine distance between consecutive residual-stream
vectors falls below `threshold`, sustained for W consecutive windows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def consecutive_distances(acts: np.ndarray) -> np.ndarray:
    """acts: (T, hidden) per-agent turn activations -> (T-1,) distances."""
    if len(acts) < 2:
        return np.array([])
    return np.array([cosine_distance(acts[i], acts[i + 1])
                     for i in range(len(acts) - 1)])


def window_score(acts: np.ndarray, window: int) -> Optional[float]:
    """Mean consecutive cosine distance over the trailing `window` turns."""
    if len(acts) < window:
        return None
    tail = acts[-window:]
    d = consecutive_distances(tail)
    return float(d.mean()) if len(d) else None


@dataclass
class ConvergenceState:
    """Tracks convergence online for one agent."""
    window: int
    threshold: float
    min_turns: int
    consecutive_hits: int = 0
    converged_at: Optional[int] = None   # agent-turn index when criterion met

    def update(self, acts: np.ndarray) -> bool:
        """Call after each of this agent's turns with all its activations so
        far (t, hidden). Returns True once converged (latching)."""
        if self.converged_at is not None:
            return True
        t = len(acts)
        if t < max(self.min_turns, self.window):
            return False
        score = window_score(acts, self.window)
        if score is not None and score < self.threshold:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0
        if self.consecutive_hits >= self.window:
            self.converged_at = t - 1
            return True
        return False


def joint_converged(states: list[ConvergenceState], mode: str) -> bool:
    a, b = (s.converged_at is not None for s in states)
    return {"both": a and b, "either": a or b,
            "agent_a": a, "agent_b": b}[mode]


def attractor_centroid(acts: np.ndarray, converged_at: int,
                       window: int) -> np.ndarray:
    """Mean activation over the converged window ending at converged_at."""
    lo = max(0, converged_at - window + 1)
    return acts[lo:converged_at + 1].mean(axis=0)


def permutation_test(acts: np.ndarray, window: int, threshold: float,
                     min_turns: int, n_perm: int = 200,
                     seed: int = 0) -> dict:
    """Null check: shuffle turn order and re-run detection. If 'convergence'
    still fires often on shuffled trajectories, the signal is an artifact of
    embedding geometry rather than temporal dynamics.

    Returns {p_value, null_rate, observed}: p_value is the fraction of
    shuffles that converge (lower = real dynamics).
    """
    def detect(a: np.ndarray) -> bool:
        st = ConvergenceState(window, threshold, min_turns)
        for t in range(1, len(a) + 1):
            if st.update(a[:t]):
                return True
        return False

    valid = acts[~np.isnan(acts).any(axis=1)]
    observed = detect(valid)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = valid[rng.permutation(len(valid))]
        hits += detect(perm)
    return {
        "observed_converged": bool(observed),
        "null_rate": hits / n_perm,
        "p_value": (hits + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


@dataclass
class SpringbackResult:
    perturbed_at: int                    # agent-turn index of injection
    max_displacement: float              # peak cosine distance from centroid
    recovery_turns: Optional[int]        # turns to re-enter recovery_threshold
    recovered: bool
    recovery_threshold: float
    trace: list[float]                   # distance-to-centroid per post turn

    def to_dict(self) -> dict:
        return asdict(self)


def measure_springback(acts: np.ndarray, centroid: np.ndarray,
                       perturbed_at: int, recovery_threshold: float,
                       sustain: int = 2) -> SpringbackResult:
    """acts: (T, hidden) for the perturbed agent. Distance-to-centroid uses
    cosine distance; recovery = distance < recovery_threshold sustained for
    `sustain` consecutive turns."""
    post = acts[perturbed_at:]
    trace = [cosine_distance(v, centroid) for v in post
             if not np.isnan(v).any()]
    recovery_turns = None
    run = 0
    for i, d in enumerate(trace):
        if d < recovery_threshold:
            run += 1
            if run >= sustain:
                recovery_turns = i - sustain + 1
                break
        else:
            run = 0
    return SpringbackResult(
        perturbed_at=perturbed_at,
        max_displacement=float(max(trace)) if trace else float("nan"),
        recovery_turns=recovery_turns,
        recovered=recovery_turns is not None,
        recovery_threshold=recovery_threshold,
        trace=[float(d) for d in trace],
    )
