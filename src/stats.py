"""Canonical statistical helpers: effect sizes, bootstrap CIs, BH-FDR."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats as sps


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                     / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def bootstrap_ci(values: Sequence[float], stat=np.nanmean, n_boot: int = 5000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap over runs. Attractor stats are not normal; never
    report mean ± sigma without this."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0, "point": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "ci": ci}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, size=len(v), replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {"n": int(len(v)), "point": float(stat(v)),
            "lo": float(np.quantile(boots, alpha)),
            "hi": float(np.quantile(boots, 1 - alpha)),
            "ci": ci, "n_boot": n_boot}


def bootstrap_centroid_ci(vectors: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Bootstrap CI on centroid position: resample runs, recompute mean
    vector, report the distribution of distances from the full-sample mean."""
    if len(vectors) < 2:
        return {"n": len(vectors), "dispersion_hi95": float("nan")}
    rng = np.random.default_rng(seed)
    full = vectors.mean(axis=0)
    dists = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(vectors), len(vectors))
        dists.append(np.linalg.norm(vectors[idx].mean(axis=0) - full))
    return {"n": int(len(vectors)),
            "dispersion_hi95": float(np.quantile(dists, 0.95)),
            "n_boot": n_boot}


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    try:
        return float(sps.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.1) -> list[dict]:
    """BH-FDR. Returns per-test {p_raw, p_adjusted, significant} preserving
    input order; NaNs pass through unadjusted and non-significant."""
    p = np.asarray(pvals, float)
    out = [{"p_raw": float(pi), "p_adjusted": float("nan"),
            "significant": False} for pi in p]
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return out
    m = len(valid)
    order = valid[np.argsort(p[valid])]
    adj = np.empty(m)
    prev = 1.0
    for rank_from_end in range(m - 1, -1, -1):
        i = order[rank_from_end]
        val = p[i] * m / (rank_from_end + 1)
        prev = min(prev, val)
        adj[rank_from_end] = prev
    for rank, i in enumerate(order):
        out[i]["p_adjusted"] = float(adj[rank])
        out[i]["significant"] = bool(adj[rank] <= q)
    return out
