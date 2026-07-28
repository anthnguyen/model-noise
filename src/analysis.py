"""Aggregate analysis across completed runs.

Reads outputs/runs/*/summary.json (+ metrics.parquet, centroids, zarr), and
produces, under outputs/analysis/<analysis_id>/:

    run_table.csv           one row per run: all run-level metrics
    stats_tests.csv         pairwise condition tests: Cohen's d, Mann-Whitney
                            p (raw + BH-adjusted), bootstrap CIs
    clusters.json           DBSCAN attractor clustering per model space
    pca_variance.json       variance explained per model space
    permutation_tests.json  shuffled-order null check per run
    analysis_summary.json   headline numbers
    figures/                convergence heatmap, attractor map,
                            verbalizer table, recovery curves

`analysis_id` defaults to a timestamp so re-analyses never overwrite.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .attractor import cosine_distance, permutation_test
from .stats import (benjamini_hochberg, bootstrap_centroid_ci, bootstrap_ci,
                    cohens_d, mann_whitney)
from . import plotting


def discover_runs(runs_dir: Path, include_incomplete: bool = False) -> list[Path]:
    runs = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not (d / "summary.json").exists():
            continue
        with open(d / "manifest.json") as f:
            status = json.load(f).get("status")
        if status == "completed" or include_incomplete:
            runs.append(d)
    return runs


def build_run_table(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        with open(d / "summary.json") as f:
            s = json.load(f)
        sb = s.get("springback") or {}
        rows.append({
            "run_id": s["run_id"],
            "run_dir": str(d),
            "condition": s["condition"],
            "topic_id": s["topic_id"],
            "seed": s["seed"],
            "model_a": s["model_a"],
            "model_b": s["model_b"],
            "perturb_scale": s.get("perturb_scale", 0.0),
            "joint_converged": s.get("joint_converged", False),
            "agent0_converged": s.get("agent0_converged", False),
            "agent1_converged": s.get("agent1_converged", False),
            "time_to_attractor": _joint_tta(s),
            "basin_depth_turns": s.get("basin_depth_turns"),
            "springback_recovered": sb.get("recovered"),
            "max_displacement": sb.get("max_displacement"),
            "perturb_injected_at": s.get("perturb_injected_at"),
            "wall_seconds": s.get("wall_seconds"),
        })
    df = pd.DataFrame(rows)
    # attach converged-window token entropy & surprise from per-turn metrics
    ent, sur, dom = [], [], []
    for d in run_dirs:
        mp = Path(d) / "metrics.parquet"
        if not mp.exists():
            ent.append(np.nan); sur.append(np.nan); dom.append(np.nan)
            continue
        m = pd.read_parquet(mp)
        conv = m[m["converged"]]
        ent.append(conv["mean_token_entropy"].mean() if len(conv) else np.nan)
        sur.append(conv["cross_surprise"].mean()
                   if len(conv) and conv["cross_surprise"].notna().any()
                   else np.nan)
        # dominance ratio: surprise of A's tokens under B / B's under A
        if len(conv) and conv["cross_surprise"].notna().any():
            s_a = conv[conv["agent"] == 0]["cross_surprise"].mean()
            s_b = conv[conv["agent"] == 1]["cross_surprise"].mean()
            dom.append(s_a / s_b if s_b and not np.isnan(s_b) else np.nan)
        else:
            dom.append(np.nan)
    df["attractor_token_entropy"] = ent
    df["attractor_cross_surprise"] = sur
    df["dominance_ratio_a_over_b"] = dom
    return df


def _joint_tta(s: dict):
    """Joint time-to-attractor: latest of the per-agent times (both must be
    in the basin), None if either never converged."""
    a, b = s.get("agent0_time_to_attractor"), s.get("agent1_time_to_attractor")
    if a is None or b is None:
        return np.nan
    return max(a, b)


def collect_centroids(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        with open(d / "summary.json") as f:
            s = json.load(f)
        for agent in range(2):
            fp = Path(d) / f"centroid_agent{agent}.npy"
            if fp.exists():
                rows.append({
                    "run_id": s["run_id"], "agent": agent,
                    "condition": s["condition"], "topic_id": s["topic_id"],
                    "seed": s["seed"],
                    "model": s["model_a" if agent == 0 else "model_b"],
                    "vector": np.load(fp).astype(np.float32),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def cluster_attractors(centroids: pd.DataFrame, eps: float = 0.05,
                       min_samples: int = 3, var_target: float = 0.90) -> dict:
    """DBSCAN in PCA space, per model (spaces are not comparable across
    models). Also reports PCA variance structure."""
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    out: dict = {}
    for model in sorted(centroids["model"].unique()):
        sub = centroids[centroids["model"] == model]
        X = np.stack(sub["vector"].to_numpy())
        if len(X) < min_samples + 1:
            out[model] = {"n_centroids": len(X),
                          "note": "too few centroids to cluster"}
            continue
        n_comp = int(min(len(X) - 1, X.shape[1], 200))
        pca = PCA(n_components=n_comp, random_state=0)
        Z = pca.fit_transform(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, var_target) + 1)
        Zk = Z[:, :k]
        # normalize so eps is scale-free across models
        scale = np.linalg.norm(Zk, axis=1).mean()
        Zn = Zk / scale if scale > 0 else Zk
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Zn)
        n_clusters = len(set(labels) - {-1})
        sil = None
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = float(silhouette_score(Zn[mask], labels[mask]))
        out[model] = {
            "n_centroids": int(len(X)),
            "pca_components_for_90pct_var": k,
            "top5_variance_ratio": pca.explained_variance_ratio_[:5].tolist(),
            "n_clusters": int(n_clusters),
            "fraction_unclustered": float((labels == -1).mean()),
            "silhouette": sil,
            "labels": labels.tolist(),
            "eps": eps, "min_samples": min_samples,
        }
    return out


def condition_tests(df: pd.DataFrame, metrics: list[str],
                    fdr_q: float = 0.1) -> pd.DataFrame:
    """All pairwise condition comparisons on each metric; BH across the whole
    family."""
    rows = []
    conds = sorted(df["condition"].unique())
    for metric in metrics:
        for c1, c2 in combinations(conds, 2):
            x = df[df["condition"] == c1][metric].astype(float)
            y = df[df["condition"] == c2][metric].astype(float)
            ci_x = bootstrap_ci(x)
            ci_y = bootstrap_ci(y)
            rows.append({
                "metric": metric, "cond_1": c1, "cond_2": c2,
                "n_1": ci_x["n"], "n_2": ci_y["n"],
                "mean_1": ci_x["point"], "ci95_1": (ci_x["lo"], ci_x["hi"]),
                "mean_2": ci_y["point"], "ci95_2": (ci_y["lo"], ci_y["hi"]),
                "cohens_d": cohens_d(x, y),
                "p_raw": mann_whitney(x, y),
            })
    out = pd.DataFrame(rows)
    if len(out):
        bh = benjamini_hochberg(out["p_raw"].tolist(), q=fdr_q)
        out["p_bh_adjusted"] = [b["p_adjusted"] for b in bh]
        out["significant_fdr"] = [b["significant"] for b in bh]
    return out


def attractor_shift(centroids: pd.DataFrame) -> dict:
    """Cosine distance between mean self-play and mean mixed-play centroid,
    per model — does interaction move the attractor?"""
    out = {}
    for model in sorted(centroids["model"].unique()):
        sub = centroids[centroids["model"] == model]
        self_rows = sub[sub["condition"].isin(["self_a", "self_b"])]
        mixed_rows = sub[sub["condition"] == "mixed"]
        if len(self_rows) == 0 or len(mixed_rows) == 0:
            continue
        c_self = np.stack(self_rows["vector"].to_numpy()).mean(axis=0)
        c_mixed = np.stack(mixed_rows["vector"].to_numpy()).mean(axis=0)
        out[model] = {
            "cosine_shift": cosine_distance(c_self, c_mixed),
            "n_self": int(len(self_rows)), "n_mixed": int(len(mixed_rows)),
            "self_dispersion": bootstrap_centroid_ci(
                np.stack(self_rows["vector"].to_numpy())),
            "mixed_dispersion": bootstrap_centroid_ci(
                np.stack(mixed_rows["vector"].to_numpy())),
        }
    return out


def run_permutation_tests(run_dirs: list[Path], window: int, threshold: float,
                          min_turns: int, n_perm: int = 200) -> dict:
    import zarr
    out = {}
    for d in run_dirs:
        entry = {}
        for agent in range(2):
            zp = Path(d) / f"acts_agent{agent}.zarr"
            if not zp.exists():
                continue
            acts = np.asarray(zarr.open(str(zp), mode="r")[:], np.float32)
            entry[f"agent{agent}"] = permutation_test(
                acts, window, threshold, min_turns, n_perm=n_perm)
        out[d.name] = entry
    return out


def collect_recovery_traces(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        with open(d / "summary.json") as f:
            s = json.load(f)
        sb = s.get("springback")
        if not sb:
            continue
        for i, dist in enumerate(sb["trace"]):
            rows.append({"run_id": s["run_id"],
                         "perturb_scale": s.get("perturb_scale", 0.0),
                         "post_turn": i, "dist": dist})
    return pd.DataFrame(rows)


def collect_verbalizations(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for d in run_dirs:
        vp = Path(d) / "verbalizations.json"
        if not vp.exists():
            continue
        with open(vp) as f:
            v = json.load(f)
        with open(d / "summary.json") as f:
            s = json.load(f)
        for agent in range(2):
            entry = v.get(f"agent{agent}")
            if not entry:
                continue
            for phase in ("attractor", "pre_attractor"):
                rec = entry.get(phase)
                if not rec:
                    continue
                rows.append({
                    "run_id": s["run_id"], "agent": agent,
                    "model": entry["model"], "phase": phase,
                    "text": rec.get("patchscope_text"),
                    "perplexity": rec.get("patchscope_perplexity"),
                    "top_tokens": [t for t, _ in
                                   rec.get("logit_lens_top_tokens", [])][:8],
                })
    return rows


# --------------------------------------------------------------------------- #

def analyze(runs_dir: str | Path, out_root: str | Path,
            analysis_id: str | None = None, convergence: dict | None = None,
            dpi: int = 200, n_perm: int = 200, fdr_q: float = 0.1,
            skip_permutation: bool = False) -> Path:
    runs_dir = Path(runs_dir)
    analysis_id = analysis_id or time.strftime("analysis_%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / analysis_id
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    convergence = convergence or {"window": 5, "threshold": 0.01,
                                  "min_turns": 10}

    run_dirs = discover_runs(runs_dir)
    if not run_dirs:
        raise SystemExit(f"No completed runs found under {runs_dir}")
    print(f"[analyze] {len(run_dirs)} completed runs -> {out_dir}")

    df = build_run_table(run_dirs)
    df.to_csv(out_dir / "run_table.csv", index=False)

    centroids = collect_centroids(run_dirs)

    metrics_to_test = ["time_to_attractor", "attractor_token_entropy",
                       "attractor_cross_surprise", "basin_depth_turns"]
    tests = condition_tests(df, metrics_to_test, fdr_q=fdr_q)
    tests.to_csv(out_dir / "stats_tests.csv", index=False)

    clusters = cluster_attractors(centroids) if len(centroids) else {}
    with open(out_dir / "clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)

    shift = attractor_shift(centroids) if len(centroids) else {}

    perm = {}
    if not skip_permutation:
        perm = run_permutation_tests(
            run_dirs, convergence["window"], convergence["threshold"],
            convergence["min_turns"], n_perm=n_perm)
        with open(out_dir / "permutation_tests.json", "w") as f:
            json.dump(perm, f, indent=2)

    # figures
    plotting.convergence_heatmap(df, fig_dir, dpi=dpi)
    if len(centroids):
        plotting.attractor_map(centroids, fig_dir, dpi=dpi)
    traces = collect_recovery_traces(run_dirs)
    if len(traces):
        plotting.recovery_curves(
            traces, fig_dir,
            recovery_threshold=2 * convergence["threshold"], dpi=dpi)
    verb_rows = collect_verbalizations(run_dirs)
    if verb_rows:
        plotting.verbalizer_table(verb_rows, fig_dir, dpi=dpi)

    n_conv = int(df["joint_converged"].sum())
    summary = {
        "analysis_id": analysis_id,
        "n_runs": len(df),
        "n_joint_converged": n_conv,
        "convergence_rate": n_conv / len(df),
        "time_to_attractor_ci": bootstrap_ci(
            df["time_to_attractor"].astype(float)),
        "attractor_shift": shift,
        "clusters": {m: {k: v for k, v in c.items() if k != "labels"}
                     for m, c in clusters.items()},
        "n_significant_tests_fdr": (int(tests["significant_fdr"].sum())
                                    if len(tests) else 0),
        "permutation_artifact_flags": [
            rid for rid, e in perm.items()
            for a, r in e.items()
            if r["observed_converged"] and r["null_rate"] > 0.05
        ],
        "convergence_params": convergence,
        "fdr_q": fdr_q,
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("n_runs", "n_joint_converged",
                               "convergence_rate",
                               "n_significant_tests_fdr")}, indent=2))
    return out_dir
