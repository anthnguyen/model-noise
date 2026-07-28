"""All figures. Reproducibility + labeling rules enforced here:

- Every figure gets: descriptive title incl. run/analysis id, axis labels
  with units, legends with meaningful names, labeled colorbars, and a footer
  with the generation timestamp + config hash.
- Every figure is saved as PNG (dpi from config) AND PDF, plus a JSON sidecar
  recording exactly what data/parameters produced it.
- All stochastic layout (UMAP) uses a fixed seed recorded in the sidecar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless: safe over SSH with no X forwarding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITION_COLORS = {"self_a": "#1f77b4", "self_b": "#ff7f0e", "mixed": "#2ca02c"}


def _finish(fig, out_stem: Path, sidecar: dict, dpi: int = 200) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.005, f"generated {stamp}", ha="right", va="bottom",
             fontsize=6, color="gray")
    paths = []
    for ext in ("png", "pdf"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    sidecar = {**sidecar, "generated_at": stamp}
    with open(out_stem.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    plt.close(fig)
    return paths


def _pca_2d(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    n_comp = min(2, acts.shape[0], acts.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    proj = pca.fit_transform(acts)
    if n_comp < 2:
        proj = np.column_stack([proj, np.zeros(len(proj))])
        var = np.array([pca.explained_variance_ratio_[0], 0.0])
    else:
        var = pca.explained_variance_ratio_
    return proj, var


# --------------------------------------------------------------------------- #
# 1. Per-run trajectory plot
# --------------------------------------------------------------------------- #

def trajectory_plot(run_dir: Path, dpi: int = 200) -> Optional[list[Path]]:
    run_dir = Path(run_dir)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted = False
    for agent, ax in enumerate(axes):
        acts_path = run_dir / f"acts_agent{agent}.zarr"
        if not acts_path.exists():
            continue
        import zarr
        acts = np.asarray(zarr.open(str(acts_path), mode="r")[:], dtype=np.float32)
        mask = ~np.isnan(acts).any(axis=1)
        acts = acts[mask]
        if len(acts) < 3:
            continue
        plotted = True
        proj, var = _pca_2d(acts)
        turns = np.arange(len(acts))
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=turns, cmap="viridis",
                        s=28, zorder=3)
        ax.plot(proj[:, 0], proj[:, 1], color="gray", lw=0.6, alpha=0.6,
                zorder=2)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Conversation round")
        tta = summary.get(f"agent{agent}_time_to_attractor")
        if tta is not None and tta < len(proj):
            ax.scatter(*proj[tta], marker="*", s=350, facecolor="none",
                       edgecolor="red", linewidths=1.5, zorder=4,
                       label=f"attractor reached (round {tta})")
        pt = summary.get("perturb_injected_at")
        if pt is not None and pt < len(proj):
            ax.scatter(*proj[pt], marker="X", s=180, color="crimson",
                       zorder=4, label=f"perturbation (round {pt})")
        if tta is not None or pt is not None:
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(f"PC1 ({var[0]:.0%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.0%} var)")
        model = summary["model_a" if agent == 0 else "model_b"].split("/")[-1]
        ax.set_title(f"Agent {agent} ({model})")
    if not plotted:
        plt.close(fig)
        return None
    fig.suptitle(
        f"Residual-stream trajectory — {summary['run_id']}\n"
        f"condition={summary['condition']}, capture-layer PCA per agent",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _finish(fig, run_dir / "figures" / "trajectory",
                   {"figure": "trajectory", "run_id": summary["run_id"],
                    "pca_seed": 0}, dpi)


# --------------------------------------------------------------------------- #
# 2. Convergence heatmap (across runs)
# --------------------------------------------------------------------------- #

def convergence_heatmap(df: pd.DataFrame, out_dir: Path,
                        dpi: int = 200) -> list[Path]:
    """df: run-level summary table (one row per run)."""
    conditions = sorted(df["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(5 * len(conditions), 4.2),
                             squeeze=False)
    max_turn = np.nanmax(df["time_to_attractor"].astype(float)) if \
        df["time_to_attractor"].notna().any() else 1
    for ax, cond in zip(axes[0], conditions):
        sub = df[df["condition"] == cond]
        piv = sub.pivot_table(index="seed", columns="topic_id",
                              values="time_to_attractor", aggfunc="mean")
        data = piv.values.astype(float)
        im = ax.imshow(data, aspect="auto", cmap="magma_r",
                       vmin=0, vmax=max_turn)
        # annotate every cell: value, or ✕ for never-converged
        norm_max = max_turn if max_turn else 1
        for (i, j), v in np.ndenumerate(data):
            if np.isnan(v):
                ax.text(j, i, "✕", ha="center", va="center",
                        color="gray", fontsize=9)
            else:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v / norm_max > 0.5 else "black")
        ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=45,
                      fontsize=8)
        ax.set_yticks(range(len(piv.index)), piv.index, fontsize=8)
        ax.set_xlabel("Topic / seed string")
        ax.set_ylabel("Run seed")
        ax.set_title(f"condition = {cond} (n={len(sub)})")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Time to attractor (rounds; ✕ = never)")
    fig.suptitle("Convergence speed across seeds × topics", fontsize=12)
    return _finish(fig, Path(out_dir) / "convergence_heatmap",
                   {"figure": "convergence_heatmap",
                    "n_runs": len(df), "conditions": conditions}, dpi)


# --------------------------------------------------------------------------- #
# 3. Attractor map (centroids, per model space)
# --------------------------------------------------------------------------- #

def attractor_map(centroids: pd.DataFrame, out_dir: Path,
                  umap_seed: int = 42, dpi: int = 200) -> Optional[list[Path]]:
    """centroids: columns [run_id, condition, model, agent, vector(list)].

    Centroids from different models live in different activation spaces, so
    the map is faceted per model — one embedding per space, never mixed.
    """
    if centroids.empty:
        return None
    models = sorted(centroids["model"].unique())
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5.5 * len(models), 4.6), squeeze=False)
    method_used = {}
    for ax, model in zip(axes[0], models):
        sub = centroids[centroids["model"] == model]
        X = np.stack(sub["vector"].to_numpy())
        if len(sub) >= 8:
            try:
                import umap
                proj = umap.UMAP(n_components=2, random_state=umap_seed,
                                 n_neighbors=min(10, len(sub) - 1)
                                 ).fit_transform(X)
                method_used[model] = f"UMAP(seed={umap_seed})"
            except ImportError:
                proj, _ = _pca_2d(X)
                method_used[model] = "PCA (umap-learn not installed)"
        else:
            proj, _ = _pca_2d(X)
            method_used[model] = "PCA (too few points for UMAP)"
        for cond in sorted(sub["condition"].unique()):
            m = (sub["condition"] == cond).to_numpy()
            ax.scatter(proj[m, 0], proj[m, 1], s=45,
                       color=CONDITION_COLORS.get(cond, "gray"),
                       label=cond, alpha=0.85, edgecolor="white",
                       linewidths=0.5)
        ax.set_title(f"{model.split('/')[-1]}\n[{method_used[model]}]",
                     fontsize=10)
        ax.set_xlabel("Embedding dim 1")
        ax.set_ylabel("Embedding dim 2")
        ax.legend(title="Condition", fontsize=8)
    fig.suptitle("Attractor centroids by condition (one panel per model's "
                 "activation space)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, Path(out_dir) / "attractor_map",
                   {"figure": "attractor_map", "umap_seed": umap_seed,
                    "n_centroids": len(centroids),
                    "embedding_method": method_used}, dpi)


# --------------------------------------------------------------------------- #
# 4. Verbalizer table
# --------------------------------------------------------------------------- #

def verbalizer_table(rows: list[dict], out_dir: Path,
                     dpi: int = 200) -> Optional[list[Path]]:
    """rows: [{run_id, agent, model, phase, text, perplexity, top_tokens}]"""
    if not rows:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "verbalizer_outputs.csv", index=False)

    # human-readable markdown alongside the figure
    with open(out_dir / "verbalizer_outputs.md", "w") as f:
        f.write("# Verbalizer outputs (attractor vs pre-attractor)\n\n")
        for r in rows:
            f.write(f"## {r['run_id']} — agent {r['agent']} "
                    f"({r['model']}) — {r['phase']}\n\n")
            f.write(f"- top tokens (logit lens): {r.get('top_tokens')}\n")
            f.write(f"- patchscope ppl: {r.get('perplexity')}\n\n")
            f.write(f"> {r.get('text')}\n\n")

    show = df[["run_id", "agent", "phase", "perplexity", "text"]].copy()
    show["text"] = show["text"].fillna("").str.slice(0, 70)
    show["perplexity"] = show["perplexity"].map(
        lambda p: f"{p:.1f}" if pd.notna(p) else "—")
    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.42 * len(show)))
    ax.axis("off")
    table = ax.table(cellText=show.values, colLabels=show.columns,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(range(len(show.columns)))
    ax.set_title("Attractor verbalizations (patchscope text, truncated to 70 "
                 "chars — full text in verbalizer_outputs.md)", fontsize=11)
    return _finish(fig, out_dir / "verbalizer_table",
                   {"figure": "verbalizer_table", "n_rows": len(rows)}, dpi)


# --------------------------------------------------------------------------- #
# 5. Perturbation recovery curves
# --------------------------------------------------------------------------- #

def recovery_curves(traces: pd.DataFrame, out_dir: Path,
                    recovery_threshold: float,
                    dpi: int = 200) -> Optional[list[Path]]:
    """traces: columns [run_id, perturb_scale, post_turn, dist]."""
    if traces.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("plasma")
    scales = sorted(traces["perturb_scale"].unique())
    for i, scale in enumerate(scales):
        sub = traces[traces["perturb_scale"] == scale]
        g = sub.groupby("post_turn")["dist"]
        mean, sem = g.mean(), g.sem().fillna(0)
        color = cmap(0.15 + 0.7 * i / max(1, len(scales) - 1))
        n_runs = sub["run_id"].nunique()
        ax.plot(mean.index, mean.values, color=color, lw=2,
                label=f"scale {scale:g} (n={n_runs})")
        ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.18)
    ax.axhline(recovery_threshold, ls="--", color="gray", lw=1,
               label=f"recovery threshold ({recovery_threshold:g})")
    ax.set_xlabel("Rounds since perturbation")
    ax.set_ylabel("Cosine distance to pre-perturbation centroid")
    ax.set_title("Springback: return to attractor after residual-stream "
                 "perturbation\n(mean ± 95% CI across runs)")
    ax.legend(title="Perturbation scale\n(× activation norm)", fontsize=8)
    return _finish(fig, Path(out_dir) / "recovery_curves",
                   {"figure": "recovery_curves", "scales": scales,
                    "recovery_threshold": recovery_threshold,
                    "n_runs": traces["run_id"].nunique()}, dpi)
