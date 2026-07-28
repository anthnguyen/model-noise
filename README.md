# Multi-Agent Attractor Exploration Suite

Two LLM agents converse from a cipher-like seed string; we capture their
residual streams every turn, detect convergence to attractor states, perturb
the stream to measure basin depth (springback), and verbalize the attractor
centroids (logit lens + patchscope). Headless, resumable, SLURM-friendly.

## Quick start (laptop, no GPU) — the dry-run path

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python tests/test_smoke.py                    # full pipeline self-test
python run.py --config configs/mock/smoke.yaml
python analyze.py --threshold 0.05 --min-turns 8
```

The `mock` backend has no torch dependency; its fake "activations" follow a
contraction map toward a seed-derived fixed point, so convergence detection,
perturbation, springback, and every figure are exercised for real.

## Quick start (GPU cluster)

```bash
ssh cluster
git clone <this repo> && cd attractors
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch transformers accelerate bitsandbytes sentencepiece

python scripts/preflight.py --gpu --models    # checks CUDA, packages, model access
./scripts/run_experiment.sh configs/mvp/self_a.yaml 0   # MVP run 1/3
```

Gemma is gated on the Hub: `huggingface-cli login` and accept the license, or
swap `configs/mvp/self_b.yaml` to an ungated model.

### RunPod

RunPod pods have no job scheduler and — unless you attach a **Network
Volume** — no storage that survives a stop/terminate. `/workspace` is the
network-volume mount point when one is attached; put the repo there.

```bash
git clone <this repo> /workspace/attractors && cd /workspace/attractors
./scripts/runpod_setup.sh                    # venv + HF cache, both on /workspace
python scripts/preflight.py --gpu --models
./scripts/launch_tmux.sh configs/mvp 0        # survives closing your laptop
```

`runpod_setup.sh` reuses the template's system torch when present (avoids
re-downloading multi-GB CUDA wheels) and points `HF_HOME` at
`/workspace/hf_cache` so model weights survive a pod restart. If you're
running **Spot/interruptible** pods for the price, that's exactly what the
checkpoint system is for — a reclaimed pod just means re-running
`launch_tmux.sh` on a fresh pod once it's back, which resumes every
in-progress run from its last checkpoint and skips everything completed
(same as any other interruption). Point `--config`'s `run.output_dir` (or
`--override run.output_dir=/workspace/attractors/outputs/runs`) at the
volume too if your repo itself isn't already under `/workspace`.

### MVP demo (the "show someone" artifact)

```bash
for c in configs/mvp/*.yaml; do ./scripts/run_experiment.sh "$c" 0; done
python analyze.py --analysis-id mvp_demo
```

### Full sweep

```bash
python scripts/gen_configs.py --out configs/sweep \
    --seeds 10 --conditions self_a self_b mixed \
    --pert-scales 0 0.1 0.3 0.5 --max-turns 100

# single machine, 4 GPUs (one shard per GPU):
for g in 0 1 2 3; do ./scripts/batch_run.sh configs/sweep $g 4 & done; wait

# or SLURM:
N=$(ls configs/sweep/*.yaml | wc -l)
sbatch --array=0-$((N-1))%16 scripts/slurm_array.sbatch configs/sweep

python scripts/status.py            # progress bar; use with watch -n 30
python analyze.py                   # stats + figures when done
```

## Unattended runs (surviving a closed laptop / lost connection)

**Your SSH session dying does not lose work** — checkpointing (see below)
means rerunning the same command always resumes. But the *process* itself
needs to keep running after you disconnect:

- **Cluster has SLURM (preferred):** `sbatch` submits and returns instantly;
  the job runs on compute nodes independent of your session. Close your
  laptop the moment `sbatch` returns — nothing to detach from.
  ```bash
  N=$(ls configs/sweep/*.yaml | wc -l)
  sbatch --array=0-$((N-1))%16 scripts/slurm_array.sbatch configs/sweep
  ```
  Reconnect anytime: `squeue -u $USER` or `python scripts/status.py`.
  A preempted/requeued task resumes from its own checkpoint automatically.

- **Raw GPU box, no scheduler:** run inside `tmux` so the batch survives the
  SSH connection closing.
  ```bash
  ./scripts/launch_tmux.sh configs/sweep 0     # starts tmux session "attractors"
  # Ctrl-b then d to detach — safe to close your laptop now
  tmux attach -t attractors                    # reattach later, from anywhere
  ```
  `mosh` instead of `ssh` is worth using here too — it survives sleep/IP
  changes without killing the tmux session's connection.

Either way, if the job does get killed uncleanly, just rerun the same
`batch_run.sh` / `sbatch` command: completed runs are skipped
(`--skip-completed`) and the interrupted one resumes exactly from its last
checkpoint (`python scripts/status.py -v` shows what's still in progress).

## Crash safety / "insurance"

- **Everything resumes.** Transcript lines are fsynced per message, zarr rows
  written per message, metrics + checkpoint flushed every 5 rounds and on
  SIGINT/SIGTERM. Re-running the same config continues from the checkpoint;
  `batch_run.sh` / the SLURM script skip completed runs, so re-launching a
  crashed sweep is always safe.
- **Resume is exact.** Per-turn sampling seeds are derived as
  `run.seed * 1_000_003 + turn * 2 + agent`, so a resumed run reproduces the
  identical conversation (covered by `test_resume_determinism`).
- **Runs never clobber each other**: run ids encode condition/seed/topic/
  perturbation; analyses are timestamped by default.
- **Best-effort extras**: wandb, verbalization, and per-run figures can fail
  without killing or corrupting a run.

## Reproducibility

- Each run dir contains `config.yaml` (fully resolved — rerun from this alone),
  `manifest.json` (env, package versions, GPU, config hash, status).
- All figures are saved as PNG + PDF **with a JSON sidecar** recording the
  data and parameters (including the UMAP seed) that produced them.
- No hardcoded paths; no unseeded randomness (PCA `random_state=0`,
  UMAP `random_state=42`, all numpy RNGs derive from `run.seed`).

## Run directory layout

```
outputs/runs/<run_id>/
├── config.yaml           frozen resolved config
├── manifest.json         status, env, config hash
├── transcript.jsonl      one line per message
├── acts_agent{0,1}.zarr  (rounds, hidden) fp16, memory-mappable
├── metrics.parquet       per-turn: entropy, logprob, cross-surprise,
│                         act norm, consecutive cosine distance, …
├── centroid_agent{i}.npy attractor centroid (if converged)
├── verbalizations.json   logit-lens tokens + patchscope text & ppl
├── summary.json          run-level results incl. springback trace
├── checkpoint.json       resume state
└── figures/trajectory.{png,pdf,json}
```

## Metrics

Per run (`summary.json`, aggregated into `run_table.csv`): attractor existence
& time-to-attractor per agent, joint convergence, centroid, basin depth
(rounds to recover after perturbation), max displacement. From
`metrics.parquet`: token entropy at attractor, cross-model surprise (partner's
mean −logprob/token of the generator's message), dominance ratio (A→B / B→A
surprise). Across runs (`analyze.py`): attractor shift (self vs mixed centroid
distance per model), DBSCAN multistability clustering + silhouette,
PCA variance structure, permutation-null convergence check, Cohen's d +
Mann-Whitney with Benjamini-Hochberg FDR (q=0.1), bootstrap 95% CIs.

## Analysis outputs

```
outputs/analysis/<analysis_id>/
├── run_table.csv, stats_tests.csv, clusters.json,
├── permutation_tests.json, analysis_summary.json
└── figures/  convergence_heatmap, attractor_map, recovery_curves,
              verbalizer_table (+ verbalizer_outputs.{csv,md})
```

## Design notes & deviations from the original plan

- **HF hooks instead of nnsight/vLLM.** Capture is a forward pass with
  `output_hidden_states=True`; perturbation is a forward hook on the capture
  layer adding `scale × ‖h‖ × unit-vector` to the last token each generation
  step. Same measurements, two fewer fragile dependencies, and the capture
  pass shares logits with the entropy/surprise metrics. vLLM does not expose
  residual streams cleanly, and generation here is not the bottleneck.
- **Instruct variants** (`Qwen2.5-7B-Instruct`, `gemma-2-9b-it`) — base
  models don't hold multi-turn conversations. Swap names in configs if you
  want base-model dynamics.
- **Mixed-play centroids are never embedded in a shared space** — Qwen and
  Gemma have different hidden sizes/geometry, so the attractor map and
  clustering are faceted per model.
- **Capture layer** defaults to `int(0.75 × num_layers)`; override with
  `models.agent_?.capture_layer`.
- Convergence: mean consecutive cosine distance over a 5-turn window < 0.01
  for 5 consecutive windows (per agent; `converge_on: both` by default).
  The 0.01 default is calibrated for real models; mock configs use 0.05.

## Dry-run checklist before burning GPU hours

1. `python tests/test_smoke.py` — must print `ALL SMOKE TESTS PASSED`.
2. `python scripts/gen_configs.py --out configs/sweep_mock --backend mock --seeds 3 --topics t01 t02 --pert-scales 0 0.6 --max-turns 40`
   then `./scripts/batch_run.sh configs/sweep_mock` — rehearses the exact
   batch machinery (skip/resume/logs) you'll use on the cluster.
3. `python analyze.py --threshold 0.05 --min-turns 8` — inspect
   `outputs/analysis/*/figures/`. Note: mock runs *will* show
   `permutation_artifact_flags` — mock trajectories sit at the fixed point
   for most rounds, so shuffled order still "converges". That is the null
   check working, not a bug; judge it on real-model runs.
4. On the cluster: `python scripts/preflight.py --gpu --models`.
5. One real short run:
   `python run.py --config configs/mvp/self_a.yaml --override conversation.max_turns=10`
