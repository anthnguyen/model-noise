# Multi-Agent Attractor Exploration Suite

Two LLM agents converse from a cipher-like seed string. We capture their
residual streams every turn, detect convergence to attractor states, perturb
the stream to measure basin depth (springback), and verbalize the attractor
centroids (logit lens + patchscope). Headless, resumable, SLURM and RunPod
friendly.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch transformers accelerate bitsandbytes sentencepiece
```

Gemma is gated on the Hub: run `huggingface-cli login` and accept the
license, or swap the model name in your config to an ungated one.

## Running

```bash
python run.py --config configs/mvp/self_a.yaml --gpu 0
```

Generate and run a full sweep:

```bash
python scripts/gen_configs.py --out configs/sweep \
    --seeds 10 --conditions self_a self_b mixed \
    --pert-scales 0 0.1 0.3 0.5 --max-turns 100

./scripts/batch_run.sh configs/sweep 0        # single GPU
./scripts/launch_tmux.sh configs/sweep 0      # same, detached in tmux
sbatch --array=0-N scripts/slurm_array.sbatch configs/sweep   # SLURM

python scripts/status.py       # progress
python analyze.py              # stats + figures
```

Every run is resumable: rerunning the same config picks up from its last
checkpoint, and batch scripts skip runs that already completed.

## Run directory layout

```
outputs/runs/<run_id>/
├── config.yaml           frozen resolved config
├── manifest.json         status, env, config hash
├── transcript.jsonl      one line per message
├── acts_agent{0,1}.zarr  (rounds, hidden) fp16, memory-mappable
├── metrics.parquet       per-turn entropy, logprob, cross-surprise,
│                         act norm, consecutive cosine distance, etc.
├── centroid_agent{i}.npy attractor centroid (if converged)
├── verbalizations.json   logit-lens tokens + patchscope text and ppl
├── summary.json          run-level results incl. springback trace
├── checkpoint.json       resume state
└── figures/trajectory.{png,pdf,json}
```

## Analysis outputs

```
outputs/analysis/<analysis_id>/
├── run_table.csv, stats_tests.csv, clusters.json,
├── permutation_tests.json, analysis_summary.json
└── figures/  convergence_heatmap, attractor_map, recovery_curves,
              verbalizer_table (+ verbalizer_outputs.{csv,md})
```

## Metrics

Per run: attractor existence and time-to-attractor per agent, joint
convergence, centroid, basin depth, max displacement, token entropy at
attractor, cross-model surprise, dominance ratio. Across runs: attractor
shift between self-play and mixed-play, DBSCAN multistability clustering,
PCA variance structure, permutation-null convergence check, Cohen's d and
Mann-Whitney with Benjamini-Hochberg FDR, bootstrap 95% CIs.

## Design notes

- Capture and perturbation use plain HF forward hooks (`output_hidden_states`
  plus a hook on the capture layer) instead of nnsight or vLLM. Same
  measurements, fewer dependencies, and the capture pass shares logits with
  the entropy and surprise metrics.
- Default models are instruct variants (Qwen2.5-7B-Instruct, gemma-2-9b-it)
  since base models don't hold multi-turn conversations.
- Mixed-play centroids are never embedded in a shared space since different
  models have different hidden sizes; the attractor map and clustering are
  faceted per model.
- Capture layer defaults to `int(0.75 * num_layers)`, overridable per agent.
- Convergence: mean consecutive cosine distance over a 5-turn window below
  0.01 for 5 consecutive windows.
