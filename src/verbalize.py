"""Verbalize attractor centroids.

Two methods, both run against the model that produced the centroid:

- logit_lens: project the centroid through the final norm + unembedding and
  report top-k tokens. Cheap, deterministic, a good first look.
- patchscope: inject the centroid at a placeholder token in a carrier prompt
  and let the model describe it in natural language (Patchscopes-style).
  Also reports the perplexity of the generated description (AV coherence:
  compare attractor centroid vs pre-attractor baseline vectors).
"""

from __future__ import annotations

import numpy as np

from .backends import Backend


def verbalize_run(cfg, backends: tuple[Backend, Backend],
                  centroids: list, pre_attractor: list) -> dict:
    """centroids / pre_attractor: per-agent vectors (or None).

    pre_attractor is a baseline vector from early in the run (mean of the
    first `window` turns) so coherence deltas have a reference point.
    """
    vcfg = cfg["verbalize"]
    out: dict = {"methods": vcfg["methods"]}
    for agent in range(2):
        backend = backends[agent]
        entry: dict = {"model": backend.name}
        for label, vec in (("attractor", centroids[agent]),
                           ("pre_attractor", pre_attractor[agent])):
            if vec is None:
                entry[label] = None
                continue
            vec = np.asarray(vec, dtype=np.float32)
            rec: dict = {}
            if "logit_lens" in vcfg["methods"]:
                rec["logit_lens_top_tokens"] = backend.logit_lens(
                    vec, vcfg["top_k_tokens"])
            if "patchscope" in vcfg["methods"]:
                text, ppl = backend.verbalize_vector(
                    vec, vcfg["max_new_tokens"],
                    seed=cfg["run"]["seed"] + 999)
                rec["patchscope_text"] = text
                rec["patchscope_perplexity"] = ppl
            entry[label] = rec
        out[f"agent{agent}"] = entry
    return out
