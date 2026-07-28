"""Model backends.

Two implementations of the same interface:

- HFBackend: transformers + bitsandbytes 4-bit. Residual-stream capture is a
  forward pass with output_hidden_states=True over prompt+response (also yields
  logprobs/entropy in the same pass). Perturbation is a forward hook on the
  decoder layer that adds a fixed random direction to the last-token position
  at every generation step of the perturbed turn.

  Note: the plan mentioned nnsight; plain HF hooks are used instead because
  they work identically under bitsandbytes 4-bit, add no extra dependency on
  the cluster, and every capture is paired with the exact logits used for the
  entropy/surprise metrics.

- MockBackend: no torch. Text is templated; "activations" follow a contraction
  map toward a seed-derived fixed point, so convergence genuinely happens,
  perturbation genuinely displaces the state, and springback is real. This
  exercises every downstream code path (detection, zarr, metrics, figures)
  on a laptop.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TurnOutput:
    text: str
    n_tokens: int
    activation: np.ndarray          # (hidden,) residual stream, capture layer, last token
    mean_token_entropy: float       # nats/token over generated tokens
    mean_token_logprob: float       # generator's own mean logprob (nats/token)


@dataclass
class Perturbation:
    vector: np.ndarray              # unit direction, (hidden,)
    scale: float                    # multiplied by running activation norm


class Backend:
    """One loaded model. May serve both agents in self-play."""

    name: str
    hidden_size: int
    num_layers: int
    capture_layer: int

    def generate_turn(self, messages: list[dict], gen_params: dict,
                      max_new_tokens: int, seed: int,
                      perturb: Optional[Perturbation] = None) -> TurnOutput:
        raise NotImplementedError

    def score_text(self, messages: list[dict], response_text: str) -> float:
        """Mean logprob (nats/token) of response_text as this model's next
        assistant message given `messages`. Used for cross-model surprise."""
        raise NotImplementedError

    def verbalize_vector(self, vector: np.ndarray, max_new_tokens: int,
                         seed: int) -> tuple[str, float]:
        """Patchscope-style: inject vector at a placeholder token, generate a
        description. Returns (text, perplexity of that text under the model)."""
        raise NotImplementedError

    def logit_lens(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Project vector through final norm + unembedding; top-k (token, prob)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #

_MOCK_VOCAB = (
    "cipher key lattice glyph residue spiral echo braid knot mirror prime "
    "vault thread sigil orbit fold pulse rune weave anchor drift loom hollow"
).split()


class MockBackend(Backend):
    """Deterministic pseudo-model for dry runs. hidden_size=64.

    Dynamics: h_{t+1} = h* + rho * (h_t - h*) + noise, with the fixed point h*
    derived from (model name, seed string). rho=0.62 gives convergence around
    turn ~15-25 under threshold 0.05; perturbation kicks the state and the
    contraction pulls it back over a few turns (measurable springback).
    """

    def __init__(self, name: str, seed_string: str, master_seed: int):
        self.name = name
        self.hidden_size = 64
        self.num_layers = 12
        self.capture_layer = 9
        digest = hashlib.sha256(f"{name}|{seed_string}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        self.fixed_point = rng.normal(0, 1.0, self.hidden_size).astype(np.float64)
        self.fixed_point *= 8.0 / np.linalg.norm(self.fixed_point)
        self.rho = 0.62
        self.noise_sigma = 0.02
        # per-agent-instance state keyed by who is calling; the conversation
        # runner passes distinct seeds per (turn, agent) so we key off history
        self._state: dict[int, np.ndarray] = {}
        self._master_seed = master_seed

    def _step(self, state_key: int, seed: int,
              perturb: Optional[Perturbation]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h = self._state.get(state_key)
        if h is None:
            h = self.fixed_point + rng.normal(0, 4.0, self.hidden_size)
        h = self.fixed_point + self.rho * (h - self.fixed_point) \
            + rng.normal(0, self.noise_sigma, self.hidden_size)
        if perturb is not None:
            h = h + perturb.scale * np.linalg.norm(h) * perturb.vector
        self._state[state_key] = h
        return h.copy()

    def generate_turn(self, messages, gen_params, max_new_tokens, seed,
                      perturb=None):
        # state key 0/1 by parity of message count => distinct per-agent state
        state_key = len(messages) % 2
        h = self._step(state_key, seed, perturb)
        rng = np.random.default_rng(seed + 7)
        # entropy decays as the state approaches the fixed point
        dist = float(np.linalg.norm(h - self.fixed_point))
        ent = 0.5 + 2.5 * (1 - math.exp(-dist / 4.0)) + rng.normal(0, 0.05)
        n_words = int(rng.integers(20, 40))
        words = rng.choice(_MOCK_VOCAB, size=n_words)
        text = " ".join(words.tolist()).capitalize() + "."
        return TurnOutput(
            text=text,
            n_tokens=n_words,
            activation=h.astype(np.float32),
            mean_token_entropy=max(ent, 0.05),
            mean_token_logprob=-max(ent, 0.05),
        )

    def score_text(self, messages, response_text):
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(response_text.encode()).digest()[:4], "big")
        )
        return float(-2.0 + rng.normal(0, 0.3))

    def verbalize_vector(self, vector, max_new_tokens, seed):
        rng = np.random.default_rng(seed)
        idx = np.argsort(-np.abs(vector))[:6] % len(_MOCK_VOCAB)
        text = "Themes of " + ", ".join(_MOCK_VOCAB[i] for i in idx) + "."
        return text, float(np.exp(2.0 + rng.normal(0, 0.2)))

    def logit_lens(self, vector, top_k):
        idx = np.argsort(-np.abs(vector))[:top_k] % len(_MOCK_VOCAB)
        w = np.abs(vector)[np.argsort(-np.abs(vector))[:top_k]]
        p = w / w.sum()
        return [(_MOCK_VOCAB[i], float(pi)) for i, pi in zip(idx, p)]


# --------------------------------------------------------------------------- #
# HF backend
# --------------------------------------------------------------------------- #

class HFBackend(Backend):
    def __init__(self, name: str, quantization: str = "4bit",
                 device: str = "cuda:0", capture_layer: Optional[int] = None,
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = getattr(torch, dtype)
        kwargs: dict = {"torch_dtype": torch_dtype, "device_map": device}
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantization == "4bit"),
                load_in_8bit=(quantization == "8bit"),
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model.eval()

        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.capture_layer = (
            capture_layer if capture_layer is not None
            else int(0.75 * self.num_layers)
        )
        self._decoder_layers = self.model.model.layers

    # -- internals ---------------------------------------------------------- #

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _perturb_hook(self, perturb: Perturbation):
        torch = self.torch
        direction = torch.tensor(
            perturb.vector, dtype=self.model.dtype, device=self.device
        )

        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            norm = hs[:, -1, :].norm(dim=-1, keepdim=True)
            hs[:, -1, :] = hs[:, -1, :] + perturb.scale * norm * direction
            return output

        return self._decoder_layers[self.capture_layer].register_forward_hook(hook)

    @staticmethod
    def _entropy_from_logits(torch, logits):
        logp = torch.log_softmax(logits.float(), dim=-1)
        return -(logp.exp() * logp).sum(-1)

    # -- interface ---------------------------------------------------------- #

    def generate_turn(self, messages, gen_params, max_new_tokens, seed,
                      perturb=None):
        torch = self.torch
        torch.manual_seed(seed)
        prompt = self._render(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # Capture via a hook on just the target layer instead of
        # output_hidden_states=True, which would materialize every layer's
        # hidden states over the full sequence (an O(num_layers) memory
        # multiplier that grows with conversation length since history is
        # unbounded by default).
        captured: dict = {}

        def capture_hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["act"] = hs[0, -1, :].detach().clone()
            return output

        cap_handle = self._decoder_layers[self.capture_layer].register_forward_hook(
            capture_hook)
        handle = self._perturb_hook(perturb) if perturb is not None else None
        try:
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=gen_params.get("temperature", 0.7) > 0,
                    temperature=gen_params.get("temperature", 0.7),
                    top_p=gen_params.get("top_p", 0.9),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Scoring/capture pass over prompt+response. The perturbation hook
            # stays active here (if any) so the captured activation reflects
            # the same intervention the generation saw.
            with torch.no_grad():
                fwd = self.model(out_ids)
        finally:
            cap_handle.remove()
            if handle is not None:
                handle.remove()

        resp_ids = out_ids[0, prompt_len:]
        n_new = resp_ids.shape[0]
        act = captured["act"].float().cpu().numpy()

        logits = fwd.logits[0, prompt_len - 1:-1, :]  # predicting resp tokens
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1)
        ent = self._entropy_from_logits(torch, logits)

        text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        return TurnOutput(
            text=text,
            n_tokens=int(n_new),
            activation=act,
            mean_token_entropy=float(ent.mean()) if n_new else 0.0,
            mean_token_logprob=float(tok_logp.mean()) if n_new else 0.0,
        )

    def score_text(self, messages, response_text):
        torch = self.torch
        prompt = self._render(messages, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        resp_ids = self.tokenizer(
            response_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        full = torch.cat([prompt_ids, resp_ids], dim=1).to(self.device)
        with torch.no_grad():
            logits = self.model(full).logits
        n_resp = resp_ids.shape[1]
        pred = logits[0, -n_resp - 1:-1, :]
        logp = torch.log_softmax(pred.float(), dim=-1)
        tok = logp.gather(-1, full[0, -n_resp:].unsqueeze(-1).to(logp.device))
        return float(tok.mean())

    def verbalize_vector(self, vector, max_new_tokens, seed):
        torch = self.torch
        torch.manual_seed(seed)
        carrier = 'Tell me everything about the concept "X".\n\nThe concept "X" means'
        ids = self.tokenizer(carrier, return_tensors="pt").to(self.device)
        # position of the *second* X occurrence (in the continuation clause)
        x_id = self.tokenizer.encode("X", add_special_tokens=False)[-1]
        positions = (ids.input_ids[0] == x_id).nonzero().flatten()
        pos = int(positions[-1]) if len(positions) else ids.input_ids.shape[1] - 1

        vec = torch.tensor(vector, dtype=self.model.dtype, device=self.device)

        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > pos:  # only on the prefill pass
                hs[:, pos, :] = vec
            return output

        handle = self._decoder_layers[self.capture_layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            handle.remove()
        text = self.tokenizer.decode(
            out[0, ids.input_ids.shape[1]:], skip_special_tokens=True
        )
        # coherence proxy: perplexity of the verbalization under the model
        with torch.no_grad():
            full = out
            logits = self.model(full).logits
        n = full.shape[1] - ids.input_ids.shape[1]
        if n <= 0:
            return text, float("nan")
        logp = torch.log_softmax(logits[0, -n - 1:-1, :].float(), dim=-1)
        tok = logp.gather(-1, full[0, -n:].unsqueeze(-1))
        ppl = float(torch.exp(-tok.mean()))
        return text, ppl

    def logit_lens(self, vector, top_k):
        torch = self.torch
        vec = torch.tensor(vector, dtype=self.model.dtype, device=self.device)
        normed = self.model.model.norm(vec)
        logits = self.model.lm_head(normed)
        probs = torch.softmax(logits.float(), dim=-1)
        top = probs.topk(top_k)
        return [
            (self.tokenizer.decode([int(i)]), float(p))
            for i, p in zip(top.indices, top.values)
        ]


# --------------------------------------------------------------------------- #

def load_backends(cfg) -> tuple[Backend, Backend]:
    """Returns (backend_a, backend_b); shares one instance when identical."""
    a_cfg, b_cfg = cfg["models"]["agent_a"], cfg["models"]["agent_b"]
    if cfg["backend"] == "mock":
        seed_string = cfg["conversation"]["seed_string"]
        master = cfg["run"]["seed"]
        a = MockBackend(a_cfg["name"], seed_string, master)
        b = a if a_cfg["name"] == b_cfg["name"] else MockBackend(
            b_cfg["name"], seed_string, master)
        return a, b
    a = HFBackend(a_cfg["name"], a_cfg["quantization"], a_cfg["device"],
                  a_cfg["capture_layer"], a_cfg["dtype"])
    if (a_cfg["name"] == b_cfg["name"]
            and a_cfg["quantization"] == b_cfg["quantization"]
            and a_cfg["device"] == b_cfg["device"]):
        return a, a
    b = HFBackend(b_cfg["name"], b_cfg["quantization"], b_cfg["device"],
                  b_cfg["capture_layer"], b_cfg["dtype"])
    return a, b
