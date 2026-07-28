"""The agent-vs-agent conversation loop.

Turn indexing: a "round" is one message from agent A then one from agent B.
`turn` below indexes rounds (0..max_turns-1); each agent produces exactly one
activation per round, stored at acts_agent{i}[turn].

Determinism: the sampling seed for every generation is
    run.seed * 1_000_003 + turn * 2 + agent
so a resumed run regenerates the exact same continuation it would have
produced uninterrupted.

Crash safety: transcript lines are fsynced per message; zarr rows are written
per message; metrics + checkpoint flush every `save_every_turns` rounds and on
SIGINT/SIGTERM.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from .attractor import (ConvergenceState, attractor_centroid, cosine_distance,
                        joint_converged, measure_springback)
from .backends import Backend, Perturbation, load_backends
from .config import Config
from .storage import RunStore


class GracefulExit(Exception):
    pass


def _install_signal_handlers():
    def handler(signum, _frame):
        raise GracefulExit(f"signal {signum}")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _perspective(transcript: list[dict], agent: int, framing: str,
                 seed_string: str, history_window: Optional[int]) -> list[dict]:
    """Build the chat-message view for `agent`: own messages are 'assistant',
    partner messages are 'user'. The system prompt carries the framing; the
    seed string arrives as the opening user message."""
    messages = [{"role": "system", "content": framing}]
    body = [{"role": "assistant" if m["agent"] == agent else "user",
             "content": m["text"]} for m in transcript]
    if not body or body[0]["role"] == "assistant":
        body.insert(0, {"role": "user", "content": seed_string})
    else:
        body[0] = {"role": "user",
                   "content": seed_string + "\n\n" + body[0]["content"]}
    if history_window is not None and len(body) > history_window:
        head, tail = body[:1], body[-(history_window - 1):]
        # never let the trimmed view start with an assistant message
        if tail and tail[0]["role"] == "assistant":
            tail = tail[1:]
        body = head + tail
    return messages + body


class ConversationRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backends: tuple[Backend, Backend] = load_backends(cfg)
        hs = (self.backends[0].hidden_size, self.backends[1].hidden_size)
        self.store = RunStore(
            cfg.run_dir, cfg["conversation"]["max_turns"], hs,
            cfg.dump, cfg.config_hash(),
        )
        conv = cfg["convergence"]
        self.states = [
            ConvergenceState(conv["window"], conv["threshold"], conv["min_turns"])
            for _ in range(2)
        ]
        self.pert_cfg = cfg["perturbation"]
        self.centroids: list[Optional[np.ndarray]] = [None, None]
        self.pert_state = {
            "injected_at": None,       # round index of first injection
            "remaining": self.pert_cfg["duration_turns"],
            "patience_left": self.pert_cfg["patience"],
            "direction": None,
        }
        self._wandb = None
        if cfg.get("wandb.enabled"):
            try:
                import wandb
                self._wandb = wandb.init(
                    project=cfg.get("wandb.project", "attractors"),
                    name=cfg.run_id, config=cfg.raw, resume="allow",
                    id=cfg.run_id.replace("/", "_"),
                )
            except Exception as e:  # wandb must never kill a run
                print(f"[warn] wandb init failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #

    def _agent_acts(self, agent: int) -> np.ndarray:
        """All non-NaN activations for an agent so far, ordered by round."""
        arr = self.store.read_activations(agent)
        return arr[~np.isnan(arr).any(axis=1)]

    def _maybe_perturbation(self, turn: int, agent: int) -> Optional[Perturbation]:
        p = self.pert_cfg
        if not p["enabled"] or agent != p["target_agent"]:
            return None
        if self.pert_state["remaining"] <= 0 and self.pert_state["injected_at"] is not None:
            return None

        trigger = False
        if p["turn"] is not None:
            trigger = turn >= p["turn"]
        else:
            # trigger `patience` rounds after joint convergence
            if joint_converged(self.states, self.cfg["convergence"]["converge_on"]):
                if self.pert_state["patience_left"] > 0 and self.pert_state["injected_at"] is None:
                    self.pert_state["patience_left"] -= 1
                else:
                    trigger = True
        if not trigger:
            return None

        if self.pert_state["direction"] is None:
            rng = np.random.default_rng(self.cfg["run"]["seed"] + 424242)
            v = rng.normal(0, 1, self.backends[agent].hidden_size)
            self.pert_state["direction"] = v / np.linalg.norm(v)
        if self.pert_state["injected_at"] is None:
            self.pert_state["injected_at"] = turn
        self.pert_state["remaining"] -= 1
        return Perturbation(vector=self.pert_state["direction"], scale=p["scale"])

    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        cfg = self.cfg
        conv = cfg["conversation"]
        _install_signal_handlers()

        transcript = self.store.load_transcript()
        start_turn = 0
        ckpt = self.store.load_checkpoint()
        if ckpt and cfg["run"]["resume"]:
            start_turn = ckpt["next_turn"]
            saved = ckpt.get("pert_state")
            if saved:
                self.pert_state.update(saved)
                if self.pert_state["direction"] is not None:
                    self.pert_state["direction"] = np.asarray(
                        self.pert_state["direction"])
            # rebuild convergence state from stored activations
            for agent in range(2):
                acts = self._agent_acts(agent)
                for t in range(1, len(acts) + 1):
                    self.states[agent].update(acts[:t])
            if start_turn > 0:
                print(f"[resume] {cfg.run_id} from round {start_turn}")

        status = "completed"
        t0 = time.time()
        turn = start_turn - 1
        try:
            pbar = tqdm(range(start_turn, conv["max_turns"]),
                        desc=cfg.run_id, initial=start_turn,
                        total=conv["max_turns"])
            for turn in pbar:
                for agent in (0, 1):
                    backend = self.backends[agent]
                    messages = _perspective(
                        transcript, agent, conv["framing"],
                        conv["seed_string"], conv["history_window"])
                    seed = cfg["run"]["seed"] * 1_000_003 + turn * 2 + agent
                    perturb = self._maybe_perturbation(turn, agent)
                    out = backend.generate_turn(
                        messages, conv["generation"], conv["max_new_tokens"],
                        seed, perturb)

                    # cross-model surprise: partner scores these tokens from
                    # its own perspective of the conversation so far
                    surprise = None
                    if cfg["metrics"]["compute_cross_surprise"]:
                        partner = self.backends[1 - agent]
                        partner_view = _perspective(
                            transcript, 1 - agent, conv["framing"],
                            conv["seed_string"], conv["history_window"])
                        try:
                            surprise = -partner.score_text(partner_view, out.text)
                        except Exception as e:
                            print(f"[warn] surprise failed t{turn}a{agent}: {e}",
                                  file=sys.stderr)

                    msg = {"turn": turn, "agent": agent, "text": out.text,
                           "n_tokens": out.n_tokens,
                           "perturbed": perturb is not None,
                           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    transcript.append(msg)
                    self.store.append_message(msg)
                    self.store.write_activation(turn, agent, out.activation)

                    acts = self._agent_acts(agent)
                    prev_dist = (cosine_distance(acts[-2], acts[-1])
                                 if len(acts) >= 2 else np.nan)
                    was_converged = self.states[agent].converged_at is not None
                    self.states[agent].update(acts)
                    if (self.states[agent].converged_at is not None
                            and not was_converged):
                        self.centroids[agent] = attractor_centroid(
                            acts, self.states[agent].converged_at,
                            cfg["convergence"]["window"])

                    dist_centroid = (
                        cosine_distance(out.activation, self.centroids[agent])
                        if self.centroids[agent] is not None else np.nan)

                    row = {
                        "turn": turn, "agent": agent,
                        "n_tokens": out.n_tokens,
                        "mean_token_entropy": out.mean_token_entropy,
                        "mean_token_logprob": out.mean_token_logprob,
                        "cross_surprise": surprise,
                        "act_norm": float(np.linalg.norm(out.activation)),
                        "cos_dist_prev": prev_dist,
                        "dist_to_centroid": dist_centroid,
                        "converged": self.states[agent].converged_at is not None,
                        "perturbed": perturb is not None,
                    }
                    self.store.add_metrics_row(row)
                    if self._wandb:
                        self._wandb.log(
                            {f"a{agent}/{k}": v for k, v in row.items()
                             if isinstance(v, (int, float)) and v is not None},
                            step=turn * 2 + agent)

                if (turn + 1) % cfg["run"]["save_every_turns"] == 0:
                    self._checkpoint(turn + 1)

                pbar.set_postfix(
                    conv="".join(
                        "Y" if s.converged_at is not None else "n"
                        for s in self.states),
                    pert=self.pert_state["injected_at"] is not None)
        except GracefulExit as e:
            status = "interrupted"
            print(f"\n[interrupt] {e} — state saved, rerun to resume.")
        except Exception:
            status = "failed"
            self._checkpoint(max(turn, start_turn))
            self.store.finalize(status="failed")
            raise
        finally:
            if status != "failed":
                last = turn + 1 if status == "completed" else max(turn, start_turn)
                self._checkpoint(last)

        summary = self._summarize(status, time.time() - t0)
        self.store.write_summary(summary)
        self.store.finalize(status=status)
        if self._wandb:
            self._wandb.summary.update(
                {k: v for k, v in summary.items()
                 if isinstance(v, (int, float, bool, str))})
            self._wandb.finish()
        return summary

    def _checkpoint(self, next_turn: int) -> None:
        self.store.flush_metrics()
        ps = dict(self.pert_state)
        if ps["direction"] is not None:
            ps["direction"] = np.asarray(ps["direction"]).tolist()
        self.store.save_checkpoint(next_turn, extra={"pert_state": ps})

    # ------------------------------------------------------------------ #

    def _summarize(self, status: str, wall_s: float) -> dict:
        cfg = self.cfg
        summary: dict = {
            "run_id": cfg.run_id,
            "status": status,
            "condition": cfg["condition"],
            "topic_id": cfg["topic_id"],
            "seed": cfg["run"]["seed"],
            "seed_string": cfg["conversation"]["seed_string"],
            "model_a": cfg["models"]["agent_a"]["name"],
            "model_b": cfg["models"]["agent_b"]["name"],
            "backend": cfg["backend"],
            "perturb_enabled": cfg["perturbation"]["enabled"],
            "perturb_scale": (cfg["perturbation"]["scale"]
                              if cfg["perturbation"]["enabled"] else 0.0),
            "wall_seconds": round(wall_s, 1),
            "config_hash": cfg.config_hash(),
        }
        for agent in range(2):
            st = self.states[agent]
            acts = self._agent_acts(agent)
            key = f"agent{agent}"
            summary[f"{key}_converged"] = st.converged_at is not None
            summary[f"{key}_time_to_attractor"] = st.converged_at
            if self.centroids[agent] is not None:
                np.save(self.store.run_dir / f"centroid_agent{agent}.npy",
                        self.centroids[agent])
                summary[f"{key}_centroid_file"] = f"centroid_agent{agent}.npy"
            if (self.pert_state["injected_at"] is not None
                    and self.centroids[agent] is not None
                    and agent == self.pert_cfg["target_agent"]):
                sb = measure_springback(
                    acts, self.centroids[agent],
                    self.pert_state["injected_at"],
                    recovery_threshold=2 * cfg["convergence"]["threshold"])
                summary["springback"] = sb.to_dict()
                summary["basin_depth_turns"] = sb.recovery_turns
        summary["perturb_injected_at"] = self.pert_state["injected_at"]
        summary["joint_converged"] = joint_converged(
            self.states, cfg["convergence"]["converge_on"])
        return summary
