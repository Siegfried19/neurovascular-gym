"""Online training loop.

Same schedule as the official TD-MPC2 ``OnlineTrainer``:

1. the first ``seed_steps`` env steps use uniform random actions;
2. when ``seed_steps`` is reached, ``initial_updates`` updates run in one burst;
3. after that, ``updates_per_step`` updates per env step;
4. an episode enters the replay only when it ends;
5. evaluation is due every ``evaluation.interval`` steps and runs at the next
   episode boundary.

Additions for SOFA: a step that raises or flags ``simulation_error`` is
dropped, the episode so far is kept (as truncated: no terminal flag, so its last
value is bootstrapped) and counted (``end: error`` in ``episodes.jsonl``,
``error_rate`` in ``metrics.jsonl``), and a new one starts; a step that flags
``wall_breached`` (a device's centerline outside the vessel mesh, two_device/wall.py)
is treated the same way under its own name (``end: breach``, ``breach_rate``,
``wall_breaches``: the simulator's failure, not the policy's, so no -1; user
decision 2026-11-09); checkpoints hold the full state so a crashed run can resume.

Optional parts, each off unless configured; with all of them off a run is bit for
bit what this loop gave before they existed (tests/test_two_device_training.py):

* ``training.cutoff_mixture``: each episode's cutoff, set with
  ``env.set_cutoff(k)`` before its reset (``curriculum.CutoffMixture``);
* ``planning.residual``: the planner's box around the scripted suggestion in
  phase 1 (``curriculum.ResidualSchedule``);
* ``replay.stratify_by`` and ``training.demos``: see ``replay.py`` / ``demos.py``
  (set up in ``setup.start_run``).

Logs (``RunLogger``): ``episodes.jsonl`` one line per episode, with the task
metrics the env reports (``metrics.EpisodeStats``); ``metrics.jsonl`` one line
per ``logging.print_interval`` steps, averaged over the last ``METRICS_WINDOW``
episodes (reach rates per phase, success, errors, shield, catheter, step rate,
cutoff weights); ``events.jsonl`` checkpoints, dropped steps, worker
replacements, weight updates.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .checkpoint import capture_rng, restore_rng, save_checkpoint
from .curriculum import CutoffMixture, ResidualSchedule
from .evaluate import evaluate
from .logger import MetricAverager, RunLogger
from .metrics import EpisodeStats, progress_line, window_summary
from .replay import EpisodeReplay

METRICS_WINDOW = 100           # episodes the console's reach rates and metrics.jsonl average over
CUTOFF_SEED_OFFSET = 7_919     # the cutoff draws' own random stream: training.seed + this


def step_failure(info: Mapping[str, Any]) -> Optional[str]:
    """Why a step is dropped and its episode closed: ``simulation_error`` (NaN: stEVE put the
    instruments back to the start) or ``wall_breach`` (``info["wall_breached"]``: a device's
    centerline outside the vessel mesh); None for a step to keep."""
    if info.get("simulation_error"):
        return "simulation_error"
    if info.get("wall_breached"):
        return "wall_breach"
    return None


class _Episode:
    def __init__(self, observation: np.ndarray, label_names: List[str],
                 info: Optional[Mapping[str, Any]] = None, cutoff: Optional[int] = None) -> None:
        self.observations = [observation]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminated: List[float] = []
        self.labels: Dict[str, List[float]] = {n: [] for n in label_names}
        self.stats = EpisodeStats(info, cutoff)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def last_info(self) -> Mapping[str, Any]:
        return self.stats.last_info

    def add(self, action, reward, terminated, labels, next_observation,
            info: Optional[Mapping[str, Any]] = None) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.terminated.append(float(terminated))
        for name in self.labels:
            self.labels[name].append(float(labels[name]))
        self.observations.append(next_observation)
        self.stats.add(self.actions[-1], info or {})


class Trainer:
    def __init__(self, config: Mapping[str, Any], env, agent, replay: EpisodeReplay,
                 run_dir: str | Path, *, resume_state: Optional[Mapping[str, Any]] = None,
                 quiet: bool = False) -> None:
        self.cfg = config
        self.env = env
        self.agent = agent
        self.replay = replay
        self.run_dir = Path(run_dir)
        self.logger = RunLogger(self.run_dir, quiet=quiet)
        self.label_names = list(env.label_names)
        self.rng = np.random.default_rng(int(config["training"]["seed"]))
        self.step = 0
        self.episode_index = 0
        self.success_count = 0
        self.env_errors = 0
        self.wall_breaches = 0
        self.updates = MetricAverager()
        self.recent = deque(maxlen=50)
        self.window = deque(maxlen=METRICS_WINDOW)
        self.last_update: Dict[str, float] = {}
        self.mixture = CutoffMixture.from_config(config["training"].get("cutoff_mixture"),
                                                 int(config["training"]["seed"]) + CUTOFF_SEED_OFFSET)
        self.residual = ResidualSchedule.from_config(config.get("planning", {}).get("residual"))
        if self.mixture is not None and not self._env_supports_cutoff():
            raise ValueError("training.cutoff_mixture needs an env with set_cutoff(k) "
                             "(a truncation that cuts the episode at a phase)")
        self._last_print = (time.time(), 0)
        if resume_state is not None:
            self.step = int(resume_state["total_env_steps"])
            self.episode_index = int(resume_state.get("episode_index", 0))
            self.success_count = int(resume_state.get("success_count", 0))
            self.env_errors = int(resume_state.get("env_errors", 0))
            self.wall_breaches = int(resume_state.get("wall_breaches", 0))
            if "rng_state" in resume_state:
                restore_rng(resume_state["rng_state"], self.rng)
            if self.mixture is not None and resume_state.get("cutoff_mixture"):
                self.mixture.load_state_dict(resume_state["cutoff_mixture"])
            if self.residual is not None and resume_state.get("residual"):
                self.residual.load_state_dict(resume_state["residual"])

    # ------------------------------------------------------------ helpers --
    def _env_supports_cutoff(self) -> bool:
        return callable(getattr(self.env, "set_cutoff", None))

    @property
    def eval_cutoff(self) -> Optional[int]:
        """Evaluation episodes are never cut (only when the run sets cutoffs at all)."""
        return None if self.mixture is None else self.mixture.full_episode_cutoff

    def _draw_seed_and_cutoff(self) -> tuple[int, Optional[int]]:
        seed = int(self.rng.integers(0, 2**31 - 1))
        return seed, (None if self.mixture is None else self.mixture.sample())

    def _new_episode(self) -> tuple[np.ndarray, _Episode]:
        seed, cutoff = self._draw_seed_and_cutoff()
        if cutoff is not None:
            self.env.set_cutoff(cutoff)
        observation, info = self.env.reset(seed=seed)
        self._check_cutoff(info, cutoff)
        return observation, _Episode(observation, self.label_names, info, cutoff)

    def _check_cutoff(self, info: Mapping[str, Any], cutoff: Optional[int], **where: Any) -> None:
        """An env that reports its cutoff must have the one the trainer drew."""
        if cutoff is not None and info.get("cutoff") is not None and int(info["cutoff"]) != cutoff:
            self._event("cutoff_mismatch", drawn=cutoff, env_reports=int(info["cutoff"]), **where)

    def _act(self, episode: _Episode, observation: np.ndarray) -> np.ndarray:
        if self.residual is None:
            return self.agent.act(observation, first_step=episode.length == 0, eval_mode=False)
        return self.agent.act(observation, first_step=episode.length == 0, eval_mode=False,
                              bounds=self.residual.box_for(episode.last_info))

    def _drop_step(self, episode: _Episode, failure: str, **where: Any) -> None:
        """A step the env flagged (``step_failure``) or that raised: not stored; the episode so far
        is closed as ``error`` (a simulation error or an exception) or ``breach`` (``wall_breach``)."""
        if failure == "wall_breach":
            self.wall_breaches += 1
        else:
            self.env_errors += 1
        self._event("dropped_step", **where, reason=failure, episode_length=episode.length,
                    **episode.stats.where())
        self._close_episode(episode, reason="breach" if failure == "wall_breach" else "error")

    def _close_episode(self, episode: _Episode, *, reason: str, success: bool = False) -> None:
        if episode.length == 0:
            return
        stats = episode.stats
        self.replay.add(np.stack(episode.observations), np.stack(episode.actions),
                        np.asarray(episode.rewards), np.asarray(episode.terminated),
                        {n: np.asarray(v) for n, v in episode.labels.items()},
                        phase=np.asarray(stats.phases))
        self.episode_index += 1
        self.success_count += int(success)
        record = {"step": self.step, "episode": self.episode_index, "length": episode.length,
                  "return": float(np.sum(episode.rewards)), "success": bool(success),
                  "end": reason}
        for name, values in episode.labels.items():
            record[f"{name}_max"] = float(np.max(values))
            record[f"{name}_mean"] = float(np.mean(values))
        record.update(stats.fields())
        self.logger.write("episodes", record)
        self.recent.append(record)
        self.window.append(record)
        if self.mixture is not None:
            self.mixture.record(stats.cutoff, stats.max_phase, success)
        if self.residual is not None:
            check = self.residual.record(stats.max_phase, success)
            if check is not None:
                self._event("residual_check", **check)

    def _curriculum_tick(self) -> None:
        if self.mixture is None:
            return
        change = self.mixture.maybe_update(self.step)
        if change is not None:
            self._event("cutoff_weights", **change)
            self.logger.print("cutoff weights " + " ".join(
                f"{k}:{w:.2f}" for k, w in change["weights"].items())
                + " (reach " + " ".join(f"{k}:{r:.2f}" for k, r in change["reach"].items()) + ")")

    def _event(self, kind: str, **fields: Any) -> None:
        self.logger.write("events", {"step": self.step, "event": kind, **fields})

    def _record_update(self, metrics: Mapping[str, float]) -> None:
        self.updates.add(metrics)
        self.last_update = dict(metrics)
        if self.updates.count >= int(self.cfg["logging"]["update_interval"]):
            self.logger.write("updates", {"step": self.step,
                                          "update": self.agent.update_count,
                                          **self.updates.pop()})

    def _run_evaluation(self, seeds: List[int]) -> Dict[str, Any]:
        return evaluate(self.env, self.agent, seeds, cutoff=self.eval_cutoff,
                        residual=self.residual)

    def _evaluate(self) -> Dict[str, Any]:
        ev = self.cfg["evaluation"]
        seeds = [int(ev["seed"]) + i for i in range(int(ev["episodes"]))]
        result = self._run_evaluation(seeds)
        self.logger.write("eval", {"step": self.step, **result["summary"]})
        s = result["summary"]
        self.logger.print(f"eval @ {self.step}: success {s['success_rate']:.2f}, "
                          f"return {s['return_mean']:.3f}, length {s['length_mean']:.1f}")
        return result

    def _metrics_extra(self) -> Dict[str, Any]:
        """More fields for metrics.jsonl (the parallel trainers add theirs)."""
        return {}

    def _print_progress(self, started: float, start_step: int) -> None:
        now = time.time()
        rate = (self.step - start_step) / max(now - started, 1e-9)
        recent = list(self.recent)
        success = np.mean([r["success"] for r in recent]) if recent else float("nan")
        ret = np.mean([r["return"] for r in recent]) if recent else float("nan")
        loss = self.last_update.get("total_loss", float("nan"))
        self.logger.print(f"step {self.step} | episodes {self.episode_index} | "
                          f"last-{len(recent)} success {success:.2f} return {ret:.3f} | "
                          f"loss {loss:.4f} | {rate:.1f} steps/s")
        summary = window_summary(list(self.window))
        line = progress_line(summary)
        if line is not None or self.mixture is not None:
            weights = ("" if self.mixture is None else " | cutoff weights " + " ".join(
                f"{k}:{w:.2f}" for k, w in self.mixture.weight_map().items()))
            self.logger.print((line or " ") + weights)
        last_time, last_step = self._last_print
        record = {"step": self.step, "episodes": self.episode_index,
                  "updates": self.agent.update_count, "env_errors": self.env_errors,
                  "wall_breaches": self.wall_breaches, "steps_per_s": rate,
                  "steps_per_s_recent": (self.step - last_step) / max(now - last_time, 1e-9),
                  **summary}
        if self.mixture is not None:
            record["cutoff_weights"] = self.mixture.weight_map()
        if self.residual is not None:
            record["residual_delta"] = self.residual.delta
        if self.replay.num_demo_episodes:
            record["demo_episodes"] = self.replay.num_demo_episodes
        if self.replay.stratify_by is not None:
            record["stratum_starts"] = self.replay.stratum_sizes()
        record.update(self._metrics_extra())
        self.logger.write("metrics", record)
        self._last_print = (now, self.step)

    def checkpoint_payload(self) -> Dict[str, Any]:
        payload = {
            "algorithm": "TD-MPC2", "config": dict(self.cfg),
            "agent": self.agent.state_dict(), "total_env_steps": self.step,
            "episode_index": self.episode_index, "success_count": self.success_count,
            "env_errors": self.env_errors, "wall_breaches": self.wall_breaches,
            "rng_state": capture_rng(self.rng),
            "observation_dim": int(self.env.observation_space.shape[0]),
            "observation_names": list(getattr(self.env, "observation_names", [])),
            "label_names": self.label_names,
        }
        if self.mixture is not None:
            payload["cutoff_mixture"] = self.mixture.state_dict()
        if self.residual is not None:
            payload["residual"] = self.residual.state_dict()
        if self.cfg["checkpoint"].get("save_replay", True):
            payload["replay"] = self.replay.state_dict()
        return payload

    def save(self) -> Path:
        path = save_checkpoint(self.run_dir / "checkpoints" / f"step_{self.step}.pt",
                               self.checkpoint_payload())
        self._event("checkpoint", path=str(path))
        return path

    # --------------------------------------------------------------- loop --
    def train(self) -> Dict[str, Any]:
        t = self.cfg["training"]
        total_steps = int(t["total_steps"])
        seed_steps = int(t["seed_steps"])
        initial_updates = int(t["initial_updates"])
        updates_per_step = int(t["updates_per_step"])
        checkpoint_every = int(self.cfg["checkpoint"]["interval"])
        eval_every = int(self.cfg["evaluation"]["interval"])
        print_every = int(self.cfg["logging"]["print_interval"])
        action_shape = self.env.action_space.shape

        pretrained = self.agent.update_count > 0
        eval_due = False
        started, start_step = time.time(), self.step
        self._last_print = (started, self.step)
        self.logger.print(f"training from step {self.step} to {total_steps}; "
                          f"replay has {self.replay.size} transitions")
        observation, episode = self._new_episode()
        while self.step < total_steps:
            if self.step < seed_steps:
                action = self.rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
            else:
                action = self._act(episode, observation)
            try:
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                failure = step_failure(info)
            except Exception as exc:  # SOFA or label failure inside the step
                failure, info = f"env_error: {exc!r}", {}
            if failure is not None:
                self._drop_step(episode, failure)
                observation, episode = self._new_episode()
                continue

            episode.add(action, reward, terminated, info["labels"], next_observation, info)
            observation = next_observation
            self.step += 1
            self._curriculum_tick()

            if self.step >= seed_steps and self.replay.can_sample():
                count = updates_per_step if pretrained else initial_updates
                if not pretrained:
                    self.logger.print(f"pretraining on seed data: {count} updates")
                    pretrained = True
                for _ in range(count):
                    self._record_update(self.agent.update(self.replay))

            if self.step % eval_every == 0:
                eval_due = True
            if terminated or truncated:
                self._close_episode(episode, reason="terminated" if terminated else "truncated",
                                    success=bool(info.get("is_success", False)))
                if eval_due:
                    self._evaluate()
                    eval_due = False
                observation, episode = self._new_episode()
            if self.step % checkpoint_every == 0:
                self.save()
            if self.step % print_every == 0:
                self._print_progress(started, start_step)

        final = self.save() if self.cfg["checkpoint"].get("save_final", True) else None
        summary = {"total_env_steps": self.step, "episodes": self.episode_index,
                   "success_count": self.success_count, "env_errors": self.env_errors,
                   "wall_breaches": self.wall_breaches,
                   "updates": self.agent.update_count,
                   "final_checkpoint": str(final) if final else None}
        self._event("finished", **summary)
        self.logger.print(f"finished: {summary}")
        return summary
