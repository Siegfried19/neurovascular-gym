"""Deterministic evaluation episodes (planner mean action, fixed seeds).

``cutoff``: when the run sets cutoffs (``training.cutoff_mixture``), evaluation
episodes are given the cutoff that never cuts, so they run to success, failure
or the step limit. ``residual``: the planner's phase-1 box, as in training.
Episode records carry the task metrics the env reports (``metrics.EpisodeStats``);
``summarize`` averages them, ``report`` adds what ``scripts/evaluate.py`` prints
(Wilson interval, highest-phase histogram, events, times, forces).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .curriculum import SUCCESS_PHASE
from .metrics import EpisodeStats


def episode_record(seed: int, total_reward: float, length: int, success: bool,
                   sim_error: bool, labels: Mapping[str, List[float]],
                   extra: Optional[Mapping[str, Any]] = None, *, wall_breach: bool = False) -> Dict[str, Any]:
    record = {"seed": int(seed), "return": total_reward, "length": length,
              "success": success, "simulation_error": sim_error, "wall_breach": bool(wall_breach)}
    for name, values in labels.items():
        record[f"{name}_max"] = max(values) if values else 0.0
        record[f"{name}_mean"] = float(np.mean(values)) if values else 0.0
    if extra:
        record.update({k: v for k, v in extra.items() if k not in ("cutoff", "sim_error")})
    return record


def summarize(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = {
        "episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "return_mean": float(np.mean([e["return"] for e in episodes])),
        "length_mean": float(np.mean([e["length"] for e in episodes])),
    }
    keys = list(dict.fromkeys(k for e in episodes for k in e))   # first-seen order
    for key in keys:
        if key.endswith("_max") or key.endswith("_mean"):
            summary[key] = float(np.mean([e[key] for e in episodes if key in e]))
    if all("max_phase" in e for e in episodes):            # task phases reported
        summary["max_phase_hist"] = phase_histogram(episodes)
        for key in ("shield_active_frac", "catheter_moving_frac", "d_p50"):
            values = [e[key] for e in episodes if key in e]
            if values:
                summary[key] = float(np.mean(values))
        events = events_per_episode(episodes)
        if events:
            summary["events_per_episode"] = events
    return summary


def phase_histogram(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Episodes by the highest phase they reached; success counts as phase 5."""
    counts = Counter(SUCCESS_PHASE if e["success"] else int(e["max_phase"]) for e in episodes)
    return {str(k): int(counts.get(k, 0)) for k in range(SUCCESS_PHASE + 1)}


def events_per_episode(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    with_events = [e.get("events", {}) for e in episodes]
    names = sorted({name for e in with_events for name in e})
    return {name: float(np.mean([e.get(name, 0) for e in with_events])) for name in names}


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def report(episodes: Sequence[Mapping[str, Any]], *, step_seconds: Optional[float] = None,
           force_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """``summarize`` plus the evaluation table: success with its Wilson interval,
    highest phases, events per episode, episode times, peak forces, shield."""
    out = summarize(episodes)
    n, wins = len(episodes), int(sum(bool(e["success"]) for e in episodes))
    low, high = wilson_interval(wins, n)
    out.update(successes=wins, success_ci95=[low, high],
               simulation_errors=int(sum(bool(e["simulation_error"]) for e in episodes)),
               wall_breaches=int(sum(bool(e.get("wall_breach")) for e in episodes)))
    if all("max_phase" in e for e in episodes):
        out["max_phase_hist"] = phase_histogram(episodes)
        out["events_per_episode"] = events_per_episode(episodes)
    lengths = np.asarray([e["length"] for e in episodes], dtype=float)
    won = np.asarray([bool(e["success"]) for e in episodes])
    times: Dict[str, Any] = {"steps_mean": float(lengths.mean()), "steps_median": float(np.median(lengths))}
    if won.any():
        times.update(success_steps_mean=float(lengths[won].mean()),
                     success_steps_median=float(np.median(lengths[won])))
    if step_seconds:
        times.update({k.replace("steps", "seconds"): v * step_seconds for k, v in list(times.items())})
    wall = [e["wall_s"] for e in episodes if "wall_s" in e]
    if wall:
        times["wall_s_per_episode"] = float(np.mean(wall))
    out["time"] = times
    names = force_labels if force_labels is not None else sorted(
        k[:-4] for k in episodes[0] if k.endswith("_max") and "force" in k)
    forces = {}
    for name in names:
        peaks = np.asarray([e[f"{name}_max"] for e in episodes if f"{name}_max" in e], dtype=float)
        if len(peaks):
            forces[name] = {"max": float(peaks.max()), "mean_of_episode_max": float(peaks.mean()),
                            "p90_of_episode_max": float(np.percentile(peaks, 90))}
    if forces:
        out["peak_force"] = forces
    for key in ("shield_delta_mean", "shield_active_frac", "shield_penalized_mean",
                "catheter_moving_frac"):
        values = [e[key] for e in episodes if key in e]
        if values:
            out[key] = float(np.mean(values))
    out["end_reasons"] = dict(sorted(Counter(end_reason(e) for e in episodes).items()))
    return out


def end_reason(episode: Mapping[str, Any]) -> str:
    """success | failure:<reason> (the env's hard failure) | simulation_error | wall_breach (a
    device's centerline outside the vessel mesh) | other (step limit or anything else that
    truncated it)."""
    if episode["success"]:
        return "success"
    if episode.get("failure_reason"):
        return f"failure:{episode['failure_reason']}"
    if episode.get("simulation_error"):
        return "simulation_error"
    if episode.get("wall_breach"):
        return "wall_breach"
    return "other"


def _act(agent, observation, length: int, info: Mapping[str, Any], residual, eval_mode: bool):
    if residual is None:
        return agent.act(observation, first_step=length == 0, eval_mode=eval_mode)
    return agent.act(observation, first_step=length == 0, eval_mode=eval_mode,
                     bounds=residual.box_for(info))


def run_episode(env, agent, seed: int, *, eval_mode: bool = True, cutoff: Optional[int] = None,
                residual=None, action_hook: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                hooks=None) -> Dict[str, Any]:
    """One episode. ``action_hook`` changes the planned action before it is sent
    (e.g. freezing the catheter); ``hooks`` may define ``on_reset(env, obs, info,
    seed)``, ``on_step(env, action, obs, reward, terminated, truncated, info)`` and
    ``on_error(env, action, error)`` (e.g. to record frames)."""
    if cutoff is not None:
        env.set_cutoff(int(cutoff))
    observation, info = env.reset(seed=int(seed))
    if hooks is not None and hasattr(hooks, "on_reset"):
        hooks.on_reset(env, observation, info, int(seed))
    stats = EpisodeStats(info)
    labels: Dict[str, List[float]] = {name: [] for name in env.label_names}
    total_reward, length, success, sim_error, breach = 0.0, 0, False, False, False
    terminated = truncated = False
    while not (terminated or truncated):
        action = _act(agent, observation, length, stats.last_info, residual, eval_mode)
        if action_hook is not None:
            action = np.asarray(action_hook(action), dtype=np.float32)
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:  # a SOFA failure ends this episode, not the run
            sim_error = True
            if hooks is not None and hasattr(hooks, "on_error"):
                hooks.on_error(env, action, exc)
            break
        total_reward += float(reward)
        length += 1
        success = bool(info.get("is_success", False))
        for name, value in info.get("labels", {}).items():
            labels[name].append(float(value))
        stats.add(action, info)
        if hooks is not None and hasattr(hooks, "on_step"):
            hooks.on_step(env, action, observation, reward, terminated, truncated, info)
        if info.get("simulation_error", False):
            sim_error = True
            break
        if info.get("wall_breached", False):
            breach = True
            break
    return episode_record(seed, total_reward, length, success, sim_error, labels, stats.fields(),
                          wall_breach=breach)


def evaluate(env, agent, seeds: Sequence[int], *, cutoff: Optional[int] = None, residual=None,
             action_hook=None, hooks=None) -> Dict[str, Any]:
    """Run one episode per seed. Torch's global RNG is restored afterwards so
    evaluation does not shift the training run's random stream."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    episodes = []
    with torch.random.fork_rng(devices=devices):
        for seed in seeds:
            torch.manual_seed(int(seed))
            episodes.append(run_episode(env, agent, seed, cutoff=cutoff, residual=residual,
                                        action_hook=action_hook, hooks=hooks))
    return {"summary": summarize(episodes), "episodes": episodes}


def _rng_state() -> Dict[str, Any]:
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class _Slot:
    def __init__(self, order: int, seed: int, label_names: Sequence[str]) -> None:
        self.order, self.seed = order, seed
        self.observation: Optional[np.ndarray] = None
        self.total_reward, self.length, self.success = 0.0, 0, False
        self.labels: Dict[str, List[float]] = {name: [] for name in label_names}
        self.stats = EpisodeStats()
        self.rng: Optional[Dict[str, Any]] = None        # torch RNG after this episode's last plan
        self.warm_start: Optional[torch.Tensor] = None   # the planner's previous_mean, this episode

    def record(self, sim_error: bool, wall_breach: bool = False) -> Dict[str, Any]:
        return episode_record(self.seed, self.total_reward, self.length, self.success,
                              sim_error, self.labels, self.stats.fields(), wall_breach=wall_breach)


def evaluate_pool(pool, agent, seeds: Sequence[int], *, cutoff: Optional[int] = None,
                  residual=None) -> Dict[str, Any]:
    """``evaluate`` on an ``EnvPool``: the same episodes (one per seed, planner
    mean action), ``len(pool)`` of them stepping at a time. Each episode plans
    with its own torch RNG stream (seeded from its seed, as ``evaluate`` does)
    and its own planner warm start, so the numbers are exactly the ones
    ``evaluate`` gives, whatever the pool size (tests/test_parallel.py checks
    this bit for bit). A failed step or reset ends that episode as a
    simulation error instead of raising."""
    from .env_pool import reset_command

    n = len(pool)
    queue = list(enumerate(int(s) for s in seeds))
    slots: List[Optional[_Slot]] = [None] * n
    records: Dict[int, Dict[str, Any]] = {}
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        while queue or any(slot is not None for slot in slots):
            commands = {}
            for i, slot in enumerate(slots):
                if slot is None or slot.observation is None:
                    continue
                if slot.rng is None:
                    torch.manual_seed(slot.seed)
                else:
                    _set_rng_state(slot.rng)
                    agent.previous_mean.copy_(slot.warm_start)
                action = _act(agent, slot.observation, slot.length, slot.stats.last_info,
                              residual, True)
                slot.rng, slot.warm_start = _rng_state(), agent.previous_mean.clone()
                commands[i] = ("step", action)
            for i in range(n):
                if slots[i] is None and queue:
                    order, seed = queue.pop(0)
                    slots[i] = _Slot(order, seed, pool.label_names)
                    commands[i] = reset_command(seed, cutoff)
            pool.send(commands)
            for i, result in pool.receive().items():
                slot = slots[i]
                if commands[i][0] == "reset" and result.ok:
                    slot.observation = result.value[0]
                    slot.stats = EpisodeStats(result.value[1])
                    continue
                done, sim_error, breach = not result.ok, not result.ok, False
                if result.ok:
                    observation, reward, terminated, truncated, info = result.value
                    slot.total_reward += float(reward)
                    slot.length += 1
                    slot.success = bool(info.get("is_success", False))
                    for name, value in info.get("labels", {}).items():
                        slot.labels[name].append(float(value))
                    slot.stats.add(commands[i][1], info)
                    slot.observation = observation
                    sim_error = bool(info.get("simulation_error", False))
                    breach = bool(info.get("wall_breached", False))
                    done = terminated or truncated or sim_error or breach
                if done:
                    records[slot.order] = slot.record(sim_error, breach)
                    slots[i] = None
    episodes = [records[k] for k in range(len(records))]
    return {"summary": summarize(episodes), "episodes": episodes}
