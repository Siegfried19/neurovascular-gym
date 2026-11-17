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
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .curriculum import SUCCESS_PHASE
from .metrics import EpisodeStats


def episode_record(seed: int, total_reward: float, length: int, success: bool,
                   sim_error: bool, labels: Mapping[str, List[float]],
                   extra: Optional[Mapping[str, Any]] = None, *, wall_breach: bool = False,
                   true_end: bool = False, q_values: Optional[Sequence[float]] = None,
                   rewards: Optional[Sequence[float]] = None, discount: float = 1.0,
                   materials: Optional[Sequence[Any]] = None, components=None) -> Dict[str, Any]:
    record = {"seed": int(seed), "return": total_reward, "length": length,
              "success": success, "simulation_error": sim_error, "wall_breach": bool(wall_breach),
              "true_end": bool(true_end)}
    for name, values in labels.items():
        record[f"{name}_max"] = max(values) if values else 0.0
        record[f"{name}_mean"] = float(np.mean(values)) if values else 0.0
    if extra:
        record.update({k: v for k, v in extra.items() if k not in ("cutoff", "sim_error")})
    if q_values and rewards and len(q_values) == len(rewards):
        record.update(q_versus_return(q_values, rewards, discount))
    if components is not None and materials and len(materials) == length:
        record["reward_parts"] = components(np.stack([np.asarray(m, dtype=np.float32) for m in materials]))
    return record


def q_versus_return(q_values: Sequence[float], rewards: Sequence[float], discount: float) -> Dict[str, float]:
    """The Q the model gave at every step against the discounted return realized from there
    (docs/wandb_metrics.md 2.3): means of q, G and q - G over the episode, and the first step's."""
    r = np.asarray(rewards, dtype=np.float64)
    g = np.zeros_like(r)
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        running = r[t] + discount * running
        g[t] = running
    q = np.asarray(q_values, dtype=np.float64)
    return {"q_mean": float(q.mean()), "g_mean": float(g.mean()), "q_minus_g_mean": float((q - g).mean()),
            "q0": float(q[0]), "g0": float(g[0])}


def summarize(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = {
        "episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "return_mean": float(np.mean([e["return"] for e in episodes])),
        "length_mean": float(np.mean([e["length"] for e in episodes])),
    }
    keys = list(dict.fromkeys(k for e in episodes for k in e))   # first-seen order
    for key in keys:
        if key.startswith(("q_", "g_")):
            continue                  # Q against the return: only over the episodes that really ended (below)
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
    tiers = [e["tier1"] for e in episodes if "tier1" in e]
    if tiers:
        summary["tier1_rate"] = float(np.mean(tiers))
        summary["tier1_ci95_low"], summary["tier1_ci95_high"] = wilson_interval(
            int(sum(bool(t) for t in tiers)), len(tiers))
    for key in ("progress_end", "lead_end"):
        values = [e[key] for e in episodes if key in e]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    # what the evaluation table had and the eval.jsonl line lacked (docs/wandb_metrics.md, section 1)
    n, wins = len(episodes), int(sum(bool(e["success"]) for e in episodes))
    low, high = wilson_interval(wins, n)
    summary["success_ci95_low"], summary["success_ci95_high"] = low, high
    summary["end_reasons"] = dict(sorted(Counter(end_reason(e) for e in episodes).items()))
    lengths = np.asarray([e["length"] for e in episodes], dtype=float)
    won = np.asarray([bool(e["success"]) for e in episodes])
    if won.any():
        summary["success_steps_mean"] = float(lengths[won].mean())
        summary["success_steps_median"] = float(np.median(lengths[won]))
    for key in sorted(k[:-4] for k in episodes[0] if k.endswith("_max") and "force" in k):
        peaks = np.asarray([e[f"{key}_max"] for e in episodes if f"{key}_max" in e], dtype=float)
        if len(peaks):
            summary[f"{key}_peak_p90"] = float(np.percentile(peaks, 90))
            summary[f"{key}_peak_max"] = float(peaks.max())
    ends = [e["progress_end"] for e in episodes if "progress_end" in e]
    if ends:
        hist = Counter("done" if e["success"] else str(min(10, int(e["progress_end"] * 10) + 1))
                       for e in episodes if "progress_end" in e)
        summary["progress_end_hist"] = dict(sorted(hist.items()))
    leads = [e["lead_end"] for e in episodes if "lead_end" in e]
    if leads:
        summary["lead_end_below_30_rate"] = float(np.mean(np.asarray(leads) < 30.0))
    # Q against the realized return, over the episodes that really ended (not truncated)
    compared = [e for e in episodes if e.get("true_end") and "q_mean" in e]
    if compared:
        for key in ("q_mean", "g_mean", "q_minus_g_mean", "q0", "g0"):
            summary[key] = float(np.mean([e[key] for e in compared]))
        summary["q_versus_return_episodes"] = len(compared)
    parts = [e["reward_parts"] for e in episodes if "reward_parts" in e]
    if parts:
        summary["reward_parts"] = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
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
    device's centerline outside the vessel mesh) | stalled (ended because the devices stood still,
    ``evaluation.end_when_stalled``) | other (step limit or anything else that truncated it)."""
    if episode["success"]:
        return "success"
    if episode.get("failure_reason"):
        return f"failure:{episode['failure_reason']}"
    if episode.get("simulation_error"):
        return "simulation_error"
    if episode.get("wall_breach"):
        return "wall_breach"
    if episode.get("stalled"):
        return "stalled"
    return "other"


def _act(agent, observation, length: int, info: Mapping[str, Any], residual, eval_mode: bool):
    if residual is None:
        return agent.act(observation, first_step=length == 0, eval_mode=eval_mode)
    return agent.act(observation, first_step=length == 0, eval_mode=eval_mode,
                     bounds=residual.box_for(info))


def _components_of(env, components):
    """The reward-breakdown function: the one given, else the env's own formula's."""
    if components is not None:
        return components
    formula = getattr(env, "reward_formula", None)
    if formula is None:
        return None
    from ..env.two_device.reward_formula import component_sums
    return lambda materials: component_sums(materials, formula.coefficients)


def _q_of(agent, observation, action) -> Optional[float]:
    reader = getattr(agent, "q_value", None)
    return None if reader is None else float(reader(observation, action))


def run_episode(env, agent, seed: int, *, eval_mode: bool = True, cutoff: Optional[int] = None,
                residual=None, action_hook: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                hooks=None, components=None, end_when_stalled: bool = False) -> Dict[str, Any]:
    """One episode. ``end_when_stalled``: the episode ends once the devices stand still (envs with the
    two-device ``Stalled`` part; ``evaluation.end_when_stalled``). ``action_hook`` changes the planned action before it is sent
    (e.g. freezing the catheter); ``hooks`` may define ``on_reset(env, obs, info,
    seed)``, ``on_step(env, action, obs, reward, terminated, truncated, info)`` and
    ``on_error(env, action, error)`` (e.g. to record frames). The record also carries the
    model's Q at every step against the realized return (agents with ``q_value``) and the reward
    broken into its terms (envs with a reward formula, or ``components``: materials -> dict)."""
    if cutoff is not None:
        env.set_cutoff(int(cutoff))
    if end_when_stalled and callable(getattr(env, "end_when_stalled", None)):
        env.end_when_stalled()
    observation, info = env.reset(seed=int(seed))
    if hooks is not None and hasattr(hooks, "on_reset"):
        hooks.on_reset(env, observation, info, int(seed))
    stats = EpisodeStats(info)
    labels: Dict[str, List[float]] = {name: [] for name in env.label_names}
    total_reward, length, success, sim_error, breach = 0.0, 0, False, False, False
    terminated = truncated = False
    q_values: List[float] = []
    rewards: List[float] = []
    materials: List[Any] = []
    components = _components_of(env, components)
    while not (terminated or truncated):
        action = _act(agent, observation, length, stats.last_info, residual, eval_mode)
        if action_hook is not None:
            action = np.asarray(action_hook(action), dtype=np.float32)
        q = _q_of(agent, observation, action)
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
        if q is not None:
            q_values.append(q)
        rewards.append(float(reward))
        if info.get("reward_materials") is not None:
            materials.append(info["reward_materials"])
        if hooks is not None and hasattr(hooks, "on_step"):
            hooks.on_step(env, action, observation, reward, terminated, truncated, info)
        if info.get("simulation_error", False):
            sim_error = True
            break
        if info.get("wall_breached", False):
            breach = True
            break
    return episode_record(seed, total_reward, length, success, sim_error, labels, stats.fields(),
                          wall_breach=breach, true_end=bool(terminated) and not sim_error and not breach,
                          q_values=q_values, rewards=rewards, discount=float(getattr(agent, "discount", 1.0)),
                          materials=materials, components=components)


def evaluate(env, agent, seeds: Sequence[int], *, cutoff: Optional[int] = None, residual=None,
             action_hook=None, hooks=None, components=None, end_when_stalled: bool = False) -> Dict[str, Any]:
    """Run one episode per seed. Torch's global RNG is restored afterwards so
    evaluation does not shift the training run's random stream."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    episodes = []
    with torch.random.fork_rng(devices=devices):
        for seed in seeds:
            torch.manual_seed(int(seed))
            episodes.append(run_episode(env, agent, seed, cutoff=cutoff, residual=residual,
                                        action_hook=action_hook, hooks=hooks, components=components,
                                        end_when_stalled=end_when_stalled))
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
        self.q_values: List[float] = []
        self.rewards: List[float] = []
        self.materials: List[Any] = []
        self.pending_q: Optional[float] = None
        self.terminated = False

    def record(self, sim_error: bool, wall_breach: bool = False, *, discount: float = 1.0,
               components=None) -> Dict[str, Any]:
        return episode_record(self.seed, self.total_reward, self.length, self.success,
                              sim_error, self.labels, self.stats.fields(), wall_breach=wall_breach,
                              true_end=self.terminated and not sim_error and not wall_breach,
                              q_values=self.q_values, rewards=self.rewards, discount=discount,
                              materials=self.materials, components=components)


class PoolEvaluation:
    """``evaluate_pool`` one round at a time: ``step(wait)`` sends what can be sent, takes the answers
    that are back and says whether every episode has ended; ``result()`` then gives the same dict as
    ``evaluate_pool``. For an evaluation that runs beside training (``evaluation.background``): the
    caller calls ``step(0)`` between its own work, inside ``torch.random.fork_rng`` so that the
    episodes' RNG streams do not touch its own."""

    def __init__(self, pool, agent, seeds: Sequence[int], *, cutoff: Optional[int] = None,
                 residual=None, components=None, end_when_stalled: bool = False) -> None:
        self.pool, self.agent = pool, agent
        self.end_when_stalled = bool(end_when_stalled)
        self.cutoff, self.residual, self.components = cutoff, residual, components
        self.queue = list(enumerate(int(s) for s in seeds))
        self.slots: List[Optional[_Slot]] = [None] * len(pool)
        self.records: Dict[int, Dict[str, Any]] = {}
        self.in_flight: Dict[int, Tuple[str, Any]] = {}  # env index -> the command it still owes an answer to

    @property
    def done(self) -> bool:
        return not self.queue and all(slot is None for slot in self.slots)

    def step(self, wait: float) -> bool:
        from .env_pool import reset_command

        if self.done:
            return True
        pool, agent, slots, in_flight = self.pool, self.agent, self.slots, self.in_flight
        commands = {}
        for i, slot in enumerate(slots):
            if slot is None or slot.observation is None or i in in_flight:
                continue
            if slot.rng is None:
                torch.manual_seed(slot.seed)
            else:
                _set_rng_state(slot.rng)
                agent.previous_mean.copy_(slot.warm_start)
            action = _act(agent, slot.observation, slot.length, slot.stats.last_info,
                          self.residual, True)
            slot.rng, slot.warm_start = _rng_state(), agent.previous_mean.clone()
            slot.pending_q = _q_of(agent, slot.observation, action)
            commands[i] = ("step", action)
        for i in range(len(pool)):
            if slots[i] is None and i not in in_flight and self.queue:
                order, seed = self.queue.pop(0)
                slots[i] = _Slot(order, seed, pool.label_names)
                commands[i] = reset_command(seed, self.cutoff, end_when_stalled=self.end_when_stalled)
        if commands:
            pool.send(commands)
            in_flight.update(commands)
        results = pool.receive_ready(wait)
        for i in sorted(results):
            result, command = results[i], in_flight.pop(i)
            slot = slots[i]
            if command[0] == "reset" and result.ok:
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
                slot.stats.add(command[1], info)
                slot.observation = observation
                if slot.pending_q is not None:
                    slot.q_values.append(slot.pending_q)
                slot.rewards.append(float(reward))
                if info.get("reward_materials") is not None:
                    slot.materials.append(info["reward_materials"])
                slot.terminated = bool(terminated)
                sim_error = bool(info.get("simulation_error", False))
                breach = bool(info.get("wall_breached", False))
                done = terminated or truncated or sim_error or breach
            if done:
                self.records[slot.order] = slot.record(sim_error, breach,
                                                       discount=float(getattr(agent, "discount", 1.0)),
                                                       components=self.components)
                slots[i] = None
        return self.done

    def cut(self) -> List[int]:
        """End the evaluation now (``evaluation.max_minutes``): every episode that has not ended, or not
        started, is recorded as a simulation error with what it had so far. Returns their seeds. The
        pool still owes answers to the cut episodes' last commands; the caller closes it."""
        seeds = []
        discount = float(getattr(self.agent, "discount", 1.0))
        for i, slot in enumerate(self.slots):
            if slot is not None:
                self.records[slot.order] = slot.record(True, False, discount=discount, components=self.components)
                seeds.append(slot.seed)
                self.slots[i] = None
        for order, seed in self.queue:
            self.records[order] = _Slot(order, seed, self.pool.label_names).record(
                True, False, discount=discount, components=self.components)
            seeds.append(seed)
        self.queue = []
        return seeds

    def result(self) -> Dict[str, Any]:
        episodes = [self.records[k] for k in range(len(self.records))]
        return {"summary": summarize(episodes), "episodes": episodes}


def evaluate_pool(pool, agent, seeds: Sequence[int], *, cutoff: Optional[int] = None,
                  residual=None, components=None, max_minutes: Optional[float] = None,
                  end_when_stalled: bool = False) -> Dict[str, Any]:
    """``evaluate`` on an ``EnvPool``: the same episodes (one per seed, planner
    mean action), ``len(pool)`` of them stepping at a time. The envs do not
    wait for each other: each env is stepped as soon as its own answer is back
    (a slow simulator step holds up only its own episode). Each episode plans
    with its own torch RNG stream (seeded from its seed, as ``evaluate`` does)
    and its own planner warm start, so the numbers are exactly the ones
    ``evaluate`` gives, whatever the pool size and whatever order the answers
    come back in (tests/test_parallel.py checks this bit for bit). A failed
    step or reset ends that episode as a simulation error instead of raising.
    ``max_minutes`` (``evaluation.max_minutes``): after that long, the episodes that have not ended
    are recorded as simulation errors (``PoolEvaluation.cut``; their seeds in ``result["cut_seeds"]``)."""
    evaluation = PoolEvaluation(pool, agent, seeds, cutoff=cutoff, residual=residual,
                                components=components, end_when_stalled=end_when_stalled)
    deadline = time.time() + 60.0 * float(max_minutes) if max_minutes else None   # None or 0: no limit
    cut: List[int] = []
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        while not evaluation.step(0.5):          # 0.5 s to wait for an answer before looking again
            if deadline is not None and time.time() > deadline:
                cut = evaluation.cut()           # ``max_minutes``: the episodes still running end as sim errors
    result = evaluation.result()
    if cut:
        result["cut_seeds"] = cut
    return result
