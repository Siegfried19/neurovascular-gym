"""Per-episode task metrics read from the env's ``info``, and their rolling summary.

What an env may report each step (all optional; a field is only written when the
env gives it, so a run on an env without them logs exactly what it always did):

| info key | meaning | becomes (per episode) |
| --- | --- | --- |
| ``phase`` | task phase after the step (0-4; also in the reset info) | ``max_phase``, ``last_phase`` (where it ended: for an episode cut by a simulation error, the phase it crashed in), the replay's per-step phase |
| ``max_phase`` | highest phase so far | ``max_phase`` |
| ``events`` | ``{name: n}`` counted on the step they happen | ``events: {name: total}`` |
| ``shield_delta`` | how much the safety shield changed the action (number or vector; a vector counts by its length) | ``shield_delta_mean``, ``shield_active_frac`` (steps where it was > 0.001) |
| ``shield_penalized`` | the part of that change the reward charges for | ``shield_penalized_mean`` |
| ``d`` | wire minus catheter insertion length, mm | ``d_p10``, ``d_p50``, ``d_p90`` |
| ``executed_action`` | the action really applied, in the policy's [-1, 1] units | ``catheter_moving_frac`` |
| ``simulation_error`` / ``sim_error`` | numerical failure | ``sim_error: true`` |
| ``wall_breached`` | a device's centerline outside the vessel mesh (two_device/wall.py); the trainers drop the step and close the episode as ``end: breach`` | ``breach_rate`` in the window summary (the step itself never reaches ``add``) |
| ``failure_reason`` | why the env ended it as a failure (last step) | ``failure_reason`` |
| ``cutoff`` | the cutoff in force (reset info) | ``cutoff`` (when the trainer set none) |
| ``tier1`` | the first tier was paid (round two) | ``tier1`` |
| ``progress_ratio`` | the task's progress, 0-1 (round two) | ``progress_end`` (where it ended) |
| ``d`` (last step) | | ``lead_end`` (the wire's lead at the end, mm) |

``catheter_moving_frac`` (actions with 4 or more values: wire push, wire turn,
catheter push, catheter turn): the fraction of steps whose catheter push
``|a[2]| > 0.05``, read from ``executed_action`` when given (and inside
[-1, 1]), else from the policy's action.

Reach rates (``reach``) count only episodes whose cutoff let them get there: for
phase ``k``, episodes with cutoff >= k (or no cutoff); for success, episodes
whose cutoff never cuts (5) or that had none.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .curriculum import SUCCESS_PHASE

CATHETER_PUSH = 2              # action index of the catheter's push, two-instrument envs
MOVING = 0.05                  # |normalized push| above this counts as moving
SHIELD_ACTIVE = 1e-3           # shield change (normalized units) above this counts as an intervention
REACH_PHASES = (1, 2, 3, 4)


def sim_error_of(info: Mapping[str, Any]) -> bool:
    return bool(info.get("simulation_error", False) or info.get("sim_error", False))


def _magnitude(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(array)) if array.size > 1 else float(array[0]) if array.size else 0.0


class EpisodeStats:
    """Accumulates one episode's task metrics, step by step."""

    def __init__(self, reset_info: Optional[Mapping[str, Any]] = None,
                 cutoff: Optional[int] = None) -> None:
        reset_info = reset_info or {}
        if cutoff is None and reset_info.get("cutoff") is not None:
            cutoff = int(reset_info["cutoff"])        # the env's own (e.g. its YAML default)
        self.cutoff = cutoff
        self.has_phase = "phase" in reset_info
        self.phases: List[int] = [int(reset_info.get("phase", 0))]
        self.max_phase = int(reset_info.get("max_phase", self.phases[0]))
        self.events: Dict[str, int] = {}
        self.has_events = False
        self.shield: List[float] = []
        self.penalized: List[float] = []
        self.d: List[float] = []
        self.catheter_moving = 0
        self.catheter_steps = 0
        self.sim_error = False
        self.tier1: Optional[bool] = None
        self.progress_end: Optional[float] = (float(reset_info["progress_ratio"])
                                              if reset_info.get("progress_ratio") is not None else None)
        self.last_info: Mapping[str, Any] = reset_info

    def add(self, action: np.ndarray, info: Mapping[str, Any]) -> None:
        self.last_info = info
        if "phase" in info:
            self.has_phase = True
        phase = int(info.get("phase", self.phases[-1]))
        self.phases.append(phase)
        self.max_phase = max(self.max_phase, phase, int(info.get("max_phase", phase)))
        events = info.get("events")
        if events is not None:
            self.has_events = True
            for name, count in events.items():
                self.events[name] = self.events.get(name, 0) + int(count)
        if info.get("shield_delta") is not None:
            self.shield.append(_magnitude(info["shield_delta"]))
        if info.get("shield_penalized") is not None:
            self.penalized.append(float(info["shield_penalized"]))
        if info.get("d") is not None:
            self.d.append(float(info["d"]))
        applied = np.asarray(action, dtype=np.float64).reshape(-1)
        executed = info.get("executed_action")
        if executed is not None:
            executed = np.asarray(executed, dtype=np.float64).reshape(-1)
            if executed.shape == applied.shape and np.all(np.abs(executed) <= 1.0 + 1e-6):
                applied = executed
        if applied.size > CATHETER_PUSH + 1:
            self.catheter_steps += 1
            self.catheter_moving += int(abs(applied[CATHETER_PUSH]) > MOVING)
        if sim_error_of(info):
            self.sim_error = True
        if info.get("tier1") is not None:
            self.tier1 = bool(info["tier1"])
        if info.get("progress_ratio") is not None:
            self.progress_end = float(info["progress_ratio"])

    def where(self) -> Dict[str, Any]:
        """The phase the episode is in now (for event records), when the env reports phases."""
        return {"phase": self.phases[-1]} if self.has_phase else {}

    def fields(self) -> Dict[str, Any]:
        """Record fields for this episode (only what the env reported)."""
        out: Dict[str, Any] = {}
        if self.has_phase:
            out["max_phase"] = self.max_phase
            out["last_phase"] = self.phases[-1]
        if self.cutoff is not None:
            out["cutoff"] = int(self.cutoff)
        if self.has_events:
            out["events"] = dict(sorted(self.events.items()))
        if self.shield:
            out["shield_delta_mean"] = float(np.mean(self.shield))
            out["shield_active_frac"] = float(np.mean(np.asarray(self.shield) > SHIELD_ACTIVE))
        if self.penalized:
            out["shield_penalized_mean"] = float(np.mean(self.penalized))
        if self.d:
            p10, p50, p90 = np.percentile(self.d, [10, 50, 90])
            out.update(d_p10=float(p10), d_p50=float(p50), d_p90=float(p90))
        if self.catheter_steps:
            out["catheter_moving_frac"] = self.catheter_moving / self.catheter_steps
        if self.sim_error:
            out["sim_error"] = True
        if self.last_info.get("failure_reason"):
            out["failure_reason"] = str(self.last_info["failure_reason"])
        if self.last_info.get("scenario"):          # this episode's scene parameters (env/randomization.py)
            out["scenario"] = {k: float(v) for k, v in self.last_info["scenario"].items()}
        if self.last_info.get("stalled"):
            out["stalled"] = True
        if self.tier1 is not None:
            out["tier1"] = bool(self.tier1)
        if self.progress_end is not None:
            out["progress_end"] = float(self.progress_end)
        if self.d:
            out["lead_end"] = float(self.d[-1])
        return out


def reach_rates(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Optional[float]]]:
    """Censor-aware reach rates over episode records that carry ``max_phase``."""
    with_phase = [r for r in records if "max_phase" in r]
    if not with_phase:
        return None
    out: Dict[str, Optional[float]] = {}
    for k in REACH_PHASES:
        pool = [r for r in with_phase if r.get("cutoff") is None or r["cutoff"] >= k]
        out[f"P{k}"] = (float(np.mean([r["success"] or r["max_phase"] >= k for r in pool]))
                        if pool else None)
    pool = [r for r in with_phase if r.get("cutoff") is None or r["cutoff"] >= SUCCESS_PHASE]
    out["success"] = float(np.mean([r["success"] for r in pool])) if pool else None
    return out


def window_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Averages over a window of episode records (what metrics.jsonl carries)."""
    n = len(records)
    out: Dict[str, Any] = {"window_episodes": n}
    if not n:
        return out
    out["success_rate"] = float(np.mean([r["success"] for r in records]))
    out["return_mean"] = float(np.mean([r["return"] for r in records]))
    out["length_mean"] = float(np.mean([r["length"] for r in records]))
    out["error_rate"] = float(np.mean([r.get("end") == "error" or bool(r.get("sim_error"))
                                       for r in records]))
    out["breach_rate"] = float(np.mean([r.get("end") == "breach" for r in records]))
    reach = reach_rates(records)
    if reach is not None:
        out["reach"] = reach
        hist = defaultdict(int)
        for r in records:
            if "max_phase" in r:
                hist[str(SUCCESS_PHASE if r["success"] else r["max_phase"])] += 1
        out["max_phase_hist"] = dict(sorted(hist.items()))
    for key in ("shield_delta_mean", "shield_active_frac", "shield_penalized_mean",
                "catheter_moving_frac", "d_p10", "d_p50", "d_p90"):
        values = [r[key] for r in records if key in r]
        if values:
            out[key] = float(np.mean(values))
    tiers = [r["tier1"] for r in records if "tier1" in r]
    if tiers:
        out["tier1_rate"] = float(np.mean(tiers))
    ends = [r["progress_end"] for r in records if "progress_end" in r]
    if ends:
        out["progress_end_mean"] = float(np.mean(ends))
        hist = defaultdict(int)
        for r in records:
            if "progress_end" in r:
                hist[str(min(10, int(r["progress_end"] * 10) + 1) if not r["success"] else "done")] += 1
        out["progress_end_hist"] = dict(sorted(hist.items()))
    leads = [r["lead_end"] for r in records if "lead_end" in r]
    if leads:
        out["lead_end_p50"] = float(np.median(leads))
    prefixed = [r for r in records if "prefix_steps" in r]
    if prefixed:
        out["prefix_step_share"] = float(sum(r["prefix_steps"] for r in prefixed)
                                         / max(sum(r["length"] for r in prefixed), 1))
        outcomes = [r.get("prefix_outcome") for r in prefixed if r.get("prefix_outcome")]
        if outcomes:
            out["prefix_gave_up_rate"] = float(np.mean([o == "gave_up" for o in outcomes]))
    parts = [r["reward_parts"] for r in records if isinstance(r.get("reward_parts"), dict)]
    if parts:
        out["reward_parts"] = {k: float(np.mean([p[k] for p in parts if k in p])) for k in parts[0]}
    starts = [r["start_grid"] for r in records if "start_grid" in r]
    if starts:
        values, counts = np.unique(starts, return_counts=True)
        out["start_grid_share"] = {str(int(v)): float(c / len(starts)) for v, c in zip(values, counts)}
        out["success_by_start_grid"] = {
            str(int(v)): float(np.mean([r["success"] for r in records if r.get("start_grid") == v]))
            for v in values}
    with_events = [r["events"] for r in records if "events" in r]
    if with_events:
        names = sorted({name for e in with_events for name in e})
        out["events_per_episode"] = {name: float(np.mean([e.get(name, 0) for e in with_events]))
                                     for name in names}
    crashed = [r["last_phase"] for r in records
               if "last_phase" in r and (r.get("end") == "error" or r.get("sim_error"))]
    if crashed:
        values, counts = np.unique(crashed, return_counts=True)
        out["error_phase_hist"] = {str(int(v)): int(c) for v, c in zip(values, counts)}
    reasons = [r["failure_reason"] for r in records if r.get("failure_reason")]
    if reasons:
        values, counts = np.unique(reasons, return_counts=True)
        out["failure_share"] = {str(v): float(c / n) for v, c in zip(values, counts)}
    cutoffs = [r["cutoff"] for r in records if "cutoff" in r]
    if cutoffs:
        values, counts = np.unique(cutoffs, return_counts=True)
        out["cutoff_share"] = {str(int(v)): float(c / len(cutoffs)) for v, c in zip(values, counts)}
    return out


def progress_line(summary: Mapping[str, Any]) -> Optional[str]:
    """The console's second line: reach rates and the task metrics, when there are any."""
    parts: List[str] = []
    reach = summary.get("reach")
    if reach:
        rates = " ".join(f"{k} {'-' if v is None else f'{v:.2f}'}" for k, v in reach.items())
        parts.append(f"reach from insertion (last {summary['window_episodes']}): {rates}")
    if "error_rate" in summary and (reach or summary["error_rate"] > 0):
        parts.append(f"sim errors {100 * summary['error_rate']:.1f}%")
    if "shield_delta_mean" in summary:
        parts.append(f"shield {summary['shield_delta_mean']:.3f}")
    if "catheter_moving_frac" in summary:
        parts.append(f"catheter moving {100 * summary['catheter_moving_frac']:.0f}%")
    if "d_p50" in summary:
        parts.append(f"d p50 {summary['d_p50']:.0f} mm")
    return "  " + " | ".join(parts) if parts else None
