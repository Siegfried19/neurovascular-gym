"""The two-device task's reward, written once (round two; work log 2026-11-09 20:36-21:33).

Every step the env hands out a vector of **raw materials** (``FIELDS``: what the referee and the force
labels measured, in mm and N) and the reward is one pure numpy function of that vector and the
coefficients (``compute``). The env calls it for the live reward (``TwoDeviceFormula``), the replay
calls it again when a batch is sampled, with the coefficients of that day: changing a coefficient or
switching a reserve on never means clearing the replay (user decision 2026-11-06 00:30).

The formula (``Coefficients``, the user's numbers of 2026-11-09)::

    r = progress * progress_mm            the nearer tip's path to the wire target got this much shorter
      + follow   * follow_mm              the catheter's path to F got this much shorter (0 past F)
      - force(cath_force_N; cath_knee 2 N, cath_width 3 N)     cap * min(1, ((F - knee)+ / width)^2)
      - force(wire_force_N; wire_knee 0.5 N, wire_width 1 N)   (cap 0.1 per device and step)
      - step_penalty                      0.002
      - wrong_vessel * wrong_vessel       0.1 per entry
      + tier1 * tier1 + tier2 * success   +1 each (the second tier ends the episode)
      - failure * hard_failure            -1 (a tip outside the vessel)
      - action_change * action_change     0 by default (see below)

A simulation-error step (``sim_error``) gets 0: the trainers drop it anyway. Both progress terms
are potential differences, so they telescope: the sum over a run is the distance covered, whatever
the path.

``progress_mode`` (user 2026-11-12, work log 11:05 / 11:20): ``potential`` (the default, as above) or
``best``: both progress terms pay only for **new** progress, the part of a step's shortening that
goes past the best (shortest) path this episode has reached so far; going back, overshooting a
mouth and coming back, or pulling the wire into the catheter costs nothing (only the step penalty).
It removes the dip in front of a side branch's mouth (the plain ruler's local maximum there: on the
parent vessel every move but turning into the mouth lengthens the path) without any technique or
vessel written in; an episode's progress terms still add up to at most the whole path. The
materials carry both (``progress_new_mm`` / ``follow_new_mm``, the running-best excess), and
``upgrade_materials`` derives them for episodes recorded before (they are fixed by the per-step
shortenings of the episode: cumulative sum, then its running maximum).

Reserves, off by default: ``arch_ruler`` measures the progress terms with the hook-point ruler
(``hook_progress_mm`` / ``hook_follow_mm``: the way left to H = O + 30 mm on the aorta, then on from
H; each tip switches to the plain path once it reached H or went past the trunk's mouth,
``PhaseState.hook_paths``; the dedicated hint of the ladder, work log 2026-11-09 18:49, fixed
2026-11-10 21:56); ``lead_soft_min_mm`` scales the
second tier's +1 by ``lead_soft_factor`` when the wire leads by less than that at the finish (work
log 2026-11-09 21:33). Their materials are stored, so switching them on needs no new episodes.

``action_change`` (user 2026-11-13, work log 07:1x: "那个抗抖动的你加一下"): a smoothness penalty
on the executed motor command. The material ``action_change`` is the mean over the action's
dimensions of (e_t - e_{t-1})^2, where e is the executed action as the observation's
``last_actions`` term shows it (velocity / limit, in [-1, 1]; newest first in its memory block): the
step's change is read off two consecutive observations, so the policy, the world model's reward
head and the planner all see exactly what is penalised (the previous action is in the state). At
the first step e_{t-1} is what the observation showed at the reset (zeros; after a scripted prefix
the script's last command). Full-speed back-and-forth (+1, -1, +1, ...) gives 4 per step, steady
motion 0. ``upgrade_materials`` derives it for episodes recorded before from their observations.

``progress_ruler`` (``nearest`` / ``wire_after_trunk``): which tip the wire-target ruler follows;
``nearest`` (the tip with the shorter path, the whole way) is the recommended default pending the
user's choice (甲 / 乙, work log 2026-11-09 20:36).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..registry import register
from .phases import PhaseDetector, PhaseState

FIELDS = ("progress_mm", "follow_mm", "hook_progress_mm", "hook_follow_mm", "wire_force_N",
          "cath_force_N", "wrong_vessel", "tier1", "success", "hard_failure", "sim_error", "lead_mm",
          "progress_new_mm", "follow_new_mm", "action_change")
ROUND_TWO_FIELDS = 12                 # the materials before progress_new_mm / follow_new_mm (2026-11-12)
PROGRESS_NEW_FIELDS = 14              # the materials before action_change (2026-11-13)
DEFAULT_ACTION_TERM = "last_actions"  # the observation term the executed action is read from
PROGRESS_MODES = ("potential", "best")
INDEX = {name: i for i, name in enumerate(FIELDS)}
RULERS = ("nearest", "wire_after_trunk")
DEFAULT_FORCE_LABELS = ("distal_force", "catheter_tip_force")


@dataclass(frozen=True)
class Coefficients:
    progress: float = 0.01            # per mm
    follow: float = 0.01              # per mm
    force_cap: float = 0.1            # per device and step, at and beyond knee + width
    cath_knee_N: float = 2.0
    cath_width_N: float = 3.0
    wire_knee_N: float = 0.5
    wire_width_N: float = 1.0
    step_penalty: float = 0.002
    wrong_vessel: float = 0.1
    tier1: float = 1.0
    tier2: float = 1.0
    failure: float = 1.0
    arch_ruler: bool = False          # reserve: the hook-point ruler for the progress terms
    lead_soft_min_mm: Optional[float] = None   # reserve: scale the second tier when the lead is short
    lead_soft_factor: float = 0.8
    progress_mode: str = "potential"  # "best": only new progress pays (module docstring)
    action_change: float = 0.0        # per unit of mean squared change of the executed action

    def __post_init__(self) -> None:
        if self.progress_mode not in PROGRESS_MODES:
            raise ValueError(f"progress_mode must be one of {PROGRESS_MODES}, got {self.progress_mode!r}")
        if self.progress_mode == "best" and self.arch_ruler:
            raise ValueError("progress_mode 'best' measures the plain ruler; switch arch_ruler off")

    @classmethod
    def from_config(cls, spec: Optional[Mapping[str, Any]]) -> "Coefficients":
        """From the config's ``reward_formula`` entry (its ``type`` and ``progress_ruler`` and
        ``force_labels`` are the env part's, not coefficients)."""
        spec = dict(spec or {})
        for key in ("type", "progress_ruler", "force_labels", "name", "action_term"):
            spec.pop(key, None)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(spec) - known)
        if unknown:
            raise KeyError(f"reward_formula: unknown coefficient(s) {unknown}; known: {sorted(known)}")
        return cls(**spec)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def force_penalty(force_N: np.ndarray, knee_N: float, width_N: float, cap: float) -> np.ndarray:
    """0 below the knee, rising with the square of the excess, ``cap`` from knee + width on."""
    excess = np.maximum(np.asarray(force_N, dtype=np.float64) - knee_N, 0.0) / max(width_N, 1e-9)
    return cap * np.minimum(excess * excess, 1.0)


def _progress_columns(col, c: Coefficients):
    if c.progress_mode == "best":
        return col("progress_new_mm"), col("follow_new_mm")
    if c.arch_ruler:
        return col("hook_progress_mm"), col("hook_follow_mm")
    return col("progress_mm"), col("follow_mm")


def new_progress(shortenings: np.ndarray) -> np.ndarray:
    """Per-step new progress of one episode from its per-step shortenings (mm, in order): how far
    each step took the cumulative shortening past its best so far (the start counts as 0)."""
    cumulative = np.cumsum(np.asarray(shortenings, dtype=np.float64))
    best = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    return np.diff(best, prepend=0.0)


def action_slice_from_names(observation_names: Sequence[str], action_dim: int,
                            term: str = DEFAULT_ACTION_TERM) -> Optional[Tuple[int, int]]:
    """(start, stop) of the newest executed action inside the flat observation: the first
    ``action_dim`` entries of the ``term`` block (``Memory`` puts the newest first). None when the
    observation has no such term."""
    prefix = f"{term}["
    where = [i for i, name in enumerate(observation_names) if str(name).startswith(prefix)]
    if not where:
        return None
    if len(where) < action_dim or where != list(range(where[0], where[0] + len(where))):
        raise ValueError(f"observation term {term!r} is not a contiguous block of at least {action_dim} values")
    return where[0], where[0] + int(action_dim)


def action_change_of(previous: np.ndarray, current: np.ndarray) -> float:
    """Mean over the dimensions of the squared change of the executed action."""
    d = np.asarray(current, dtype=np.float64).reshape(-1) - np.asarray(previous, dtype=np.float64).reshape(-1)
    return float(np.mean(d * d))


def action_changes(observations: np.ndarray, action_slice: Tuple[int, int]) -> np.ndarray:
    """Per-step ``action_change`` of one episode from its observations ((steps + 1) x dim)."""
    obs = np.asarray(observations, dtype=np.float64)
    a = obs[:, action_slice[0]:action_slice[1]]
    d = a[1:] - a[:-1]
    return np.mean(d * d, axis=1)


def upgrade_materials(materials: np.ndarray, observations: Optional[np.ndarray] = None,
                      action_slice: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """One episode's materials (steps x fields) with the fields added since, derived from the stored
    ones: ``progress_new_mm`` / ``follow_new_mm`` from ``progress_mm`` / ``follow_mm``;
    ``action_change`` from the episode's observations ((steps + 1) x dim) at ``action_slice``
    (``action_slice_from_names``). Current materials come back unchanged."""
    m = np.asarray(materials, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"materials of one episode are (steps, fields), got shape {m.shape}")
    if m.shape[1] == len(FIELDS):
        return m
    if m.shape[1] not in (ROUND_TWO_FIELDS, PROGRESS_NEW_FIELDS):
        raise ValueError(f"materials have {m.shape[1]} fields; expected {len(FIELDS)}, "
                         f"{PROGRESS_NEW_FIELDS} or {ROUND_TWO_FIELDS}")
    if m.shape[1] == ROUND_TWO_FIELDS:
        extra = np.stack([new_progress(m[:, INDEX["progress_mm"]]), new_progress(m[:, INDEX["follow_mm"]])], axis=1)
        m = np.concatenate([m, extra.astype(np.float32)], axis=1)
    if observations is None or action_slice is None:
        raise ValueError("materials recorded before action_change (2026-11-13) need the episode's "
                         "observations and the executed action's place in them (action_slice)")
    obs = np.asarray(observations)
    if obs.shape[0] != m.shape[0] + 1:
        raise ValueError(f"an episode of {m.shape[0]} steps has {m.shape[0] + 1} observations, got {obs.shape[0]}")
    change = action_changes(obs, action_slice).astype(np.float32)
    return np.concatenate([m, change[:, None]], axis=1)


def compute(materials: np.ndarray, coefficients: Coefficients) -> np.ndarray:
    """Rewards of shape ``materials.shape[:-1]`` from a (..., len(FIELDS)) materials array."""
    m = np.asarray(materials, dtype=np.float64)
    if m.shape[-1] != len(FIELDS):
        raise ValueError(f"materials have {m.shape[-1]} fields, expected {len(FIELDS)}")
    c = coefficients
    col = lambda name: m[..., INDEX[name]]
    progress, follow = _progress_columns(col, c)
    tier2 = col("success")
    if c.lead_soft_min_mm is not None:
        tier2 = tier2 * np.where(col("lead_mm") < c.lead_soft_min_mm, c.lead_soft_factor, 1.0)
    r = (c.progress * progress + c.follow * follow
         - force_penalty(col("cath_force_N"), c.cath_knee_N, c.cath_width_N, c.force_cap)
         - force_penalty(col("wire_force_N"), c.wire_knee_N, c.wire_width_N, c.force_cap)
         - c.step_penalty - c.wrong_vessel * col("wrong_vessel")
         + c.tier1 * col("tier1") + c.tier2 * tier2 - c.failure * col("hard_failure")
         - c.action_change * col("action_change"))
    return np.where(col("sim_error") > 0, 0.0, r)


COMPONENTS = ("progress", "follow", "force_cath", "force_wire", "step", "wrong_vessel", "tier1", "tier2",
              "failure", "smooth")


def components(materials: np.ndarray, coefficients: Coefficients) -> Dict[str, np.ndarray]:
    """The reward split into its terms (each of shape ``materials.shape[:-1]``; penalties negative),
    for the per-episode reward breakdown (docs/wandb_metrics.md, section 3). They add up to
    ``compute``; a simulation-error step contributes 0 to every term."""
    m = np.asarray(materials, dtype=np.float64)
    if m.shape[-1] != len(FIELDS):
        raise ValueError(f"materials have {m.shape[-1]} fields, expected {len(FIELDS)}")
    c = coefficients
    col = lambda name: m[..., INDEX[name]]
    progress, follow = _progress_columns(col, c)
    tier2 = col("success")
    if c.lead_soft_min_mm is not None:
        tier2 = tier2 * np.where(col("lead_mm") < c.lead_soft_min_mm, c.lead_soft_factor, 1.0)
    live = col("sim_error") <= 0
    parts = {
        "progress": c.progress * progress, "follow": c.follow * follow,
        "force_cath": -force_penalty(col("cath_force_N"), c.cath_knee_N, c.cath_width_N, c.force_cap),
        "force_wire": -force_penalty(col("wire_force_N"), c.wire_knee_N, c.wire_width_N, c.force_cap),
        "step": -c.step_penalty * np.ones_like(progress), "wrong_vessel": -c.wrong_vessel * col("wrong_vessel"),
        "tier1": c.tier1 * col("tier1"), "tier2": c.tier2 * tier2, "failure": -c.failure * col("hard_failure"),
        "smooth": -c.action_change * col("action_change"),
    }
    return {name: np.where(live, value, 0.0) for name, value in parts.items()}


def component_sums(materials: np.ndarray, coefficients: Coefficients) -> Dict[str, float]:
    """Each term summed over an episode's steps."""
    return {name: float(np.sum(value)) for name, value in components(materials, coefficients).items()}


def ruler_potential(state: PhaseState, ruler: str, hook: bool = False) -> Tuple[float, float]:
    """(-path of the wire-target ruler, -path to F) for a referee state: the potentials."""
    wire, catheter, to_F = state.hook_paths() if hook else state.progress_paths()
    if ruler == "nearest":
        progress = min(wire, catheter)
    elif ruler == "wire_after_trunk":
        progress = wire if state.depth_bct_w >= 0.0 else min(wire, catheter)
    else:
        raise ValueError(f"progress_ruler must be one of {RULERS}, got {ruler!r}")
    return -progress, -to_F


@register("reward")
class TwoDeviceFormula:
    """The env part: materials and reward each step, after the labels (``GuidewireEnv.step``), so the
    forces are this step's. Not a stEVE ``Reward`` (those run before the labels); the config's
    ``reward`` list carries ``ZeroReward`` and the builder hands this to the env (``reward_formula``)."""

    def __init__(self, phase_detector: PhaseDetector, progress_ruler: str = "nearest",
                 force_labels: Sequence[str] = DEFAULT_FORCE_LABELS,
                 action_term: str = DEFAULT_ACTION_TERM, **coefficients: Any) -> None:
        if progress_ruler not in RULERS:
            raise ValueError(f"progress_ruler must be one of {RULERS}, got {progress_ruler!r}")
        self.phase_detector = phase_detector
        self.progress_ruler = progress_ruler
        self.force_labels = tuple(force_labels)
        if len(self.force_labels) != 2:
            raise ValueError("force_labels: the wire's and the catheter's tip-force label names")
        self.coefficients = Coefficients.from_config(coefficients)
        self.action_term = str(action_term)   # where the env reads the executed action (action_change)
        self.potentials: Optional[Tuple[float, float, float, float]] = None
        self.best: Optional[Tuple[float, float]] = None   # the best (highest) plain potentials so far
        self.materials = np.zeros(len(FIELDS), dtype=np.float32)
        self.reward = 0.0

    def _potentials(self, state: PhaseState) -> Tuple[float, float, float, float]:
        plain = ruler_potential(state, self.progress_ruler)
        hooked = ruler_potential(state, self.progress_ruler, hook=True)
        return plain[0], plain[1], hooked[0], hooked[1]

    def reset(self, episode_nr: int = 0) -> None:
        self.potentials = self._potentials(self.phase_detector.state)
        self.best = (self.potentials[0], self.potentials[1])
        self.materials = np.zeros(len(FIELDS), dtype=np.float32)
        self.reward = 0.0

    def step(self, label_values: Mapping[str, float], action_change: float = 0.0) -> Tuple[float, np.ndarray]:
        """``action_change``: this step's change of the executed action (``action_change_of`` on the
        observation's newest action before and after the step; the env reads it)."""
        state = self.phase_detector.state
        m = np.zeros(len(FIELDS), dtype=np.float64)
        if state.sim_error:
            m[INDEX["sim_error"]] = 1.0            # the potentials stay where they were
        else:
            if self.potentials is None:
                self.potentials = self._potentials(state)
            if self.best is None:
                self.best = (self.potentials[0], self.potentials[1])
            now = self._potentials(state)
            m[INDEX["progress_mm"]] = now[0] - self.potentials[0]
            m[INDEX["follow_mm"]] = now[1] - self.potentials[1]
            m[INDEX["progress_new_mm"]] = max(0.0, now[0] - self.best[0])
            m[INDEX["follow_new_mm"]] = max(0.0, now[1] - self.best[1])
            self.best = (max(self.best[0], now[0]), max(self.best[1], now[1]))
            m[INDEX["hook_progress_mm"]] = now[2] - self.potentials[2]
            m[INDEX["hook_follow_mm"]] = now[3] - self.potentials[3]
            self.potentials = now
            m[INDEX["wire_force_N"]] = float(label_values.get(self.force_labels[0], 0.0) or 0.0)
            m[INDEX["cath_force_N"]] = float(label_values.get(self.force_labels[1], 0.0) or 0.0)
            m[INDEX["wrong_vessel"]] = float(state.events.get("wrong_vessel", 0))
            m[INDEX["tier1"]] = float(state.events.get("tier1", 0))
            m[INDEX["success"]] = float(bool(state.success))
            m[INDEX["hard_failure"]] = float(bool(state.hard_failure))
            m[INDEX["lead_mm"]] = float(state.d)
            m[INDEX["action_change"]] = float(action_change)
        self.materials = m.astype(np.float32)
        self.reward = float(compute(m, self.coefficients))
        return self.reward, self.materials
