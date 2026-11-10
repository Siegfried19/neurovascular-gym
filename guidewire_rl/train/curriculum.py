"""Which part of the task an episode practises: the cutoff mixture and the
residual-box schedule (``training.cutoff_mixture``, ``planning.residual``).

Cutoff mixture
--------------
Every episode starts at the insertion point. Before each reset the trainer draws
a cutoff ``k`` from ``phases`` (default 2, 3, 4, 5) and gives it to the env with
``env.set_cutoff(k)``; the env truncates the episode (value bootstrapped, not a
failure) as soon as its phase reaches ``k``. ``k = 5`` never cuts: the episode
runs to success, failure or the step limit.

Weights, recomputed every ``update_every`` env steps from the last
``window_episodes`` finished episodes:

1. reach rate of ``k``: among those episodes whose cutoff let them get there
   (cutoff >= k; an episode cut at 2 says nothing about phase 3), the fraction
   whose highest phase was >= k (for k = 5: that succeeded). No such episode
   yet: 0.
2. raw weight ``1 - reach rate``; normalized to sum 1 (all raw weights 0: equal).
3. floors: every cutoff gets at least ``min_weight``; the frontier gets at least
   ``frontier_min``. The frontier is the cutoff with the lowest reach rate; ties
   go to the earliest, so while several phases are never reached it is the
   earliest of them. The floors are met by fixing the cutoffs that fall below
   theirs and sharing what is left among the others in proportion to step 2.

Before the first update (empty window) this gives 0.4, 0.2, 0.2, 0.2. The draws
use their own random stream, so switching the mixture on or off does not move
any other random number of the run; the stream, the window and the weights are
saved in checkpoints.

Residual box
------------
``planning.residual``: while the env reports ``phase == 1`` together with a
``nominal_action`` (the scripted controller's suggestion; not when it says
``nominal_available: false``), the planner keeps
every action it samples or takes from the policy inside
``nominal +- delta`` (clipped to [-1, 1]). ``delta`` starts at ``delta`` and,
every ``grow_every_episodes`` finished episodes, grows by ``grow_step`` (up to
``max``) when, among those episodes that entered phase 1, at least
``grow_if_phase1_rate`` got through it (reached phase 2).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SUCCESS_PHASE = 5            # phases 0-4; "reaching 5" means success


def frontier_index(rates: Sequence[float]) -> int:
    """The cutoff with the lowest reach rate; ties -> the earliest."""
    rates = np.asarray(rates, dtype=np.float64)
    return int(np.flatnonzero(rates == rates.min())[0])


def mixture_weights(rates: Sequence[float], *, min_weight: float, frontier_min: float) -> np.ndarray:
    """Weights for cutoffs with these reach rates (see the module docstring)."""
    rates = np.clip(np.asarray(rates, dtype=np.float64), 0.0, 1.0)
    n = len(rates)
    if n == 0:
        raise ValueError("no cutoffs")
    floors = np.full(n, float(min_weight))
    floors[frontier_index(rates)] = max(float(min_weight), float(frontier_min))
    if floors.sum() > 1.0 + 1e-9:
        raise ValueError(f"min_weight and frontier_min ask for {floors.sum():.2f} > 1 in total")
    raw = 1.0 - rates
    share = raw / raw.sum() if raw.sum() > 0 else np.full(n, 1.0 / n)
    fixed = np.zeros(n, dtype=bool)
    while True:
        free = ~fixed
        budget = 1.0 - floors[fixed].sum()
        free_share = share[free].sum()
        weights = floors.copy()
        if free_share > 0:
            weights[free] = share[free] * budget / free_share
        else:
            weights[free] = budget / free.sum()
        below = free & (weights < floors - 1e-12)
        if not below.any():
            return weights
        fixed |= below


class CutoffMixture:
    """Draws each episode's cutoff and keeps the weights up to date."""

    def __init__(self, phases: Sequence[int] = (2, 3, 4, 5), *, min_weight: float = 0.1,
                 frontier_min: float = 0.4, window_episodes: int = 500,
                 update_every: int = 50_000, seed: int = 0) -> None:
        self.phases = [int(k) for k in phases]
        if not self.phases or sorted(set(self.phases)) != self.phases:
            raise ValueError(f"cutoff phases must be increasing and distinct, got {phases}")
        self.min_weight = float(min_weight)
        self.frontier_min = float(frontier_min)
        self.update_every = int(update_every)
        if self.update_every < 1:
            raise ValueError("update_every must be at least 1")
        self.window: Deque[Tuple[int, int, bool]] = deque(maxlen=int(window_episodes))
        self.rng = np.random.default_rng(seed)
        self.weights = mixture_weights(np.zeros(len(self.phases)), min_weight=self.min_weight,
                                       frontier_min=self.frontier_min)
        self.next_update = self.update_every
        self.updates = 0

    @classmethod
    def from_config(cls, spec: Optional[Mapping[str, Any]], seed: int) -> Optional["CutoffMixture"]:
        if not spec:
            return None
        if spec.get("enabled") is False:
            return None
        known = {"phases", "min_weight", "frontier_min", "window_episodes", "update_every", "enabled"}
        unknown = set(spec) - known
        if unknown:
            raise KeyError(f"training.cutoff_mixture: unknown key(s) {sorted(unknown)}")
        kwargs = {k: spec[k] for k in known - {"enabled"} if k in spec}
        return cls(seed=seed, **kwargs)

    @property
    def full_episode_cutoff(self) -> int:
        """The cutoff that never cuts (used for evaluation)."""
        return max(self.phases)

    def sample(self) -> int:
        return int(self.phases[int(self.rng.choice(len(self.phases), p=self.weights))])

    def record(self, cutoff: Optional[int], max_phase: int, success: bool) -> None:
        """A finished episode (any end, errors included)."""
        cutoff = self.full_episode_cutoff if cutoff is None else int(cutoff)
        self.window.append((cutoff, int(max_phase), bool(success)))

    def reach_rates(self) -> Tuple[List[float], List[int]]:
        """(rate, number of episodes it is computed over) per cutoff."""
        rates, counts = [], []
        for k in self.phases:
            pool = [(m, s) for c, m, s in self.window if c >= k]
            reached = [s or m >= k for m, s in pool] if k < SUCCESS_PHASE else [s for _, s in pool]
            rates.append(float(np.mean(reached)) if pool else 0.0)
            counts.append(len(pool))
        return rates, counts

    def maybe_update(self, step: int) -> Optional[Dict[str, Any]]:
        """Recompute the weights once ``step`` has passed the next multiple of
        ``update_every``; returns what changed (for the logs), else None."""
        if step < self.next_update:
            return None
        self.next_update = (step // self.update_every + 1) * self.update_every
        rates, counts = self.reach_rates()
        self.weights = mixture_weights(rates, min_weight=self.min_weight,
                                       frontier_min=self.frontier_min)
        self.updates += 1
        return {"weights": self.weight_map(), "reach": dict(zip(map(str, self.phases), rates)),
                "episodes": dict(zip(map(str, self.phases), counts)),
                "frontier": self.phases[frontier_index(rates)]}

    def weight_map(self) -> Dict[str, float]:
        return {str(k): float(w) for k, w in zip(self.phases, self.weights)}

    def state_dict(self) -> Dict[str, Any]:
        return {"phases": list(self.phases), "window": list(self.window),
                "window_episodes": self.window.maxlen, "weights": self.weights.tolist(),
                "next_update": self.next_update, "updates": self.updates,
                "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if list(state["phases"]) != self.phases:
            # different cutoffs: keep the new config's start weights, reuse the history only
            self.window.extend(tuple(x) for x in state["window"])
            return
        self.window.extend(tuple(x) for x in state["window"])
        self.weights = np.asarray(state["weights"], dtype=np.float64)
        self.next_update = int(state["next_update"])
        self.updates = int(state.get("updates", 0))
        self.rng.bit_generator.state = state["rng_state"]


class ResidualSchedule:
    """The residual box half-width ``delta`` and its growth (module docstring)."""

    def __init__(self, delta: float = 0.5, *, grow_every_episodes: int = 500,
                 grow_if_phase1_rate: float = 0.7, grow_step: float = 0.25,
                 max: float = 1.0) -> None:  # pylint: disable=redefined-builtin
        self.delta = float(delta)
        self.grow_every = int(grow_every_episodes)
        self.threshold = float(grow_if_phase1_rate)
        self.grow_step = float(grow_step)
        self.max_delta = float(max)
        self.block: List[int] = []          # max phases of the episodes since the last check
        self.history: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, spec: Optional[Mapping[str, Any]]) -> Optional["ResidualSchedule"]:
        if not spec or spec.get("enabled") is False:
            return None
        known = {"delta", "grow_every_episodes", "grow_if_phase1_rate", "grow_step", "max", "enabled"}
        unknown = set(spec) - known
        if unknown:
            raise KeyError(f"planning.residual: unknown key(s) {sorted(unknown)}")
        return cls(**{k: spec[k] for k in known - {"enabled"} if k in spec})

    def record(self, max_phase: int, success: bool) -> Optional[Dict[str, Any]]:
        """A finished episode; returns a log record when a check happened."""
        self.block.append(SUCCESS_PHASE if success else int(max_phase))
        if len(self.block) < self.grow_every:
            return None
        entered = [m for m in self.block if m >= 1]
        rate = float(np.mean([m >= 2 for m in entered])) if entered else 0.0
        before = self.delta
        if entered and rate >= self.threshold:
            self.delta = min(self.max_delta, self.delta + self.grow_step)
        self.block = []
        entry = {"phase1_pass_rate": rate, "entered_phase1": len(entered),
                 "delta_before": before, "delta": self.delta}
        self.history.append(entry)
        return entry

    def box_for(self, info: Mapping[str, Any]):
        """The box for the decision after this ``info`` (the env's last step or reset):
        its ``phase``, ``nominal_action`` and ``nominal_available`` (absent = available)."""
        nominal = info.get("nominal_action")
        if nominal is None or not info.get("nominal_available", True):
            return None
        return self.box(info.get("phase"), nominal)

    def box(self, phase: Optional[int], nominal: Optional[Sequence[float]]):
        """(low, high) for one env, or None when the box does not apply."""
        if phase is None or int(phase) != 1 or nominal is None:
            return None
        nominal = np.asarray(nominal, dtype=np.float32).reshape(-1)
        return (np.clip(nominal - self.delta, -1.0, 1.0), np.clip(nominal + self.delta, -1.0, 1.0))

    def state_dict(self) -> Dict[str, Any]:
        return {"delta": self.delta, "block": list(self.block), "history": list(self.history)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.delta = float(state["delta"])
        self.block = [int(m) for m in state.get("block", [])]
        self.history = list(state.get("history", []))
