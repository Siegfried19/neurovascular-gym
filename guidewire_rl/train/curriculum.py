"""Which part of the task an episode practises: the start curriculum (round two,
``training.start_curriculum``), the cutoff mixture and the residual-box schedule
(``training.cutoff_mixture``, ``planning.residual``).

Start curriculum (``StartCurriculum``, work log 2026-11-06 16:02, 2026-11-09 22:22, 2026-11-10 14:56)
------------------------------------------------------------------------------------------------
Episodes start part-way: the hand-written procedure plays a prefix (two_device/prefix.py) up to a
drawn point of the task and the policy takes over there. The task's progress is one number, the
referee's ``progress_ratio`` (0 at the insertion point, 1 with both devices at their goals, by the
two reward rulers), cut into ``bins`` (10) grids: grid k covers [(k-1)/bins, k/bins), grid 0 is
"from the insertion point". Every episode: with probability ``baseline_fraction`` (0.3) it starts
from the insertion point; otherwise a grid is drawn by the weights below and a ratio uniformly
inside it. Weights: the frontier grid (the earliest grid still being learned) gets the bulk, every
grid past it (closer to the end, already learned) a floor of ``floor_weight`` (0.05), every grid
before it 0. The frontier starts at the last grid and moves one grid towards the insertion point
whenever the episodes started in it (at least ``min_episodes`` in the last ``window_episodes``)
succeed at ``frontier_threshold`` (0.5) or better; checked every ``update_every`` env steps. Past
grid 1 the curriculum is over: everything starts from the insertion point. The grid an episode is
booked under is the one it really started in (the prefix may stop short: the script gave up).

``supply: self`` (user 2026-11-12, work log 14:xx): no script at all. The starts are states the policy
reached on its own in earlier training episodes (``start_library.StartLibrary``: each episode saves one
state per grid it reaches first). Only grids holding at least ``library_min`` states can be drawn; the
grid trained is the highest such grid at or below the frontier (``training_grid``; none yet: every
episode starts from the insertion point), the grids past the frontier that have states keep the
floor, and the frontier moves back from the grid trained (to one below it) by the same success rule.
So the curriculum starts where the policy's best episodes got to (just past its bottleneck, as soon as
some episode crossed it) and walks back to the insertion point as those starts are learned.

``supply: demos`` (user 2026-11-12, work log 19:26): the starts are states saved along the scripted
demonstrations, made once beforehand (``scripts/record_demos.py --starts``; ``library`` names the folder,
``start_library.StartLibrary`` in its fixed mode: nothing is added or deleted during training). The
drawing is the same as with ``self``; such a library covers every grid from the first episode, so the
grid trained is the frontier itself and the curriculum walks back from the last grid as with the
prefix, without a script in the training loop.

With either library (``self``, ``demos``) an episode started from a saved state is booked under the
grid drawn, the grid the state was saved in (user 2026-11-13; ``Trainer._absorb_reset``): a state is
saved on entering its grid, so it sits on the grid's lower edge, and the progress re-measured against
another episode's target moves it by about 0.01; booking by that would put about half of the frontier's
episodes into the grid below it.

``grid: landmarks`` (user decisions 2026-11-10 14:56, 22:24): the grids are the referee's
``landmark_grid`` instead (milestones generated from the route's structure, two_device/phases.py; on
this route nine milestones, grids 1-10), so ``bins`` must be the env's ``landmark_grids`` (checked on
the first reset that reports it). A drawn grid k is handed over at the step the prefix enters it
(``landmark_grid >= k``); everything else (baseline, weights, frontier, booking) is the same.

The numbers are the user's to review (todo_list/two_devices_round2_changes.md, section I).

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
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

SUCCESS_PHASE = 5            # phases 0-4; "reaching 5" means success
SUPPLIES = ("prefix", "self", "demos")   # scripted prefix / the policy's own start library / states saved
LIBRARY_SUPPLIES = ("self", "demos")     # along the demonstrations; the last two start from start_library.py
GRIDS = {"distance": "progress_ratio", "landmarks": "landmark_grid"}   # start grid -> the info key it reads


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


class StartCurriculum:
    """Draws where each episode starts and moves the frontier (module docstring)."""

    def __init__(self, *, bins: int = 10, baseline_fraction: float = 0.3,
                 frontier_threshold: float = 0.5, min_episodes: int = 20, window_episodes: int = 100,
                 update_every: int = 25_000, floor_weight: float = 0.05, prefix_max_steps: int = 600,
                 supply: str = "prefix", grid: str = "distance", seed: int = 0,
                 library_cap: int = 100, library_min: int = 5, library: Optional[str] = None) -> None:
        if bins < 1:
            raise ValueError("bins must be at least 1")
        if not 0.0 <= baseline_fraction <= 1.0:
            raise ValueError("baseline_fraction must be in [0, 1]")
        if supply not in SUPPLIES:
            raise ValueError(f"start_curriculum.supply must be one of {SUPPLIES}, got {supply!r}")
        if supply in LIBRARY_SUPPLIES and grid != "distance":
            raise ValueError(f"start_curriculum.supply {supply!r} grids by distance (the library's grids)")
        if supply == "demos" and not library:
            raise ValueError("start_curriculum.supply 'demos' needs library: the folder of saved states "
                             "(scripts/record_demos.py --starts)")
        if supply != "demos" and library:
            raise ValueError("start_curriculum.library is only read with supply 'demos'")
        if grid not in GRIDS:
            raise ValueError(f"start_curriculum.grid must be one of {GRIDS}, got {grid!r}")
        self.bins = int(bins)
        self.baseline_fraction = float(baseline_fraction)
        self.frontier_threshold = float(frontier_threshold)
        self.min_episodes = int(min_episodes)
        self.update_every = int(update_every)
        self.floor_weight = float(floor_weight)
        self.prefix_max_steps = int(prefix_max_steps)
        self.supply, self.grid = supply, grid
        self.library_cap, self.library_min = int(library_cap), int(library_min)   # read by StartLibrary
        self.library = None if library is None else str(library)                  # supply 'demos': its folder
        self.available: Optional[Set[int]] = None      # library supplies: grids the library can start from
        self.frontier = self.bins                      # the grid being learned; 0 = curriculum over
        self.windows: Dict[int, Deque[bool]] = {k: deque(maxlen=int(window_episodes))
                                                for k in range(self.bins + 1)}
        self.rng = np.random.default_rng(seed)
        self.next_update = self.update_every
        self.moves: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, spec: Optional[Mapping[str, Any]], seed: int) -> Optional["StartCurriculum"]:
        if not spec or spec.get("enabled") is False:
            return None
        known = {"bins", "baseline_fraction", "frontier_threshold", "min_episodes", "window_episodes",
                 "update_every", "floor_weight", "prefix_max_steps", "supply", "grid", "enabled",
                 "library_cap", "library_min", "library"}
        unknown = set(spec) - known
        if unknown:
            raise KeyError(f"training.start_curriculum: unknown key(s) {sorted(unknown)}")
        return cls(seed=seed, **{k: spec[k] for k in known - {"enabled"} if k in spec})

    @property
    def prefix_key(self) -> str:
        """The info key the prefix runs up to: the progress ratio or the landmark grid."""
        return GRIDS[self.grid]

    def grid_of(self, ratio: float) -> int:
        """The grid of a progress ratio (1..bins), or of a landmark grid as reached; 0 is the
        insertion point."""
        if ratio <= 0.0:
            return 0
        if self.grid == "landmarks":
            return int(min(self.bins, max(1, int(round(ratio)))))
        return int(min(self.bins, int(np.floor(ratio * self.bins)) + 1))

    def check_grids(self, info: Mapping[str, Any]) -> None:
        """With the landmark grid, ``bins`` must be the number of grids the env's milestones make."""
        grids = info.get("landmark_grids")
        if self.grid == "landmarks" and grids and int(grids) != self.bins:
            raise ValueError(f"training.start_curriculum.bins is {self.bins} but the route's milestones make "
                             f"{int(grids)} landmark grids; set bins: {int(grids)}")

    @property
    def uses_library(self) -> bool:
        """The starts are saved states (supply 'self' or 'demos'), not a scripted prefix."""
        return self.supply in LIBRARY_SUPPLIES

    def set_available(self, grids) -> None:
        """Library supplies: the grids the start library can start from now (StartLibrary.available)."""
        self.available = {int(k) for k in grids}

    @property
    def training_grid(self) -> int:
        """The grid the episodes are drawn at: the frontier, or with a start library the highest grid at
        or below it the library can start from (0: none yet, everything from the insertion point)."""
        if not self.uses_library:
            return self.frontier
        return max((k for k in (self.available or ()) if 0 < k <= self.frontier), default=0)

    def weights(self) -> np.ndarray:
        """Probabilities over grids 0..bins (0 = insertion point)."""
        w = np.zeros(self.bins + 1)
        front = self.training_grid
        if front <= 0:
            w[0] = 1.0
            return w
        w[0] = self.baseline_fraction
        rest = 1.0 - self.baseline_fraction
        passed = list(range(self.frontier + 1, self.bins + 1))
        if self.uses_library:
            passed = [k for k in passed if k in (self.available or ())]
        floor = min(self.floor_weight, rest / max(len(passed) + 1, 1))
        for k in passed:
            w[k] = floor
        w[front] = rest - floor * len(passed)
        return w

    def sample(self) -> Tuple[int, Optional[float]]:
        """(grid, ratio to hand over at); (0, None) = from the insertion point."""
        w = self.weights()
        k = int(self.rng.choice(len(w), p=w))
        if k == 0:
            return 0, None
        if self.uses_library:
            return k, None                             # the trainer picks a state of grid k from the library
        if self.grid == "landmarks":
            return k, float(k)
        low, high = (k - 1) / self.bins, k / self.bins
        return k, float(self.rng.uniform(low, high))

    def record(self, start_grid: int, success: bool) -> None:
        self.windows[int(max(0, min(start_grid, self.bins)))].append(bool(success))

    def rates(self) -> Dict[str, Optional[float]]:
        return {str(k): (float(np.mean(v)) if len(v) else None) for k, v in self.windows.items()}

    def maybe_update(self, step: int) -> Optional[Dict[str, Any]]:
        """Move the frontier when its grid is learned; returns the change (for the logs), else None."""
        if step < self.next_update or self.frontier <= 0:
            return None
        self.next_update = (step // self.update_every + 1) * self.update_every
        front = self.training_grid
        if front <= 0:
            return None
        window = self.windows[front]
        if len(window) >= self.min_episodes and float(np.mean(window)) >= self.frontier_threshold:
            before = self.frontier
            self.frontier = front - 1
            change = {"frontier_before": before, "frontier": self.frontier,
                      "rate": float(np.mean(window)), "episodes": len(window), "step": int(step)}
            self.moves.append(change)
            return change
        return None

    def weight_map(self) -> Dict[str, float]:
        return {str(k): float(w) for k, w in enumerate(self.weights())}

    def state_dict(self) -> Dict[str, Any]:
        return {"bins": self.bins, "frontier": self.frontier,
                "windows": {str(k): list(v) for k, v in self.windows.items()},
                "window_episodes": self.windows[0].maxlen, "next_update": self.next_update,
                "moves": list(self.moves), "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["bins"]) != self.bins:
            return                                     # a different grid: start the new one fresh
        self.frontier = int(state["frontier"])
        for k, values in state["windows"].items():
            self.windows[int(k)].extend(bool(v) for v in values)
        self.next_update = int(state["next_update"])
        self.moves = list(state.get("moves", []))
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
