"""The training schedule of round two (``training.schedule``; todo_list/two_devices_round2_changes.md
section G, user decisions 2026-11-10 16:19 and 16:22): which crutch is in use and when it is taken
away, driven by the evaluations. Off unless the config has ``training.schedule`` (then the trainer
behaves exactly as without it until the first evaluation).

Stages (``stage``)
------------------
1. ``curriculum``: the start curriculum moves its frontier towards the insertion point
   (``curriculum.py``); the demonstration quota stays at its start value. Over when the frontier is
   at the insertion point (or there is no start curriculum) and the merged success rate of the last
   two evaluations is at least ``pass_rate`` (0.5).
2. ``demo_ramp``: the demonstration quota goes linearly to 0 over ``demo_ramp_steps`` env steps
   (150 000; the user's 10-20 x 10^4), under the guard below. The demonstrations stay in the replay;
   they are just no longer guaranteed a share of the batch.
3. ``wrap_up``: training goes on until the second-tier success rate has not improved on its best
   for ``stop_patience`` (5) evaluations in a row; then the run stops (``stop``). With
   ``hint_ramp: true`` the optional experiment comes first:
4. ``hint_ramp`` (off by default): the hint rewards (progress and catheter-follow) go to 0 over
   ``hint_ramp_steps`` (200 000) under the guard; afterwards the stop rule as in 3.

Guard (every ramp, at every evaluation)
---------------------------------------
The rate compared is the merged success rate of the last two evaluations. At the ramp's start it is
the baseline. A drop of more than ``guard_drop`` (0.15) below the baseline, or a rate below
``guard_floor`` (0.30), triggers: the ramp stops, its value goes back to the value of the previous
evaluation and holds for ``hold_evals`` (3) evaluations; when the rate is back (not more than
``guard_drop`` below the baseline and at least ``guard_floor``) the ramp goes on at half speed; a
second trigger freezes the value where it is for good (the ramp is over).

Ladder (``ladder``; the start curriculum's frontier stuck)
---------------------------------------------------------
"Not learning" = the frontier grid k's last episodes succeed below ``stuck_rate`` (0.10) for
``stuck_evals`` (3) evaluations in a row, grid k + 1 above ``next_rate`` (0.50), and the frontier
has stayed at k for more than ``stuck_steps`` (200 000) env steps. Then level 1, the dedicated hint:
the reward's arch ruler (``reward_formula`` ``arch_ruler``: progress up the aorta to the hook point,
the plain path after it) is switched on, and off again once grid k passes ``next_rate``. Each level
has ``level_steps`` (200 000) env steps. Levels 2 (the action box around the advisor's action) and 3
(imitating the advisor) need an advisor that can say what the hand-written procedure would do from
any state; it is not built, so the ladder reports ``needs_advisor`` and stays at level 1.

Force factor (``force_start`` < 1; user 2026-11-13: "先轻后重", the force penalty light first)
----------------------------------------------------------------------------------------------
The force penalty's cap (``reward_formula`` ``force_cap``, both devices) is multiplied by
``force_scale``: ``force_start`` (e.g. 0.25) from the first batch on; once the merged success rate of
the last two evaluations reaches ``force_pass_rate`` (``pass_rate`` when not given) it climbs linearly
to 1 over ``force_ramp_steps`` (150 000) env steps under the same guard as the other ramps (a rollback
takes the factor back down to the previous evaluation's value; a second trigger freezes it where it
is, below 1). It runs beside the stages, not as one of them, so it may overlap the demonstration
ramp; the stop rule waits while it is still climbing. Only the replay's rewards (what the agent
learns from) are scaled; the env's own reward, the episode logs and the evaluations stay at the
config's full penalty. ``force_start: 1`` (the default) = no factor, nothing changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

STAGES = ("curriculum", "demo_ramp", "wrap_up", "hint_ramp", "done")


@dataclass
class Ramp:
    """A value going linearly from ``start`` to 0 over ``steps`` env steps, under the guard."""

    start: float
    steps: int
    value: float = 0.0
    speed: float = 1.0
    baseline: Optional[float] = None
    previous_value: float = 0.0
    hold: int = 0                     # evaluations still to hold after a trigger
    triggers: int = 0
    frozen: bool = False
    last_step: Optional[int] = None

    def __post_init__(self) -> None:
        self.value = self.previous_value = float(self.start)

    @property
    def over(self) -> bool:
        return self.frozen or self.value <= 0.0

    def on_evaluation(self, step: int, rate: float, drop: float, floor: float,
                      hold_evals: int) -> Optional[str]:
        """Advance or guard at an evaluation; returns what happened (for the events)."""
        if self.over:
            return None
        if self.baseline is None:                     # the ramp starts here
            self.baseline, self.last_step = float(rate), int(step)
            return "start"
        bad = rate < self.baseline - drop or rate < floor
        steps = int(step) - int(self.last_step)
        self.last_step = int(step)
        if self.hold > 0:
            self.hold -= 1
            if self.hold == 0 and not bad:
                self.speed *= 0.5
                return "resume_half_speed"
            if self.hold == 0:
                self.hold = 1                         # not back yet: keep holding
            return "hold"
        if bad:
            self.triggers += 1
            if self.triggers >= 2:
                self.frozen = True
                return "frozen"
            self.value = self.previous_value
            self.hold = int(hold_evals)
            return "rollback"
        self.previous_value = self.value
        self.value = max(0.0, self.value - self.start * self.speed * steps / max(self.steps, 1))
        return "advance"


@dataclass
class Schedule:
    """Stages, guard and ladder (module docstring)."""

    demo_fraction: float = 0.0                    # the quota's start value (training.demos.fraction)
    demo_ramp_steps: int = 150_000
    pass_rate: float = 0.5
    guard_drop: float = 0.15
    guard_floor: float = 0.30
    hold_evals: int = 3
    stop_patience: int = 5
    hint_ramp: bool = False
    hint_ramp_steps: int = 200_000
    ladder: bool = True
    stuck_rate: float = 0.10
    next_rate: float = 0.50
    stuck_evals: int = 3
    stuck_steps: int = 200_000
    level_steps: int = 200_000
    force_start: float = 1.0                      # factor on the force penalty until its ramp begins
    force_ramp_steps: int = 150_000
    force_pass_rate: Optional[float] = None       # merged rate that starts the force ramp (None: pass_rate)
    stage: str = "curriculum"
    rates: List[float] = field(default_factory=list)
    best: float = -1.0
    since_best: int = 0
    stop: bool = False
    demo: Optional[Ramp] = None
    hint: Optional[Ramp] = None
    force: Optional[Ramp] = None                  # what is still missing of the force factor (1 - factor)
    arch_hint: bool = False                       # ladder level 1 on
    ladder_level: int = 0
    ladder_grid: Optional[int] = None
    ladder_since: Optional[int] = None
    stuck_count: int = 0
    frontier_seen: Optional[int] = None
    frontier_since: int = 0

    KNOWN = ("demo_ramp_steps", "pass_rate", "guard_drop", "guard_floor", "hold_evals", "stop_patience",
             "hint_ramp", "hint_ramp_steps", "ladder", "stuck_rate", "next_rate", "stuck_evals",
             "stuck_steps", "level_steps", "force_start", "force_ramp_steps", "force_pass_rate")
    RAMPS = ("demo", "hint", "force")

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.force_start) <= 1.0:
            raise ValueError(f"training.schedule force_start must be in [0, 1], got {self.force_start!r}")

    @classmethod
    def from_config(cls, spec: Optional[Mapping[str, Any]], demo_fraction: float) -> Optional["Schedule"]:
        if not spec or spec.get("enabled") is False:
            return None
        unknown = set(spec) - set(cls.KNOWN) - {"enabled"}
        if unknown:
            raise KeyError(f"training.schedule: unknown key(s) {sorted(unknown)}; known: {sorted(cls.KNOWN)}")
        return cls(demo_fraction=float(demo_fraction), **{k: spec[k] for k in cls.KNOWN if k in spec})

    # ------------------------------------------------------------------ the knobs ----
    @property
    def demo_value(self) -> float:
        """The demonstration quota now."""
        return self.demo.value if self.demo is not None else self.demo_fraction

    @property
    def hint_scale(self) -> float:
        """Factor on the hint rewards' coefficients (progress, follow) now."""
        return self.hint.value if self.hint is not None else 1.0

    @property
    def force_scale(self) -> float:
        """Factor on the force penalty's cap now: ``force_start`` until its ramp begins, 1 at its end."""
        return 1.0 - self.force.value if self.force is not None else float(self.force_start)

    @property
    def force_climbing(self) -> bool:
        """The force factor's ramp has begun and is neither at 1 nor frozen."""
        return self.force is not None and not self.force.over

    @property
    def stage_index(self) -> int:
        """The stage as a number (``STAGES`` order: 0 curriculum ... 4 done), for plotting."""
        return STAGES.index(self.stage)

    @property
    def guard_trips(self) -> int:
        """How often the guard stopped a ramp so far (all ramps)."""
        return sum(r.triggers for r in (self.demo, self.hint, self.force) if r is not None)

    # --------------------------------------------------------------- evaluations ----
    def merged(self) -> float:
        last = self.rates[-2:]
        return float(sum(last) / len(last))

    def on_evaluation(self, step: int, success_rate: float, curriculum=None) -> List[Dict[str, Any]]:
        """One evaluation's success rate (second tier): move the stages, guard the ramps, watch the
        ladder. Returns the changes, one dict each (the trainer logs them as events)."""
        self.rates.append(float(success_rate))
        rate = self.merged()
        out: List[Dict[str, Any]] = []
        if self.ladder and curriculum is not None:
            out += self._ladder(step, curriculum)
        if self.force_start < 1.0:                    # the force factor: beside the stages
            line = self.pass_rate if self.force_pass_rate is None else self.force_pass_rate
            if self.force is None and rate >= line:
                self.force = Ramp(start=1.0 - float(self.force_start), steps=self.force_ramp_steps)
            if self.force is not None:
                what = self.force.on_evaluation(step, rate, self.guard_drop, self.guard_floor, self.hold_evals)
                if what:
                    out.append({"what": f"force_ramp_{what}", "step": int(step), "rate": rate,
                                "force_scale": self.force_scale})
        if self.stage == "curriculum":
            done = curriculum is None or curriculum.frontier <= 0
            if done and rate >= self.pass_rate:
                out.append(self._go("demo_ramp", step, rate))
                self.demo = Ramp(start=self.demo_fraction, steps=self.demo_ramp_steps)
        if self.stage == "demo_ramp":
            what = self.demo.on_evaluation(step, rate, self.guard_drop, self.guard_floor, self.hold_evals)
            if what:
                out.append({"what": f"demo_ramp_{what}", "step": int(step), "rate": rate,
                            "demo_fraction": self.demo.value})
            if self.demo.over:
                out.append(self._go("hint_ramp" if self.hint_ramp else "wrap_up", step, rate))
                if self.hint_ramp:
                    self.hint = Ramp(start=1.0, steps=self.hint_ramp_steps)
                self.best, self.since_best = -1.0, 0
        if self.stage == "hint_ramp":
            what = self.hint.on_evaluation(step, rate, self.guard_drop, self.guard_floor, self.hold_evals)
            if what:
                out.append({"what": f"hint_ramp_{what}", "step": int(step), "rate": rate,
                            "hint_scale": self.hint.value})
            if self.hint.over:
                out.append(self._go("wrap_up", step, rate))
                self.best, self.since_best = -1.0, 0
        if self.stage == "wrap_up":
            if success_rate > self.best:
                self.best, self.since_best = float(success_rate), 0
            else:
                self.since_best += 1
            if self.since_best >= self.stop_patience and not self.force_climbing:
                self.stop = True
                out.append(self._go("done", step, rate, best=self.best))
        return out

    def _go(self, stage: str, step: int, rate: float, **extra: Any) -> Dict[str, Any]:
        before, self.stage = self.stage, stage
        return {"what": "stage", "from": before, "to": stage, "step": int(step), "rate": rate, **extra}

    def _ladder(self, step: int, curriculum) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        k = int(curriculum.frontier)
        if k != self.frontier_seen:
            self.frontier_seen, self.frontier_since, self.stuck_count = k, int(step), 0
        rates = curriculum.rates()
        here, ahead = rates.get(str(k)), rates.get(str(k + 1))
        if self.arch_hint and self.ladder_grid is not None:
            passed = rates.get(str(self.ladder_grid))
            if passed is not None and passed > self.next_rate or k < self.ladder_grid:
                self.arch_hint, self.ladder_level = False, 0
                out.append({"what": "ladder_off", "step": int(step), "grid": self.ladder_grid})
                self.ladder_grid = None
            elif step - int(self.ladder_since) > self.level_steps and self.ladder_level == 1:
                self.ladder_level = 2
                out.append({"what": "ladder_needs_advisor", "step": int(step), "grid": self.ladder_grid,
                            "note": "levels 2-3 (action box, imitation) need the advisor, not built"})
            return out
        if k <= 0 or here is None or ahead is None:
            return out
        self.stuck_count = self.stuck_count + 1 if here < self.stuck_rate else 0
        if (self.stuck_count >= self.stuck_evals and ahead > self.next_rate
                and step - self.frontier_since > self.stuck_steps):
            self.arch_hint, self.ladder_level, self.ladder_grid, self.ladder_since = True, 1, k, int(step)
            out.append({"what": "ladder_level_1", "step": int(step), "grid": k, "rate": here,
                        "rate_next": ahead})
        return out

    # ---------------------------------------------------------------- checkpoints ----
    def state_dict(self) -> Dict[str, Any]:
        state = asdict(self)
        for name in self.RAMPS:
            ramp = getattr(self, name)
            state[name] = asdict(ramp) if ramp is not None else None
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, value in state.items():
            if key in self.RAMPS:
                if value is None:
                    setattr(self, key, None)
                else:
                    ramp = Ramp(start=value["start"], steps=value["steps"])
                    for k, v in value.items():
                        setattr(ramp, k, v)
                    setattr(self, key, ramp)
            elif hasattr(self, key):
                setattr(self, key, value)
