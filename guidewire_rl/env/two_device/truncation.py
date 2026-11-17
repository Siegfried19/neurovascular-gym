"""Episode ends that cut an episode short (the value of the last state is bootstrapped).

``PhaseCutoff``: the episode stops once the referee's phase reaches the cutoff k. The trainer
sets k before each reset with ``env.set_cutoff(k)`` (or ``env.reset(options={"cutoff": k})``),
k in {2, 3, 4, 5}; 5 never stops it (only success, a hard failure, ``MaxSteps`` or ``SimError``
end the episode). The cutoff lives in the referee (``PhaseDetector.cutoff``), so ``info["cutoff"]``
shows the one in force; it stays until set again.

``TipForceBlowup``: a single contact pushing with more than ``limit_N`` (30 N) in any substep of the
step means the contact solve has blown up. Measured 2026-11-03 (hook corner on the wire's shaft): the
catheter tip's force went 0.6 -> 6 -> 38 -> 1360 -> 3109 N over consecutive steps, all finite numbers, so
``stop_on_nan`` saw nothing until the wire was thrown 15-21 mm out of the vessel. What is legitimate in
this setting is far below: over 52k validated steps (2026-11-07/08, scripted and random pushes, three
alarm distances) the largest tip force was 12 N (one random jab), the scripted procedure's 3.5 N, so 30 N
is 2.5x the largest seen and catches the runaway at its 38 N step. Such a step is the simulator's
failure (MuJoCo resets on accelerations beyond 1e10, PhysX caps the depenetration velocity; SOFA has no
such guard). It is marked as a simulation error (``simulation.simulation_error``, so ``SimError``, the
labels and the trainers treat it exactly like NaN: the step is dropped, the episode closed as an error)
and truncated the same step. It reads the tip-force monitor's ``peak_contact_N`` (the largest normal
multiplier of any contact in any substep, over all devices and their whole length, not only the 3 mm
tip windows), which is in by the time truncations are evaluated. ``ForceBlowupInfo`` reports the peak
as ``info["force_blowup_N"]`` (0 when none), to tell it from NaN.

``Stalled``: the episode stops once neither device's inserted length has changed by ``min_mm`` (net,
over the last ``window_steps`` steps): the policy has stopped, or rubs back and forth on the spot
(rotation alone does not count as moving). Such an episode crawls: devices pressed into a narrow vessel
and nearly still make the contact solver run to its iteration cap, 7-11 s per step against 0.3-2 s
(docs/simulation_speed.md, 2026-11-15), so two or three of them held a 20-episode evaluation for an
hour. Measured on 94 saved episodes of the 400k policy (``runs/round2_dagger_v3/final_400k``) with 100
steps / 3 mm: 33 of the 34 episodes that ran to the step limit end early (a third of all steps), and 3
of 60 successes that stood still for a while and then went on are ended too, so a success rate with
the rule on is 2-3 points lower than without. On the delivered policy under changed conditions (1251
episodes, 2026-11-16) 100 steps ended 43 of 1059 successes, 8 of 31 in one condition; 200 steps / 3 mm
ends 9 (at most 2 per condition) and still 150 of the 167 episodes that ran to the step limit, so the
config now says 200 (the class default stays 100). It is a truncation like the step limit (not a failure,
not a simulation error; the value of the last state is bootstrapped). Off unless asked for:
``always: true`` turns it on for every episode of this env (training episodes too);
``env.end_when_stalled()`` turns it on for the next episode only (the evaluations do this with
``evaluation.end_when_stalled``). ``StallInfo`` reports ``info["stalled"]``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Sequence

import numpy as np
from eve.info import Info
from eve.truncation import Truncation

from ..registry import register
from .phases import DEFAULT_CUTOFF, PhaseDetector


@register("truncation")
class PhaseCutoff(Truncation):
    def __init__(self, phase_detector: PhaseDetector, cutoff: int = DEFAULT_CUTOFF) -> None:
        self.phase_detector = phase_detector
        self.phase_detector.set_cutoff(cutoff)

    @property
    def cutoff(self) -> int:
        return int(self.phase_detector.cutoff)

    def set_cutoff(self, cutoff: int) -> None:
        self.phase_detector.set_cutoff(cutoff)

    @property
    def truncated(self) -> bool:
        state = self.phase_detector.state
        return (not state.sim_error) and state.phase >= self.cutoff

    def step(self) -> None:
        pass

    def reset(self, episode_nr: int = 0) -> None:
        pass


@register("truncation")
class TipForceBlowup(Truncation):
    """A contact-solve blow-up, read off the tip-force monitor's largest contact force (module docstring)."""

    def __init__(self, intervention, labels: Sequence[Any] = (), limit_N: float = 30.0,
                 label_names: Sequence[str] = ("distal_force", "catheter_tip_force")) -> None:
        self.intervention = intervention
        self.limit_N = float(limit_N)
        self.labels = [l for l in labels if getattr(l, "name", None) in label_names]
        if not self.labels:
            raise KeyError(f"TipForceBlowup: none of {list(label_names)} is in env.labels")
        self.last_blowup_N = 0.0

    def peak_N(self) -> float:
        """The largest single contact force of the current step so far (the monitor's running peak;
        a monitor without one: the largest 3 mm window sample)."""
        peak = 0.0
        for label in self.labels:
            monitor = getattr(label, "monitor", None)
            if monitor is None:
                continue
            value = getattr(monitor, "peak_contact_N", None)
            if value is None:
                samples = monitor.samples.get(label.instrument, [])
                value = float(np.max(samples)) if samples else 0.0
            peak = max(peak, float(value))
        return peak

    @property
    def truncated(self) -> bool:
        if self.last_blowup_N > 0.0:
            return True
        peak = self.peak_N()
        if peak > self.limit_N:
            self.last_blowup_N = peak
            self.intervention.simulation.simulation_error = True
            self.intervention.simulation.force_blowup_N = peak
            return True
        return False

    def step(self) -> None:
        self.last_blowup_N = 0.0
        self.intervention.simulation.force_blowup_N = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("info")
class ForceBlowupInfo(Info):
    """``info["force_blowup_N"]``: the contact force that tripped ``TipForceBlowup`` this step, else 0
    (the gate leaves it on the simulation object, ``force_blowup_N``)."""

    def __init__(self, intervention, name: str = "force_blowup") -> None:
        super().__init__(name)
        self.intervention = intervention

    @property
    def info(self) -> Dict[str, Any]:
        return {"force_blowup_N": float(getattr(self.intervention.simulation, "force_blowup_N", 0.0))}


@register("truncation")
class Stalled(Truncation):
    """Neither device's inserted length changed by ``min_mm`` over ``window_steps`` steps (module docstring)."""

    def __init__(self, intervention, window_steps: int = 100, min_mm: float = 3.0, always: bool = False) -> None:
        if int(window_steps) < 1 or float(min_mm) <= 0.0:
            raise ValueError("Stalled: window_steps >= 1 and min_mm > 0")
        self.intervention = intervention
        self.window_steps, self.min_mm, self.always = int(window_steps), float(min_mm), bool(always)
        self.requested = False                # ``end_when_stalled()``: on for the next episode
        self.active = self.always             # in force for the current episode
        self.history: Deque[np.ndarray] = deque(maxlen=self.window_steps + 1)
        self.stalled = False

    def end_when_stalled(self) -> None:
        self.requested = True

    def _inserted(self) -> np.ndarray:
        return np.asarray(self.intervention.simulation.inserted_lengths, dtype=np.float64).reshape(-1).copy()

    @property
    def truncated(self) -> bool:
        return self.stalled

    def step(self) -> None:
        self.history.append(self._inserted())
        full = len(self.history) == self.window_steps + 1
        self.stalled = bool(self.active and full
                            and np.all(np.abs(self.history[-1] - self.history[0]) < self.min_mm))
        self.intervention.simulation.stalled = self.stalled

    def reset(self, episode_nr: int = 0) -> None:
        self.active, self.requested = self.always or self.requested, False
        self.history.clear()
        self.history.append(self._inserted())
        self.stalled = False
        self.intervention.simulation.stalled = False


@register("info")
class StallInfo(Info):
    """``info["stalled"]``: ``Stalled`` ended the episode this step (it leaves the flag on the simulation object)."""

    def __init__(self, intervention, name: str = "stall") -> None:
        super().__init__(name)
        self.intervention = intervention

    @property
    def info(self) -> Dict[str, Any]:
        return {"stalled": bool(getattr(self.intervention.simulation, "stalled", False))}
