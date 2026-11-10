"""Reward terms of the two-device task (todo_list/two_devices_training.md, section 2), one stEVE
``Reward`` each, listed one by one under ``env.reward`` (they add up).

All read the referee (``PhaseDetector``); none reads a centerline-keeping term. A step with a
simulation error (stEVE has thrown the devices back to the insertion point) gives 0 from every
term and moves no potential (the episode is truncated there, not punished).

  - ``LeadingTipProgress`` (0.01 / mm): the tip further along the route (the one with the shorter
    path to the target, ``route.path_to_target``) gets closer. A potential: -min(path_w, path_c),
    so it stays continuous when the other tip takes the lead.
  - ``CatheterToF`` (0.003 / mm): the catheter tip's path to F (15 mm into the right carotid, the
    success depth) gets shorter; 0 past F.
  - Both measure P0 (over the arch) with the arch ruler (``PhaseState.progress_paths``): path left
    to the hook point H = O + 30 mm along the aorta plus the fixed path from H to the target (to F);
    0 + fixed once past H. P0 must push the catheter past O to H, which the plain path would
    charge. The two rulers agree at H, where P1 starts, so the potential does not jump there
    (except by how far the catheter got past H in the two steps the referee takes to confirm P1).
    A fall back to P0 is charged by the jump back (as every potential term).
  - ``HookPrepPotential`` (0.006 / mm): the potential clamp(-d, 0, 35): pulling the wire back
    behind the hook (d down to -35 mm) earns, pushing it out costs, until the episode first goes
    P2 -> P3; from then on (to the end of the episode) the term is locked: no reward and no claw
    back while the wire comes out of the catheter.
  - ``PhasePotential`` (0.2 / phase): the phase number as a potential (a fall back costs what the
    advance earned).
  - ``WindowPenalty`` (0.001 / mm): d outside the phase's window
    (P0 [10, 80], P1 [-120, 0], P2 [-120, -30], P3 [-100, 80], P4 [15, 60]). In P2 with the wire
    already out (d > 0) and its tip past the trunk's mouth (the catheter slipped out of the
    opening after P3/P4), P3's window [-100, 80] instead: decided from the state, not from where
    the episode came from.
  - ``EventPenalty``: per event (``penalties``; wrong vessel -0.1).
  - ``ShieldPenalty`` (0.05): times the shell's penalized intervention (L1, normalized units).
  - ``StepPenalty`` (-0.002 per step).
  - ``OutcomeReward``: +1 on success, -1 on a hard failure (both end the episode).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from eve.reward import Reward

from ..registry import register
from .phases import PhaseDetector, PhaseState

DEFAULT_WINDOWS = {0: (10.0, 80.0), 1: (-120.0, 0.0), 2: (-120.0, -30.0), 3: (-100.0, 80.0),
                   4: (15.0, 60.0)}


class _PhaseReward(Reward):
    """Reads the referee; 0 on a simulation-error step."""

    def __init__(self, phase_detector: PhaseDetector, factor: float) -> None:
        self.phase_detector = phase_detector
        self.factor = float(factor)
        self.reward = 0.0

    def step(self) -> None:
        state = self.phase_detector.state
        self.reward = 0.0 if state.sim_error else float(self._step(state))

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._reset(self.phase_detector.state)

    def _step(self, state: PhaseState) -> float:
        raise NotImplementedError

    def _reset(self, state: PhaseState) -> None:
        pass


class _Potential(_PhaseReward):
    """factor x (phi(now) - phi(before))."""

    def __init__(self, phase_detector: PhaseDetector, factor: float) -> None:
        super().__init__(phase_detector, factor)
        self.phi: Optional[float] = None

    def potential(self, state: PhaseState) -> float:
        raise NotImplementedError

    def _reset(self, state: PhaseState) -> None:
        self.phi = self.potential(state)

    def _step(self, state: PhaseState) -> float:
        phi = self.potential(state)
        out = self.factor * (phi - self.phi)
        self.phi = phi
        return out


@register("reward")
class LeadingTipProgress(_Potential):
    def __init__(self, phase_detector: PhaseDetector, factor: float = 0.01) -> None:
        super().__init__(phase_detector, factor)

    def potential(self, state: PhaseState) -> float:
        wire, catheter, _ = state.progress_paths()
        return -min(wire, catheter)


@register("reward")
class CatheterToF(_Potential):
    def __init__(self, phase_detector: PhaseDetector, factor: float = 0.003) -> None:
        super().__init__(phase_detector, factor)

    def potential(self, state: PhaseState) -> float:
        return -state.progress_paths()[2]


@register("reward")
class HookPrepPotential(_Potential):
    def __init__(self, phase_detector: PhaseDetector, factor: float = 0.006,
                 depth_mm: float = 35.0, lock_from: int = 2, lock_to: int = 3) -> None:
        super().__init__(phase_detector, factor)
        self.depth_mm = float(depth_mm)
        self.lock_from, self.lock_to = int(lock_from), int(lock_to)
        self.locked = False

    def potential(self, state: PhaseState) -> float:
        return min(max(-state.d, 0.0), self.depth_mm)

    def _reset(self, state: PhaseState) -> None:
        self.locked = False
        super()._reset(state)

    def _step(self, state: PhaseState) -> float:
        if self.locked:
            return 0.0
        out = super()._step(state)
        if state.previous_phase == self.lock_from and state.phase == self.lock_to:
            self.locked = True
        return out


@register("reward")
class PhasePotential(_Potential):
    def __init__(self, phase_detector: PhaseDetector, factor: float = 0.2) -> None:
        super().__init__(phase_detector, factor)

    def potential(self, state: PhaseState) -> float:
        return float(state.phase)


@register("reward")
class WindowPenalty(_PhaseReward):
    def __init__(self, phase_detector: PhaseDetector, factor: float = 0.001,
                 windows: Optional[Mapping[int, Sequence[float]]] = None,
                 select_wire_out: Sequence[float] = DEFAULT_WINDOWS[3],
                 cap_mm: Optional[float] = None) -> None:
        super().__init__(phase_detector, factor)
        windows = DEFAULT_WINDOWS if windows is None else windows
        self.windows: Dict[int, tuple] = {int(k): (float(v[0]), float(v[1])) for k, v in windows.items()}
        self.select_wire_out = (float(select_wire_out[0]), float(select_wire_out[1]))
        # Per-step cap on the counted excess (mm). Without it an exploring policy that shoots the
        # wire 100-200 mm ahead pays 0.1-0.2 per step, ten times the progress reward's rate, and the
        # episode return is nothing but this term (run A, 2026-10-31: mean return -56 at 38k steps).
        self.cap_mm = None if cap_mm is None else float(cap_mm)

    def window(self, state: PhaseState) -> tuple:
        """The phase's window; in P2 with the wire out of the catheter (d > 0) and its tip past
        the trunk's mouth, ``select_wire_out`` (P3's)."""
        if int(state.phase) == 2 and state.d > 0 and state.depth_bct_w > 0:
            return self.select_wire_out
        return self.windows[int(state.phase)]

    def outside_mm(self, state: PhaseState) -> float:
        low, high = self.window(state)
        outside = max(low - state.d, 0.0, state.d - high)
        return outside if self.cap_mm is None else min(outside, self.cap_mm)

    def _step(self, state: PhaseState) -> float:
        return -self.factor * self.outside_mm(state)


@register("reward")
class EventPenalty(_PhaseReward):
    def __init__(self, phase_detector: PhaseDetector,
                 penalties: Optional[Mapping[str, float]] = None) -> None:
        super().__init__(phase_detector, 1.0)
        self.penalties = {"wrong_vessel": -0.1} if penalties is None else dict(penalties)

    def _step(self, state: PhaseState) -> float:
        return sum(float(value) * float(state.events.get(event, 0))
                   for event, value in self.penalties.items())


@register("reward")
class ShieldPenalty(Reward):
    def __init__(self, action_map, factor: float = 0.05) -> None:
        self.action_map = action_map
        self.factor = float(factor)
        self.reward = 0.0

    def step(self) -> None:
        self.reward = -self.factor * float(self.action_map.last.penalized)

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0


@register("reward")
class StepPenalty(Reward):
    def __init__(self, factor: float = -0.002) -> None:
        self.factor = float(factor)
        self.reward = 0.0

    def step(self) -> None:
        self.reward = self.factor

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0


@register("reward")
class OutcomeReward(_PhaseReward):
    def __init__(self, phase_detector: PhaseDetector, success: float = 1.0,
                 failure: float = -1.0) -> None:
        super().__init__(phase_detector, 1.0)
        self.success = float(success)
        self.failure = float(failure)

    def _step(self, state: PhaseState) -> float:
        if state.success:
            return self.success
        if state.hard_failure:
            return self.failure
        return 0.0
