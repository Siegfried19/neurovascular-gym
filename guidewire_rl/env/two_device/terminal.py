"""Episode ends that are real ends (no value bootstrapped): success and hard failure, both decided
by the referee (``PhaseDetector``, phases.py: the conditions and their numbers)."""

from __future__ import annotations

from eve.terminal import Terminal

from ..registry import register
from .phases import PhaseDetector


@register("terminal")
class PlatformReached(Terminal):
    """Wire tip <= 5 mm from the target, catheter >= 15 mm into the right carotid, wire 15-45 mm
    ahead, all for 8 steps in a row."""

    def __init__(self, phase_detector: PhaseDetector) -> None:
        self.phase_detector = phase_detector

    @property
    def terminal(self) -> bool:
        return bool(self.phase_detector.state.success)

    def step(self) -> None:
        pass

    def reset(self, episode_nr: int = 0) -> None:
        pass


@register("terminal")
class HardFailure(Terminal):
    """A tip outside the vessel, or both tips at the same insertion length."""

    def __init__(self, phase_detector: PhaseDetector) -> None:
        self.phase_detector = phase_detector

    @property
    def terminal(self) -> bool:
        return bool(self.phase_detector.state.hard_failure)

    def step(self) -> None:
        pass

    def reset(self, episode_nr: int = 0) -> None:
        pass
