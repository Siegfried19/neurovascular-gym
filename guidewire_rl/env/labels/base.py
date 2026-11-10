"""Labels: per-step quantities measured in simulation for training targets.

A label is never part of the observation. The env reports each label's value
for the control step just completed in ``info["labels"][name]``; the replay
stores it next to the transition, and auxiliary heads learn to predict it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class Label(ABC):
    name: str

    def attach(self, env) -> None:
        """Called after every reset, once the scene exists. The simulator may have rebuilt
        the scene in that reset, so this must (re)connect to the current scene and do
        nothing if it is still connected."""

    @abstractmethod
    def begin_step(self) -> None:
        """Called just before the simulator advances one control step."""

    @abstractmethod
    def finish_step(self) -> Tuple[float, Dict[str, Any]]:
        """Return ``(value, details)`` for the control step just completed."""

    def abort_step(self) -> None:
        """Called instead of ``finish_step`` when the step raised."""
