"""Per-episode changes of the scene (todo_list/small_changes_generalization.md, section 2).

``EpisodeRandomizer`` draws this episode's scene parameters right before the intervention's reset
builds the scene, so the simulator is built with them. Two groups, each optional:

**friction**: the wall friction coefficient (``simulation.friction``; our ``SofaBeamAdapter`` rebuilds
the scene at every reset and hands that number to the contact solver).

**execution** (user 2026-11-16: the robot does not execute a command exactly, add it to the
randomization): what the simulator executes differs from the commanded velocities
(mm/s, rad/s; push and rotation of each device). Per control step, per channel::

    blended  = (1 - f) * command[t - k] + f * command[t - k - 1]    # latency of delay = k + f control steps
    executed = blended * gain * (1 + jitter * u),  u uniform in [-1, 1], new every step
    executed = 0 where |executed| is below the dead band

``delay`` (in control steps, whole part ``k`` and fraction ``f``; 0.5 s at 7.5 Hz is 3.75 steps: user
2026-11-16; with a velocity held over a control step, a latency moves the device by exactly this
blend of two past commands; before the episode's first command the drive is at rest, so those
"commands" are zero), ``gain`` (one per channel, constant over the episode: the user's "the actual
output differs from the command by a constant within an episode"), the
``jitter`` amplitude and the two dead bands are drawn once per episode. A push or pull that would
end at or below 0 mm or past the device's length is dropped again after the change (as the
intervention does for the command). The policy's observation and the records keep the *commanded*
action (``intervention.last_action``, ``info["raw_action"]``): a robot knows what it asked for, not
what its drive did. Only steps taken through ``GuidewireEnv.step`` are changed (``live``); the start's
own pushes are executed as they are.

Config (``env.randomize``; ``false`` = off, nothing is drawn and no random number is consumed)::

    randomize:
      friction: {range: [0.01, 0.3], log: true, nominal_fraction: 0.3}
      execution: {delay_steps: [0.0, 4.0], gain: [0.7, 1.3], jitter: [0.0, 0.1],
                  deadband_push_mm_s: [0.0, 0.3], deadband_rot_rad_s: [0.0, 0.03], nominal_fraction: 0.3}

``nominal_fraction`` of the episodes keep the configured friction (``env.scene.simulation.friction``)
/ an exact execution (drawn separately for the two groups), the others draw uniformly
(friction with ``log: true``: uniformly in the logarithm) from the ranges. A fixed value for an
exam: a range with both ends equal, or for friction ``--set env.scene.simulation.friction=0.05``
with ``randomize`` off.

What was drawn is ``randomizer.drawn`` (``ScenarioInfo`` puts it in ``info["scenario"]``, from where
the episode records take it), so that success can be counted per value.

Seeding: a reset with a seed restarts the generator from that seed (mixed with a constant, so the
draw is not the target's draw); a reset without one continues the stream.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from eve.info import Info

from .registry import register

_SEED_SALT = 0x5CE9A210
_EXECUTION_RANGES = {"delay_steps": (0.0, 0.0), "gain": (1.0, 1.0), "jitter": (0.0, 0.0),
                     "deadband_push_mm_s": (0.0, 0.0), "deadband_rot_rad_s": (0.0, 0.0)}


class EpisodeRandomizer:
    """Draws the episode's scene parameters and writes them into the scene's parts (module docstring)."""

    def __init__(self, simulation, friction: Optional[Mapping[str, Any]] = None,
                 execution: Optional[Mapping[str, Any]] = None, intervention=None) -> None:
        self.simulation = simulation
        self.intervention = intervention          # for the devices' lengths (the insertion limits)
        self.execution = self._execution_spec(execution) if execution else None
        self.live = False                         # True only inside GuidewireEnv.step
        self._history: list = []                  # this episode's commands, newest last
        self._episode: Dict[str, Any] = {}
        if self.execution is not None:
            self._simulation_step = simulation.step
            simulation.step = self._step          # the instance's step: commanded -> executed
        self.nominal_friction = float(simulation.friction)
        self.friction = self._friction_spec(friction) if friction else None
        if self.friction is not None and getattr(simulation, "rebuild_every_resets", 1) != 1:
            raise ValueError("randomize.friction needs a scene rebuilt at every reset "
                             "(simulation.rebuild_every_resets: 1): the coefficient is read when the scene is built")
        self._rng = np.random.default_rng()
        self.drawn: Dict[str, float] = {"friction": self.nominal_friction}
        if self.execution is not None:
            self._draw_execution(nominal=True)

    @staticmethod
    def _friction_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = set(spec) - {"range", "log", "nominal_fraction"}
        if unknown:
            raise KeyError(f"randomize.friction: unknown keys {sorted(unknown)}")
        low, high = (float(x) for x in spec["range"])
        if not 0.0 < low <= high:
            raise ValueError(f"randomize.friction.range must be 0 < low <= high, got {[low, high]}")
        fraction = float(spec.get("nominal_fraction", 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"randomize.friction.nominal_fraction must be in [0, 1], got {fraction}")
        return {"low": low, "high": high, "log": bool(spec.get("log", False)), "nominal_fraction": fraction}

    @staticmethod
    def _execution_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = set(spec) - set(_EXECUTION_RANGES) - {"nominal_fraction"}
        if unknown:
            raise KeyError(f"randomize.execution: unknown keys {sorted(unknown)}")
        out: Dict[str, Any] = {}
        for key, default in _EXECUTION_RANGES.items():
            low, high = (float(x) for x in spec.get(key, default))
            if not low <= high:
                raise ValueError(f"randomize.execution.{key} must be [low, high] with low <= high, got {[low, high]}")
            out[key] = (low, high)
        if out["delay_steps"][0] < 0.0 or out["delay_steps"][1] > 16.0:
            raise ValueError("randomize.execution.delay_steps must lie in [0, 16] control steps")
        if out["gain"][0] <= 0.0 or out["jitter"][0] < 0.0 or out["jitter"][1] >= 1.0:
            raise ValueError("randomize.execution: gain must be positive and jitter in [0, 1)")
        if out["deadband_push_mm_s"][0] < 0.0 or out["deadband_rot_rad_s"][0] < 0.0:
            raise ValueError("randomize.execution: dead bands must not be negative")
        fraction = float(spec.get("nominal_fraction", 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"randomize.execution.nominal_fraction must be in [0, 1], got {fraction}")
        out["nominal_fraction"] = fraction
        return out

    def _draw_execution(self, nominal: bool) -> None:
        spec = self.execution
        if nominal:
            episode = {"delay": 0.0, "gain": None, "jitter": 0.0, "deadband": (0.0, 0.0)}
        else:
            episode = {"delay": float(self._rng.uniform(*spec["delay_steps"])),
                       "gain": self._rng.uniform(*spec["gain"], size=8),     # per channel; up to 4 devices
                       "jitter": float(self._rng.uniform(*spec["jitter"])),
                       "deadband": (float(self._rng.uniform(*spec["deadband_push_mm_s"])),
                                    float(self._rng.uniform(*spec["deadband_rot_rad_s"])))}
        self._episode = episode
        self._history = []
        gain = episode["gain"]
        self.drawn.update(
            delay_steps=episode["delay"], jitter=episode["jitter"],
            deadband_push_mm_s=episode["deadband"][0], deadband_rot_rad_s=episode["deadband"][1],
            **{f"gain_{k}": (1.0 if gain is None else float(gain[k])) for k in range(4)})

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng([int(seed), _SEED_SALT])
        if self.friction is not None:
            spec = self.friction
            if self._rng.random() < spec["nominal_fraction"]:
                value = self.nominal_friction
            elif spec["log"]:
                value = math.exp(self._rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
            else:
                value = float(self._rng.uniform(spec["low"], spec["high"]))
            self.simulation.friction = float(value)
            self.drawn = {"friction": float(value)}
        if self.execution is not None:
            self._draw_execution(nominal=bool(self._rng.random() < self.execution["nominal_fraction"]))

    def executed(self, command: np.ndarray) -> np.ndarray:
        """What the drive does with this step's commanded velocities (``(devices, 2)``; module docstring)."""
        command = np.asarray(command, dtype=np.float64)
        episode = self._episode
        if episode["gain"] is None:                 # an exact episode
            return command
        whole = int(episode["delay"])
        fraction = episode["delay"] - whole
        self._history.append(command.copy())
        del self._history[: -(whole + 2)]           # keep what the delay can still reach

        def past(steps_back: int) -> np.ndarray:    # the drive is at rest before the first command
            return self._history[-1 - steps_back] if steps_back < len(self._history) else np.zeros_like(command)

        out = (1.0 - fraction) * past(whole) + fraction * past(whole + 1)
        flat = out.reshape(-1)
        noise = self._rng.uniform(-1.0, 1.0, size=flat.shape)
        flat = flat * episode["gain"][: flat.size] * (1.0 + episode["jitter"] * noise)
        out = flat.reshape(command.shape)
        push_band, rot_band = episode["deadband"]
        out[:, 0] = np.where(np.abs(out[:, 0]) < push_band, 0.0, out[:, 0])
        out[:, 1] = np.where(np.abs(out[:, 1]) < rot_band, 0.0, out[:, 1])
        return out

    def _limits(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        inserted = getattr(self.intervention, "instrument_lengths_inserted", None)
        maximum = getattr(self.intervention, "instrument_lengths_maximum", None)
        return (None if inserted is None else np.asarray(inserted, dtype=np.float64),
                None if maximum is None else np.asarray(maximum, dtype=np.float64))

    def _step(self, action: np.ndarray, duration: float):
        if not self.live:
            return self._simulation_step(action, duration)
        out = self.executed(np.asarray(action, dtype=np.float64).reshape(-1, 2))
        inserted, maximum = self._limits()
        if inserted is not None:
            end = inserted + out[:, 0] * duration
            drop = end <= 0.0
            if maximum is not None:
                drop |= end >= maximum
            out[drop, 0] = 0.0
        return self._simulation_step(out.astype(np.asarray(action).dtype).reshape(np.shape(action)), duration)


class InterventionWithRandomizer:
    """The intervention with the episode's draw right before its reset. Everything else is the intervention's."""

    def __init__(self, intervention, randomizer: EpisodeRandomizer) -> None:
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "randomizer", randomizer)

    def reset(self, episode_number: int = 0, seed=None, options=None) -> None:
        self.randomizer.reset(seed)
        self.intervention.reset(episode_number, seed, options)

    def __getattr__(self, name: str):
        try:
            intervention = self.__dict__["intervention"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(intervention, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.__dict__["intervention"], name, value)      # same reason as builder._InterventionWithCArm


@register("info")
class ScenarioInfo(Info):
    """``info["scenario"]``: this episode's scene parameters (``{"friction": ...}``), drawn or configured."""

    def __init__(self, simulation, randomizer: Optional[EpisodeRandomizer] = None, name: str = "scenario") -> None:
        super().__init__(name)
        self.simulation = simulation
        self.randomizer = randomizer

    @property
    def info(self) -> Dict[str, Any]:
        if self.randomizer is not None:
            return {"scenario": dict(self.randomizer.drawn)}
        return {"scenario": {"friction": float(self.simulation.friction)}}
