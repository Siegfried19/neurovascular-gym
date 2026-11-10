"""What the policy of the two-device task sees (todo_list/two_devices_training.md, section 2).

Only what a real setup can measure: the two devices in the 2D X-ray image (tracked tip and shape),
the roadmap's openings and the target marked on the image, the robot's encoders (insertion
lengths, rotations), the commands it executed, the script's suggestion and the phase. 3D truth
stays with the referee, rewards, labels and the shell.

Image coordinates are the fluoroscopy's 2D tracking frame (mm). ``Normalize2DTask`` maps them to
[-1, 1] over the task region's box (``route.task_box_2d``: the arch, its three branches and the
right carotid up to 100 mm past its opening, widened by each centerline point's radius and 5 mm);
shape points relative to the tip are divided by the same half-box, so both are in the same units.

Terms:

  - ``InstrumentTip2D``: one device's tip in the image (mm; wrap with ``TipNoise2D`` and
    ``Normalize2DTask``).
  - ``InstrumentShape2D``: ``n_points`` points behind one device's tip, every ``spacing_mm`` along
    its image (projected) curve, relative to the tip, normalized.
  - ``TipNoise2D`` (wrapper): adds tracking noise to a 2D position, uniform in +-sigma per axis and
    step, sigma drawn per episode in [``noise_min_mm``, ``noise_max_mm``] (0.5-1 mm).
  - ``Normalize2DTask`` (wrapper): image mm -> [-1, 1] over the task box.
  - ``TaskTarget2D``: the target in the image, mm (wrap with ``Normalize2DTask``).
  - ``JunctionMap2D``: the next two openings on the route that neither tip is >= 8 mm into, each as
    its image position (offset by up to +-3 mm per axis, fixed for the episode, then put on a 2 mm
    grid: an annotated roadmap) and the image direction of the first 10 mm of the vessel to enter;
    past the last opening the target fills in (position, route direction there). 8 numbers.
  - ``PhaseOneHot``: the referee's phase, 5 numbers.
  - ``InsertionDiff``: d = L_wire - L_cath, / 100 mm, clipped to [-1, 1].
  - ``InsertionLengthsNormalized``: each insertion length / that device's length.
  - ``ScriptSuggestion``: the script's suggested next action (``NominalAdvisor``), zeros without one.
  - ``FrameHistory``: earlier terms' outputs (as the policy saw them, noise included) from the
    previous ``frames`` steps, most recent first; at a reset filled with the first frame.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from eve.observation import Observation

from ..registry import register
from .phases import N_PHASES, WIRE, PhaseDetector, devices_of
from .route import Route, task_box_2d, to_tracking2d, unit


def _episode_rng(intervention) -> np.random.Generator:
    """A generator seeded from the intervention's (seeded by ``env.reset(seed)``)."""
    source = getattr(intervention, "_np_random", None)
    if source is None:
        return np.random.default_rng()
    return np.random.default_rng(int(source.integers(0, 2**31)))


def sample_behind_2d(polyline2d: np.ndarray, n_points: int, spacing_mm: float) -> np.ndarray:
    """``n_points`` points at 1, 2, ... x ``spacing_mm`` along a 2D polyline from its first point
    (the tip); a curve too short ends in its last point."""
    pts = np.asarray(polyline2d, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    pts = pts[keep]
    if len(pts) < 2:
        return np.repeat(pts[:1], n_points, axis=0)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = spacing_mm * np.arange(1, n_points + 1)
    return np.stack([np.interp(s, cum, pts[:, k]) for k in range(2)], axis=1)


class _TaskBox:
    """The task box of a scene, computed once per (vessel tree, fluoroscopy angles)."""

    def __init__(self, intervention) -> None:
        self.intervention = intervention
        self._key = None
        self.low = self.high = None

    def get(self):
        fluoro = self.intervention.fluoroscopy
        key = (id(self.intervention.vessel_tree), tuple(np.ravel(fluoro.image_rot_zx)),
               tuple(np.ravel(fluoro.image_center)))
        if key != self._key:
            self.low, self.high = task_box_2d(Route.of(self.intervention.vessel_tree), fluoro)
            self._key = key
        return self.low, self.high

    def normalize(self, points: np.ndarray) -> np.ndarray:
        low, high = self.get()
        return (2.0 * (np.asarray(points, dtype=np.float64) - low) / (high - low) - 1.0)

    def half(self) -> np.ndarray:
        low, high = self.get()
        return (high - low) / 2.0


@register("observation")
class InstrumentTip2D(Observation):
    """One device's tip in the image, mm (module docstring)."""

    def __init__(self, intervention, instrument: int = WIRE, name: str = "tip2d") -> None:
        self.name = name
        self.intervention = intervention
        self.instrument = int(instrument)
        self.box = _TaskBox(intervention)
        self.obs = np.zeros(2, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low, high = self.box.get()
        return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

    def step(self) -> None:
        tip = devices_of(self.intervention.simulation).tips[self.instrument]
        self.obs = to_tracking2d(tip, self.intervention.fluoroscopy).reshape(2).astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation")
class InstrumentShape2D(Observation):
    """Points behind one device's tip along its image curve, relative to the tip (module
    docstring). ``normalize: false`` keeps mm."""

    def __init__(self, intervention, instrument: int = WIRE, n_points: int = 4,
                 spacing_mm: float = 4.0, normalize: bool = True, name: str = "shape2d") -> None:
        self.name = name
        self.intervention = intervention
        self.instrument = int(instrument)
        self.n_points = int(n_points)
        self.spacing_mm = float(spacing_mm)
        self.normalize = bool(normalize)
        self.box = _TaskBox(intervention)
        self.obs = np.zeros((self.n_points, 2), dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        bound = 2.0 if self.normalize else np.inf
        return gym.spaces.Box(low=-bound, high=bound, shape=(self.n_points, 2), dtype=np.float32)

    def step(self) -> None:
        line = devices_of(self.intervention.simulation).polylines[self.instrument]
        line2d = to_tracking2d(line, self.intervention.fluoroscopy)
        relative = sample_behind_2d(line2d, self.n_points, self.spacing_mm) - line2d[0]
        if self.normalize:
            relative = relative / self.box.half()
        self.obs = relative.astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation")
class TaskTarget2D(Observation):
    """The target in the image, mm (stEVE's ``Target2D`` asks the fluoroscopy for its 2D box,
    which stEVE builds by rotating two corners of the vessel box: with a C-arm angle like
    [61, -21] some low > high and gymnasium refuses the box)."""

    def __init__(self, intervention, name: str = "target2d") -> None:
        self.name = name
        self.intervention = intervention
        self.box = _TaskBox(intervention)
        self.obs = np.zeros(2, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        low, high = self.box.get()
        return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

    def step(self) -> None:
        self.obs = np.asarray(self.intervention.target.coordinates2d, dtype=np.float32).reshape(2)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation_wrapper")
class TipNoise2D(Observation):
    """Tracking noise on a 2D position, mm (module docstring)."""

    def __init__(self, wrapped_obs: Observation, intervention, noise_min_mm: float = 0.5,
                 noise_max_mm: float = 1.0, name: Optional[str] = None) -> None:
        self.name = name or wrapped_obs.name
        self.wrapped_obs = wrapped_obs
        self.intervention = intervention
        self.noise_min_mm = float(noise_min_mm)
        self.noise_max_mm = float(noise_max_mm)
        self.sigma = 0.0
        self._rng = np.random.default_rng()
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        return self.wrapped_obs.space

    def _noisy(self) -> np.ndarray:
        clean = np.asarray(self.wrapped_obs.obs, dtype=np.float64)
        noise = self._rng.uniform(-self.sigma, self.sigma, size=clean.shape)
        return (clean + noise).astype(np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self.obs = self._noisy()

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self._rng = _episode_rng(self.intervention)
        self.sigma = float(self._rng.uniform(self.noise_min_mm, self.noise_max_mm))
        self.obs = self._noisy()


@register("observation_wrapper")
class Normalize2DTask(Observation):
    """Image mm -> [-1, 1] over the task box (module docstring)."""

    def __init__(self, wrapped_obs: Observation, intervention, name: Optional[str] = None) -> None:
        self.name = name or wrapped_obs.name
        self.wrapped_obs = wrapped_obs
        self.box = _TaskBox(intervention)
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        shape = self.wrapped_obs.space.shape
        return gym.spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

    def step(self) -> None:
        self.wrapped_obs.step()
        self.obs = self.box.normalize(self.wrapped_obs.obs).astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.wrapped_obs.reset(episode_nr)
        self.obs = self.box.normalize(self.wrapped_obs.obs).astype(np.float32)


@register("observation")
class JunctionMap2D(Observation):
    """The next openings on the route as an annotated roadmap would show them (module docstring)."""

    def __init__(self, intervention, phase_detector: PhaseDetector, n_junctions: int = 2,
                 enter_mm: float = 8.0, offset_mm: float = 3.0, grid_mm: float = 2.0,
                 name: str = "junctions") -> None:
        self.name = name
        self.intervention = intervention
        self.phase_detector = phase_detector
        self.n_junctions = int(n_junctions)
        self.enter_mm = float(enter_mm)
        self.offset_mm = float(offset_mm)
        self.grid_mm = float(grid_mm)
        self.box = _TaskBox(intervention)
        self.entries: List[np.ndarray] = []        # per route junction: annotated (x, z, dx, dz) mm/unit
        self.fill = np.zeros(4)
        self.obs = np.zeros(4 * self.n_junctions, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4 * self.n_junctions,), dtype=np.float32)

    def _direction2d(self, start: np.ndarray, direction: np.ndarray) -> np.ndarray:
        fluoro = self.intervention.fluoroscopy
        a, b = to_tracking2d(np.vstack([start, start + 10.0 * direction]), fluoro)
        return unit(b - a)

    def reset(self, episode_nr: int = 0) -> None:
        route = Route.of(self.intervention.vessel_tree, self.phase_detector.route_chain)
        fluoro = self.intervention.fluoroscopy
        rng = _episode_rng(self.intervention)
        self.entries = []
        for _name, mouth, direction in route.junctions:
            position = to_tracking2d(mouth, fluoro).reshape(2)
            position = position + rng.uniform(-self.offset_mm, self.offset_mm, size=2)
            if self.grid_mm > 0:
                position = np.round(position / self.grid_mm) * self.grid_mm
            self.entries.append(np.r_[position, self._direction2d(mouth, direction)])
        target = route.target
        self.fill = np.r_[to_tracking2d(target, fluoro).reshape(2),
                          self._direction2d(target, route.target_direction())]
        self.step()

    def step(self) -> None:
        route = self.phase_detector.route
        tips = devices_of(self.intervention.simulation).tips
        ahead = []
        for (name, _mouth, _direction), entry in zip(route.junctions, self.entries):
            if max(route.depth(tip, name) for tip in tips) < self.enter_mm:
                ahead.append(entry)
        ahead = (ahead + [self.fill] * self.n_junctions)[: self.n_junctions]
        out = []
        for entry in ahead:
            out.extend(self.box.normalize(entry[:2]).tolist())
            out.extend(entry[2:].tolist())
        self.obs = np.asarray(out, dtype=np.float32)


@register("observation")
class PhaseOneHot(Observation):
    def __init__(self, phase_detector: PhaseDetector, name: str = "phase") -> None:
        self.name = name
        self.phase_detector = phase_detector
        self.obs = np.zeros(N_PHASES, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(N_PHASES,), dtype=np.float32)

    def step(self) -> None:
        obs = np.zeros(N_PHASES, dtype=np.float32)
        obs[int(self.phase_detector.state.phase)] = 1.0
        self.obs = obs

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation")
class InsertionDiff(Observation):
    """d = L_wire - L_cath, / ``scale_mm``, clipped to [-1, 1]. (stEVE's
    ``InsertionLengthRelative`` cannot be built: its base class has no ``__init__``.)"""

    def __init__(self, intervention, scale_mm: float = 100.0, wire: int = 0, catheter: int = 1,
                 name: str = "insertion_diff") -> None:
        self.name = name
        self.intervention = intervention
        self.scale_mm = float(scale_mm)
        self.wire, self.catheter = int(wire), int(catheter)
        self.obs = np.zeros(1, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self) -> None:
        lengths = np.asarray(self.intervention.instrument_lengths_inserted, dtype=np.float64)
        d = lengths[self.wire] - lengths[self.catheter]
        self.obs = np.asarray([np.clip(d / self.scale_mm, -1.0, 1.0)], dtype=np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation")
class InsertionLengthsNormalized(Observation):
    """Each insertion length / that device's length, in [0, 1]."""

    def __init__(self, intervention, name: str = "inserted_lengths") -> None:
        self.name = name
        self.intervention = intervention
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        n = len(self.intervention.instrument_lengths_maximum)
        return gym.spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    def step(self) -> None:
        lengths = np.asarray(self.intervention.instrument_lengths_inserted, dtype=np.float64)
        maximum = np.asarray(self.intervention.instrument_lengths_maximum, dtype=np.float64)
        self.obs = np.clip(lengths / maximum, 0.0, 1.0).astype(np.float32)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation")
class ScriptSuggestion(Observation):
    """The script's suggestion for the next step (normalized), zeros without one."""

    def __init__(self, nominal_advisor=None, size: int = 4, name: str = "suggestion") -> None:
        self.name = name
        self.nominal_advisor = nominal_advisor
        self.size = int(size)
        self.obs = np.zeros(self.size, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.size,), dtype=np.float32)

    def step(self) -> None:
        action = None if self.nominal_advisor is None else self.nominal_advisor.action
        self.obs = (np.zeros(self.size, dtype=np.float32) if action is None else
                    np.asarray(action, dtype=np.float32).reshape(self.size))

    def reset(self, episode_nr: int = 0) -> None:
        self.step()


@register("observation")
class FrameHistory(Observation):
    """Earlier terms as seen in the previous ``frames`` steps (module docstring). ``terms`` name
    terms listed before this one in ``env.observation``."""

    def __init__(self, observation_terms: Mapping[str, Observation], terms: Sequence[str],
                 frames: int = 2, name: str = "history") -> None:
        missing = [t for t in terms if t not in observation_terms]
        if missing:
            raise KeyError(f"FrameHistory: terms {missing} must be listed before it in env.observation "
                           f"(listed so far: {list(observation_terms)})")
        self.name = name
        self.sources = [observation_terms[t] for t in terms]
        self.terms = list(terms)
        self.frames = int(frames)
        self.buffer: List[np.ndarray] = []
        self.obs = None

    @property
    def space(self) -> gym.spaces.Box:
        lows = np.concatenate([np.asarray(s.space.low, dtype=np.float32).reshape(-1) for s in self.sources])
        highs = np.concatenate([np.asarray(s.space.high, dtype=np.float32).reshape(-1) for s in self.sources])
        return gym.spaces.Box(low=np.tile(lows, self.frames), high=np.tile(highs, self.frames),
                              dtype=np.float32)

    def _current(self) -> np.ndarray:
        return np.concatenate([np.asarray(s.obs, dtype=np.float32).reshape(-1) for s in self.sources])

    def step(self) -> None:
        self.obs = np.concatenate(self.buffer).astype(np.float32)
        self.buffer = [self._current()] + self.buffer[:-1]

    def reset(self, episode_nr: int = 0) -> None:
        current = self._current()
        self.buffer = [current.copy() for _ in range(self.frames)]
        self.obs = np.concatenate(self.buffer).astype(np.float32)
