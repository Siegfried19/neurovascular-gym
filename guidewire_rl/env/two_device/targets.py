"""The two-device task's target (todo_list/archive/two_devices_training.md, section 1).

``RouteWindowTarget``: a point on the route's centerline between ``depth_min_mm`` (35) and
``depth_max_mm`` (55) past a reference point: by default the right carotid's opening (the trunk's
end), so the first round's 35-55 mm lie on the last 2.7 mm of ``rcca`` and the first 17.3 mm of
``rcca_rest``; with ``past_fork`` (round two: ``v8``, the ICA/ECA split; user decision 2026-11-09)
the depths count from where that side vessel leaves the route (``Route.s_of_fork``), so 35-45 mm
past the split puts the wire tip past the ICA's first sharp bend with the catheter at F on the
wire's stiff part. Candidates every ``spacing_mm`` (1 mm); ``split: train`` draws from the
even-numbered ones, ``split: eval`` from the odd-numbered ones, ``split: all`` from all.
``jitter_mm`` (default 0) moves a ``split: train`` draw by a uniform amount within +-``jitter_mm``
along the route, so training sees a band around each of its candidates instead of the same few
points (round2_dagger_v3: per-target success of one policy between 0 of 36 and 54 of 55 on the eval
points, and 100 % on the train points against 42 % on the eval points 1 mm away, both with the
policy's noise; agent_log/TRAINING_RECORD.md). Eval and all draws are never moved, and with
``jitter_mm`` below half the candidate spacing the eval points stay outside the bands.
``sampling: uniform`` (default ``grid``) makes a ``split: train`` draw any depth between
``depth_min_mm`` and ``depth_max_mm`` instead of one of its candidates (2026-11-16: with the bands
the delivered policy still lost 3 of 10 episodes on one eval point and 1 of 30 on the other four;
agent_log/CURRENT_STATE.md, section 4, item 16). The eval points are then inside what training
draws from, as single points of a continuum; eval and all draws stay on their candidates.

As every stEVE target, ``coordinates3d`` / ``coordinates2d`` are in the fluoroscopy's tracking frame
(``Target2D`` reads them); ``coordinates_vessel`` is the same point in vessel coordinates and
``branch`` the centerline piece it lies on (``rcca`` or ``rcca_rest``; the per-target C-arm reads it,
``guidewire_rl/env/c_arm/per_target.py``).
``reached`` is the *wire* tip (instrument 0) within ``threshold`` (stEVE's own targets use whichever
tip leads).
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np
from eve.intervention.target.target import Target
from eve.util.coordtransform import tracking3d_to_2d, vessel_cs_to_tracking3d

from ..registry import register
from .route import ROUTE_CHAIN, Route

SPLITS = ("train", "eval", "all")


@register("target")
class RouteWindowTarget(Target):
    """Target on the right carotid, 35-55 mm past its opening (module docstring)."""

    def __init__(self, vessel_tree, fluoroscopy, threshold: float = 5.0,
                 depth_min_mm: float = 35.0, depth_max_mm: float = 55.0, spacing_mm: float = 1.0,
                 split: str = "train", route_chain: Sequence[str] = ROUTE_CHAIN,
                 past_fork: Optional[str] = None, jitter_mm: float = 0.0,
                 sampling: str = "grid") -> None:
        if sampling not in ("grid", "uniform"):
            raise ValueError(f"sampling must be grid or uniform, got {sampling!r}")
        if sampling == "uniform" and jitter_mm > 0.0:
            raise ValueError("sampling: uniform already covers the whole window; set jitter_mm to 0")
        self.sampling = sampling
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if jitter_mm < 0.0:
            raise ValueError(f"jitter_mm must not be negative, got {jitter_mm!r}")
        self.jitter_mm = float(jitter_mm)
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.threshold = float(threshold)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)
        self.spacing_mm = float(spacing_mm)
        self.split = split
        self.route_chain = tuple(route_chain)
        self.past_fork = past_fork
        self.reached = False
        self.depth_mm = float("nan")
        self.coordinates_vessel = np.zeros(3, dtype=np.float64)
        self.coordinates3d = np.zeros(3, dtype=np.float32)
        self.branch: Optional[str] = None
        self._rng = random.Random()

    @property
    def candidate_depths(self) -> List[float]:
        n = int(round((self.depth_max_mm - self.depth_min_mm) / self.spacing_mm)) + 1
        depths = [self.depth_min_mm + i * self.spacing_mm for i in range(n)]
        if self.split == "train":
            return depths[0::2]
        if self.split == "eval":
            return depths[1::2]
        return depths

    def reference_s(self, route: Route) -> float:
        """Route arc length the depths count from: the fork's junction, else the carotid's opening."""
        return route.s_of_fork(self.past_fork) if self.past_fork else route.s_rcca

    def candidates(self) -> np.ndarray:
        """Candidate points (vessel frame), in ``candidate_depths`` order."""
        route = Route.of(self.vessel_tree, self.route_chain)
        base = self.reference_s(route)
        return np.array([route.point_at(base + depth) for depth in self.candidate_depths])

    @property
    def coordinates2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.coordinates3d)

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(int(seed))       # Python 3.12 refuses numpy integers
        depths = self.candidate_depths
        k = self._rng.randrange(len(depths))
        self.depth_mm = float(depths[k])
        if self.sampling == "uniform" and self.split == "train":
            self.depth_mm = self._rng.uniform(self.depth_min_mm, self.depth_max_mm)
        if self.jitter_mm > 0.0 and self.split == "train":     # no extra draw otherwise: old seeds keep their targets
            self.depth_mm += self._rng.uniform(-self.jitter_mm, self.jitter_mm)
        route = Route.of(self.vessel_tree, self.route_chain)
        base = self.reference_s(route)
        self.coordinates_vessel = route.point_at(base + self.depth_mm)
        self.branch = route.location_at(base + self.depth_mm).branch
        self.coordinates3d = vessel_cs_to_tracking3d(
            self.coordinates_vessel[None], self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center, self.fluoroscopy.field_of_view).reshape(3)
        self.reached = False

    def step(self) -> None:
        from .phases import WIRE, devices_of
        wire_tip = devices_of(self.fluoroscopy.simulation).tips[WIRE]
        self.reached = bool(np.linalg.norm(self.coordinates_vessel - wire_tip) < self.threshold)
