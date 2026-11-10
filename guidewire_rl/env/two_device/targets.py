"""The two-device task's target (todo_list/two_devices_training.md, section 1).

``RouteWindowTarget``: a point on the right carotid's centerline between ``depth_min_mm`` (35) and
``depth_max_mm`` (55) past its opening (the trunk's end), i.e. on the last 2.7 mm of ``rcca`` and
the first 17.3 mm of ``rcca_rest``. Candidates every ``spacing_mm`` (1 mm): 35, 36, ..., 55;
``split: train`` draws from the even-numbered ones (35, 37, ..., 55: 11 points), ``split: eval``
from the odd-numbered ones (36, ..., 54: 10 points), ``split: all`` from all 21.

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
                 split: str = "train", route_chain: Sequence[str] = ROUTE_CHAIN) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.threshold = float(threshold)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)
        self.spacing_mm = float(spacing_mm)
        self.split = split
        self.route_chain = tuple(route_chain)
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

    def candidates(self) -> np.ndarray:
        """Candidate points (vessel frame), in ``candidate_depths`` order."""
        route = Route.of(self.vessel_tree, self.route_chain)
        return np.array([route.point_at(route.s_rcca + depth) for depth in self.candidate_depths])

    @property
    def coordinates2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.coordinates3d)

    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(int(seed))       # Python 3.12 refuses numpy integers
        depths = self.candidate_depths
        k = self._rng.randrange(len(depths))
        self.depth_mm = float(depths[k])
        route = Route.of(self.vessel_tree, self.route_chain)
        self.coordinates_vessel = route.point_at(route.s_rcca + self.depth_mm)
        self.branch = route.location_at(route.s_rcca + self.depth_mm).branch
        self.coordinates3d = vessel_cs_to_tracking3d(
            self.coordinates_vessel[None], self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center, self.fluoroscopy.field_of_view).reshape(3)
        self.reached = False

    def step(self) -> None:
        from .phases import WIRE, devices_of
        wire_tip = devices_of(self.fluoroscopy.simulation).tips[WIRE]
        self.reached = bool(np.linalg.norm(self.coordinates_vessel - wire_tip) < self.threshold)
