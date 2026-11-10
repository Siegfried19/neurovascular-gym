"""G13's vessel-relative geometry features as a stEVE observation term.

Computation copied from the reference ``contact_aware_tdmpc2/env/observation.py``
(``GeometryFeatureExtractor``). Changes:

* it is an ``eve`` ``Observation``, so it sits in the observation list like any
  stEVE term and is evaluated at the same point of ``eve.Env.step``;
* ``features`` picks which groups go into the observation (order as listed);
* ``frame`` selects how wire points are compared with the vessel centerline:
  ``reference`` reproduces G13 exactly (C-arm-rotated wire points against the
  unrotated centerline — a frame mismatch), ``vessel`` first maps the wire
  points back into vessel coordinates;
* the run-time check against ``Tracking2D`` moved to the tests;
* the 1 mm point sampling advances along a long segment (the reference and
  stEVE's ``Tracking2D`` stack the points there), identical otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from eve.observation import Observation
from eve.util.coordtransform import (tracking3d_to_2d, tracking3d_to_vessel_cs,
                                     vessel_cs_to_tracking3d)

from ..registry import register

TRACKING_SPACING_MM = 1.0
GEOMETRY_SCALE_MM = 20.0

# Feature groups in G13's order; each value lists the per-dimension names.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "clearance": ("d0_normalized", "d1_normalized", "d2_normalized"),
    "heading": ("sin_delta_theta", "cos_delta_theta"),
    "bend": ("kappa_distal_normalized",),
    "radius": ("r_local_normalized",),
    "valid": ("direction_valid",),
}
G13_FEATURES = tuple(FEATURE_GROUPS)
FRAMES = ("reference", "vessel")


@dataclass(frozen=True)
class GeometryResult:
    groups: Dict[str, np.ndarray]
    clearance_mm: np.ndarray
    radius_mm: float
    bend_angle_rad: float
    selected_branch: str
    selected_segment: int
    sampled_tracking_2d_mm: np.ndarray
    sampled_points_3d_mm: np.ndarray


def check_features(features: Sequence[str]) -> List[str]:
    features = list(features)
    unknown = [f for f in features if f not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown geometry feature(s) {unknown}; choose from {list(FEATURE_GROUPS)}")
    if len(set(features)) != len(features):
        raise ValueError(f"Duplicate geometry features in {features}")
    if not features:
        raise ValueError("VesselGeometry needs at least one feature group")
    return features


def assemble(groups: Dict[str, np.ndarray], features: Sequence[str]) -> np.ndarray:
    return np.concatenate([np.asarray(groups[f], dtype=np.float32).reshape(-1)
                           for f in features]).astype(np.float32)


class GeometryCore:
    """Deterministic signed centerline-radius lumen geometry query."""

    def __init__(self, frame: str = "reference") -> None:
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
        self.frame = frame
        self._branch_identity = None
        self._starts = self._vectors = self._length2 = None
        self._radius_start = self._radius_delta = None
        self._tangent_2d = None
        self._segment_branches: list[str] = []
        self._segment_indices: list[int] = []

    def _cache_vessel(self, intervention) -> None:
        branches = intervention.vessel_tree.branches
        if not branches:
            raise RuntimeError("Vessel branches are not initialized")
        # the 2D tangents depend on the C-arm angle, which can change per episode
        # (image_rot_zx: per_target, guidewire_rl/env/c_arm/per_target.py)
        fluoro = intervention.fluoroscopy
        identity = (id(branches), tuple(np.ravel(fluoro.image_rot_zx).tolist()),
                    tuple(np.ravel(fluoro.image_center).tolist()))
        if identity == self._branch_identity:
            return
        starts, ends, r0, r1, names, indices = [], [], [], [], [], []
        for branch in branches:
            if len(branch.coordinates) < 2 or len(branch.radii) != len(branch.coordinates):
                raise RuntimeError(f"Invalid BranchWithRadii geometry: {branch.name}")
            for i in range(len(branch.coordinates) - 1):
                starts.append(branch.coordinates[i])
                ends.append(branch.coordinates[i + 1])
                r0.append(branch.radii[i])
                r1.append(branch.radii[i + 1])
                names.append(branch.name)
                indices.append(i)
        self._starts = np.asarray(starts, dtype=np.float64)
        ends = np.asarray(ends, dtype=np.float64)
        self._vectors = ends - self._starts
        self._length2 = np.einsum("ij,ij->i", self._vectors, self._vectors)
        if np.any(self._length2 <= 1e-12):
            raise RuntimeError("Zero-length vessel centerline segment")
        self._radius_start = np.asarray(r0, dtype=np.float64)
        self._radius_delta = np.asarray(r1, dtype=np.float64) - self._radius_start
        if not np.all(self._radius_start > 0) or not np.all(self._radius_start + self._radius_delta > 0):
            raise RuntimeError("Nonpositive vessel radius")
        fluoro = intervention.fluoroscopy
        projected_starts = tracking3d_to_2d(vessel_cs_to_tracking3d(
            self._starts, fluoro.image_rot_zx, fluoro.image_center,
            fluoro.field_of_view)).astype(np.float64)
        projected_ends = tracking3d_to_2d(vessel_cs_to_tracking3d(
            ends, fluoro.image_rot_zx, fluoro.image_center,
            fluoro.field_of_view)).astype(np.float64)
        if projected_starts.shape != (len(starts), 2) or projected_ends.shape != (len(starts), 2):
            raise RuntimeError("Vessel projection clipped or changed shape")
        self._tangent_2d = projected_ends - projected_starts
        self._segment_branches = names
        self._segment_indices = indices
        self._branch_identity = identity

    @staticmethod
    def sample_tracking_pair(intervention) -> tuple[np.ndarray, np.ndarray]:
        """Mirror Tracking2D's 1 mm projected-point selection, paired with 3D."""
        fluoro = intervention.fluoroscopy
        points2 = np.asarray(fluoro.tracking2d, dtype=np.float32)
        points3 = np.asarray(fluoro.tracking3d, dtype=np.float32)
        if points2.ndim != 2 or points2.shape[1] != 2 or points3.shape != (len(points2), 3) or not len(points2):
            raise RuntimeError("Current 2D/3D wire tracking arrays are inconsistent")
        sampled2 = [points2[0]]
        sampled3 = [points3[0]]
        acc_dist = 0.0
        for p2, next2, p3, next3 in zip(points2[:-1], points2[1:], points3[:-1], points3[1:]):
            if len(sampled2) >= 3 or np.all(p2 == next2):
                break
            length = np.linalg.norm(next2 - p2)
            dist_to_point = TRACKING_SPACING_MM - acc_dist
            acc_dist += length
            while acc_dist >= TRACKING_SPACING_MM and len(sampled2) < 3:
                fraction = dist_to_point / length
                sampled2.append(p2 + (next2 - p2) * fraction)
                sampled3.append(p3 + (next3 - p3) * fraction)
                acc_dist -= TRACKING_SPACING_MM
                # Not in G13/stEVE, which put every point of one long segment at the
                # first one (docs/simulator_changes.md, problem 14). Bit-identical
                # whenever a segment holds at most one point.
                dist_to_point += TRACKING_SPACING_MM
        while len(sampled2) < 3:
            sampled2.append(sampled2[-1])
            sampled3.append(sampled3[-1])
        return np.asarray(sampled2, dtype=np.float32), np.asarray(sampled3, dtype=np.float32)

    def _nearest_lumen(self, point: np.ndarray) -> tuple[float, float, int]:
        relative = point.astype(np.float64) - self._starts
        t = np.clip(np.einsum("ij,ij->i", relative, self._vectors) / self._length2, 0, 1)
        closest = self._starts + t[:, None] * self._vectors
        radius = self._radius_start + t * self._radius_delta
        signed_clearance = radius - np.linalg.norm(point - closest, axis=1)
        # max signed clearance approximates the union of the branch tubes;
        # negative means outside every represented lumen tube.
        best = int(np.argmax(signed_clearance))
        return float(signed_clearance[best]), float(radius[best]), best

    def compute(self, intervention) -> GeometryResult:
        self._cache_vessel(intervention)
        sample2, sample3 = self.sample_tracking_pair(intervention)
        if self.frame == "vessel":
            fluoro = intervention.fluoroscopy
            query3 = tracking3d_to_vessel_cs(sample3, fluoro.image_rot_zx, fluoro.image_center)
        else:
            query3 = sample3
        clearances, radii, segments = [], [], []
        for point in query3:
            clearance, radius, segment = self._nearest_lumen(point)
            clearances.append(clearance)
            radii.append(radius)
            segments.append(segment)
        clearances = np.asarray(clearances, dtype=np.float64)
        tip_segment = segments[0]
        wire = sample2[0].astype(np.float64) - sample2[1].astype(np.float64)
        vessel = self._tangent_2d[tip_segment]
        wire_norm = float(np.linalg.norm(wire))
        vessel_norm = float(np.linalg.norm(vessel))
        valid = float(wire_norm > 1e-6 and vessel_norm > 1e-6)
        if valid:
            wire /= wire_norm
            vessel = vessel / vessel_norm
            sin_delta = float(vessel[0] * wire[1] - vessel[1] * wire[0])
            cos_delta = float(np.clip(np.dot(vessel, wire), -1, 1))
        else:
            # A neutral angle is accompanied by validity=0, so it cannot be
            # confused with a measured aligned configuration.
            sin_delta, cos_delta = 0.0, 1.0
        distal = sample2[0].astype(np.float64) - sample2[1].astype(np.float64)
        proximal = sample2[1].astype(np.float64) - sample2[2].astype(np.float64)
        if np.linalg.norm(distal) > 1e-6 and np.linalg.norm(proximal) > 1e-6:
            bend = float(np.arccos(np.clip(np.dot(distal, proximal) /
                                           (np.linalg.norm(distal) * np.linalg.norm(proximal)), -1, 1)))
        else:
            bend = 0.0
        groups = {
            "clearance": np.tanh(clearances / GEOMETRY_SCALE_MM).astype(np.float32),
            "heading": np.asarray([sin_delta, cos_delta], dtype=np.float32),
            "bend": np.asarray([bend / np.pi], dtype=np.float32),
            "radius": np.asarray([np.tanh(radii[0] / GEOMETRY_SCALE_MM)], dtype=np.float32),
            "valid": np.asarray([valid], dtype=np.float32),
        }
        if not all(np.all(np.isfinite(v)) for v in groups.values()):
            raise FloatingPointError("Nonfinite G13 geometry features")
        return GeometryResult(
            groups=groups, clearance_mm=clearances.astype(np.float32),
            radius_mm=float(radii[0]), bend_angle_rad=bend,
            selected_branch=self._segment_branches[tip_segment],
            selected_segment=self._segment_indices[tip_segment],
            sampled_tracking_2d_mm=sample2, sampled_points_3d_mm=query3,
        )


@register("observation")
class VesselGeometry(Observation):
    """Wire-vs-vessel geometry: wall clearance at 3 distal points, heading
    relative to the vessel, distal bend, local radius, heading validity."""

    def __init__(self, intervention, features: Sequence[str] = G13_FEATURES,
                 frame: str = "reference", name: str = "vessel_geometry") -> None:
        self.name = name
        self.intervention = intervention
        self.features = check_features(features)
        self.frame = frame
        self.core = GeometryCore(frame)
        self.last_result: GeometryResult | None = None
        self.obs = np.zeros(self.size, dtype=np.float32)

    @property
    def size(self) -> int:
        return sum(len(FEATURE_GROUPS[f]) for f in self.features)

    @property
    def feature_names(self) -> List[str]:
        return [n for f in self.features for n in FEATURE_GROUPS[f]]

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-np.ones(self.size, dtype=np.float32),
                              high=np.ones(self.size, dtype=np.float32), dtype=np.float32)

    def step(self) -> None:
        self.last_result = self.core.compute(self.intervention)
        self.obs = assemble(self.last_result.groups, self.features)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()
