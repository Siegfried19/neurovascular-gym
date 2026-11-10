"""Two fixes to stEVE's 2D tracking observations (docs/simulator_changes.md, problems 14, 15).

Problem 14, ``Tracking2D`` sampling: points are taken every ``resolution`` mm
along the projected wire, but when one segment between two wire nodes holds
several sampling points, stEVE never advances the distance into the segment
and puts all of them at the first one. The fix advances it. Where no segment
holds two points (the usual case: Angled's tip nodes are ~0.5 mm apart) the
result is bit-identical to stEVE's.

Problem 15, ``NormalizeTracking2DEpisode`` range: stEVE maps the projected
centerline's bounding box widened by 10 % *of each coordinate's absolute value*
to [-1, 1], so the margin depends on where the origin is. With the vessel start
at the origin (synthetic arch, lab phantom) one side gets no margin at all, and
on the phantom the wire next to the aortic wall falls outside [-1, 1]. The fix
uses the bounding box of the lumen tubes instead: every centerline point
widened by its own radius, which is everything inside the vessel model. A vessel
tree that names its ``task_branch_names`` (the lab phantom with every vessel, where
the head and arms have centerlines too) is boxed by those branches only.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from guidewire_rl.env.registry import register
from . import UPSTREAM


def lumen_bounds_2d(vessel_tree, fluoroscopy) -> gym.spaces.Box:
    """Image-plane bounding box of every (task) branch's centerline ± radius."""
    from eve.util.coordtransform import tracking3d_to_2d, vessel_cs_to_tracking3d

    if fluoroscopy.field_of_view is not None:
        raise NotImplementedError("lumen_bounds_2d does not handle a limited field_of_view")
    branches = vessel_tree.branches
    task = getattr(vessel_tree, "task_branch_names", None)
    if task is not None:
        branches = [branch for branch in branches if branch.name in task]
    if not branches or any(getattr(branch, "radii", None) is None for branch in branches):
        raise ValueError("lumen_bounds_2d needs branches with radii (BranchWithRadii)")
    points = np.concatenate([branch.coordinates for branch in branches])
    radii = np.concatenate([branch.radii for branch in branches]).astype(np.float64)[:, None]
    # A rotated sphere of radius r still projects to a disc of radius r.
    projected = tracking3d_to_2d(vessel_cs_to_tracking3d(
        points, fluoroscopy.image_rot_zx, fluoroscopy.image_center, None)).astype(np.float64)
    low = np.min(projected - radii, axis=0)
    high = np.max(projected + radii, axis=0)
    return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))


if UPSTREAM:
    import eve

    @register("observation", "Tracking2D")
    class Tracking2D(eve.observation.Tracking2D):
        def _evenly_distributed_tracking(self, tracking: np.ndarray) -> np.ndarray:
            # stEVE's method with one added line (marked).
            tracking = list(tracking)
            tracking_state = [tracking[0]]
            if self.n_points > 1:
                acc_dist = 0.0
                for point, next_point in zip(tracking[:-1], tracking[1:]):
                    if len(tracking_state) >= self.n_points or np.all(point == next_point):
                        break
                    length = np.linalg.norm(next_point - point)
                    dist_to_point = self.resolution - acc_dist
                    acc_dist += length
                    while acc_dist >= self.resolution and len(tracking_state) < self.n_points:
                        unit_vector = (next_point - point) / length
                        tracking_point = point + unit_vector * dist_to_point
                        tracking_state.append(tracking_point)
                        acc_dist -= self.resolution
                        dist_to_point += self.resolution  # fix: next point one spacing further on
                while len(tracking_state) < self.n_points:
                    tracking_state.append(tracking_state[-1])
            return np.array(tracking_state, dtype=np.float32)

    @register("observation_wrapper", "NormalizeTracking2DEpisode")
    class NormalizeTracking2DEpisode(eve.observation.wrapper.NormalizeTracking2DEpisode):
        def reset(self, episode_nr: int = 0) -> None:
            self._normalization_space = lumen_bounds_2d(self.intervention.vessel_tree,
                                                        self.intervention.fluoroscopy)
            eve.observation.wrapper.Normalize.reset(self, episode_nr)
