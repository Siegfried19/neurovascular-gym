"""Seeded resets on Python 3.12 (docs/simulator_changes.md, problem 7).

``MonoPlaneStatic.reset`` draws the vessel and target seeds with a NumPy
generator (``intervention/monoplanestatic.py:110,123``), so they arrive as
NumPy integers. Python 3.12's ``random.Random`` only accepts None, int, float,
str, bytes and bytearray and raises TypeError, so every seeded reset failed.
Three stEVE classes seed ``random.Random`` this way; each gets the seed as a
plain int. ``int(seed)`` has the same value, so the random draws are the same
ones Python 3.10 produced.

``Manual`` also fixes two gaps that kept fixed targets from working (problem 19);
see its docstring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from guidewire_rl.env.registry import register
from . import UPSTREAM


def plain_seed(seed) -> Optional[int]:
    return None if seed is None else int(seed)


if UPSTREAM:
    import eve
    from eve.util.coordtransform import tracking3d_to_2d

    @register("target", "CenterlineRandom")
    class CenterlineRandom(eve.intervention.target.CenterlineRandom):
        """Also: stEVE drops candidate points that lie in the lumen of any branch not
        listed as a target branch. On a vessel tree that names its ``task_branch_names``
        (the lab phantom with every vessel) only those branches count, so the extra
        vessels, which touch the target branches at their ends and ostia, leave the
        targets as they were."""

        def reset(self, episode_nr=0, seed=None) -> None:
            super().reset(episode_nr, plain_seed(seed))

        def _init_centerline_point_cloud(self):
            task = getattr(self.vessel_tree, "task_branch_names", None)
            if task is None:
                return super()._init_centerline_point_cloud()
            # stEVE's method with the branch set narrowed to the task branches
            names = set(task)
            branch_keys = sorted((set(self.branches) if self.branches is not None else names) & names)
            excluded = names - set(branch_keys)
            points = np.vstack([np.empty((0, 3))] + [self.vessel_tree[b].coordinates for b in branch_keys])
            self._potential_targets = points[np.invert(self._in_excluded_branches(points, excluded))]

    @register("target", "Manual")
    class Manual(eve.intervention.target.Manual):
        """Also: stEVE's ``vessel_cs_to_tracking3d`` transposes the target point, so the
        lists a YAML config gives failed at the first reset; they are made arrays here.
        And stEVE sets no ``coordinates2d``, which the ``Target2D`` observation reads; it
        is derived from ``coordinates3d`` as in ``CenterlineRandom``."""

        def __init__(self, targets_vessel_cs, threshold, fluoroscopy) -> None:
            super().__init__([np.asarray(p, dtype=np.float64) for p in targets_vessel_cs],
                             threshold, fluoroscopy)

        @property
        def coordinates2d(self) -> np.ndarray:
            return tracking3d_to_2d(self.coordinates3d)

        def reset(self, episode_nr: int = 0, seed=None) -> None:
            super().reset(episode_nr, plain_seed(seed))

    @register("vessel_tree", "AorticArchRandom")
    class AorticArchRandom(eve.intervention.vesseltree.AorticArchRandom):
        def reset(self, episode_nr=0, seed=None) -> None:
            super().reset(episode_nr, plain_seed(seed))
