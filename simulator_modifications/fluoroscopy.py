"""The tracking spaces of a rotated C-arm (docs/simulator_changes.md, problem 21).

``TrackingOnly.tracking3d_space`` (stEVE ``eve/intervention/fluoroscopy/trackingonly.py``) turns the
vessel tree's bounding box into the image frame by rotating only its two corners ``low`` and
``high``. Rotated, those two points are no longer the low and high corners of anything: a coordinate
of the rotated ``low`` can be larger than the same coordinate of the rotated ``high``, and gymnasium
refuses the box (``ValueError``). On the lab phantom it happens at several of the C-arm angles
``guidewire_rl/env/c_arm/`` chooses (``image_rot_zx: per_target``; the angles are in its README), so a
single-device phantom config (stEVE's ``Tracking2D`` / ``Target2D`` ask for the space when the env
is built) stopped at the build.

The fix rotates all eight corners and takes their bounding box, which contains every rotated point
of the vessel tree's box. Who reads these spaces: stEVE's ``Tracking2D``, ``TrackingInstrument2D``,
``Target2D``, ``Tracking3D``, ``Target3D`` (for their observation space; in our configs only its
shape is used, the values are normalized by ``NormalizeTracking2DEpisode``, problem 15) and stEVE's
image fluoroscopy ``Pillow`` (not used). So no observation value of an existing config changes.

Registered under stEVE's name ``TrackingOnly`` by this module only (a second module registering the
same name would make the config's class depend on import order).
"""

from __future__ import annotations

import itertools

import gymnasium as gym
import numpy as np

from guidewire_rl.env.registry import register
from . import UPSTREAM

if UPSTREAM:
    import eve
    from eve.util.coordtransform import vessel_cs_to_tracking3d

    @register("fluoroscopy", "TrackingOnly")
    class TrackingOnly(eve.intervention.fluoroscopy.TrackingOnly):
        @property
        def tracking3d_space(self) -> gym.spaces.Box:
            low = np.asarray(self.vessel_tree.coordinate_space.low, dtype=np.float64)
            high = np.asarray(self.vessel_tree.coordinate_space.high, dtype=np.float64)
            corners = np.array([[(low, high)[pick][axis] for axis, pick in enumerate(picks)]
                                for picks in itertools.product((0, 1), repeat=3)])
            rotated = vessel_cs_to_tracking3d(corners, self.image_rot_zx, self.image_center,
                                              self.field_of_view)
            return gym.spaces.Box(low=np.min(rotated, axis=0).astype(np.float32),
                                  high=np.max(rotated, axis=0).astype(np.float32))
