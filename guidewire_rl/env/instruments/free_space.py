"""Instruments pushed into empty space: BeamAdapter with no vessel, to see the devices on their own
(a guidewire passing through a catheter, a catheter taking its shape, ...).

The simulation is the training's (``simulator_modifications.device_contact.SofaBeamAdapter``) with the
physics of ``configs/two_devices_real_mass.yaml`` by default; the "vessel" is one triangle far away,
so nothing but the two devices can touch. Gravity is off, as in every stEVE scene. Insertion at the
origin along +x; each reset builds a fresh scene with both devices drawn back to 0 and turned to 0
(at turn 0 the shaped ends bend in the x-y plane, towards +y).

    bench = FreeSpace([GlidewireStiff(), CordisSidewinder2()])
    bench.reset()
    bench.step([[10.0, 0.0], [20.0, 0.0]])        # mm/s and rad/s per device, one control step
    bench.inserted_lengths, bench.dof_positions   # mm; (N, 3) tip first, as stEVE's

Upstream stEVE only (the ``guidewire_rl`` environment).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

# two_devices_real_mass.yaml's simulation (friction is moot: there is no wall to rub on)
TRAINING_PHYSICS: Dict[str, Any] = dict(
    friction=0.1, dt_simulation=0.004, alarm_distance=1.0, instrument_collision=True,
    min_beam_length=0.05, stop_on_nan=True, densest_sampling=True, trim_sheathed=True)
STEP_SECONDS = 1.0 / 7.5                       # one control step, as the environments
FAR_MM = 5000.0


class FreeSpace:
    def __init__(self, instruments: Sequence, physics: Optional[Mapping[str, Any]] = None,
                 step_seconds: float = STEP_SECONDS) -> None:
        from simulator_modifications.device_contact import SofaBeamAdapter

        self.instruments = list(instruments)
        if not self.instruments:
            raise ValueError("FreeSpace needs at least one instrument")
        kwargs = dict(TRAINING_PHYSICS)
        kwargs.update({k: v for k, v in dict(physics or {}).items() if k != "type"})
        if len(self.instruments) < 2:                 # the sheath trimming is for two coaxial devices
            kwargs["trim_sheathed"] = False
        self.physics = kwargs
        self.step_seconds = float(step_seconds)
        self.simulation = SofaBeamAdapter(**kwargs)
        self._dir = tempfile.mkdtemp(prefix="free_space_")
        self._mesh = os.path.join(self._dir, "far.obj")
        with open(self._mesh, "w") as fh:
            fh.write(f"v {-FAR_MM} {-FAR_MM} {-FAR_MM}\nv {-FAR_MM - 1} {-FAR_MM} {-FAR_MM}\n"
                     f"v {-FAR_MM} {-FAR_MM - 1} {-FAR_MM}\nf 1 2 3\n")

    def reset(self, seed: Optional[int] = None) -> None:
        self.simulation.reset(insertion_point=np.zeros(3), insertion_direction=np.array([1.0, 0.0, 0.0]),
                              mesh_path=self._mesh, instruments=self.instruments,
                              coords_high=np.full(3, 2 * FAR_MM), coords_low=np.full(3, -2 * FAR_MM),
                              seed=seed)
        # stEVE starts each device at a random turn (sofabeamadapter.py, _add_instruments); here at 0, so
        # the shaped ends bend in the x-y plane, towards +y (nothing is inserted yet, so nothing moves)
        controller = self.simulation._instruments_combined.IRcontroller
        controller.rotationInstrument.value = np.zeros_like(np.asarray(controller.rotationInstrument.value))
        self.simulation._update_properties()

    def step(self, velocities) -> None:
        """One control step at ``velocities`` ((n, 2): mm/s, rad/s per device)."""
        action = np.array(velocities, dtype=np.float64).reshape(len(self.instruments), 2)
        self.simulation.step(action, duration=self.step_seconds)

    @property
    def dof_positions(self) -> np.ndarray:
        return np.asarray(self.simulation.dof_positions, dtype=np.float64)

    @property
    def inserted_lengths(self) -> np.ndarray:
        return np.asarray(self.simulation.inserted_lengths, dtype=np.float64)

    @property
    def rotations(self) -> np.ndarray:
        return np.asarray(self.simulation.rotations, dtype=np.float64)

    @property
    def simulation_error(self) -> bool:
        return bool(getattr(self.simulation, "simulation_error", False))

    def close(self) -> None:
        self.simulation.close()
        try:
            os.remove(self._mesh)
            os.rmdir(self._dir)
        except OSError:
            pass
