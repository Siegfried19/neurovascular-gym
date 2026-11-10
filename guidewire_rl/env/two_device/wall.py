"""Where the devices are relative to the vessel's real wall, read off the collision mesh itself.

``WallDistance`` (``env.monitors``): after every control step, the signed distance of every
collision point of every device to the vessel mesh (VTK's implicit distance to the triangles, built
once per mesh; the sign is set so that the insertion point counts as inside, whichever way the
mesh's normals point), minus that device's contact radius: the clearance between the device's wall
and the vessel wall. A contact holds it near 0 (the LCP's tolerance leaves a few hundredths of a
mm); a device whose centerline has crossed the wall (distance < ``breach_mm``, 0) has left the
vessel, which the contact handling cannot undo. Why: the referee's tubes (``route.py``) follow the
model's inner wall only to about 0.2 mm and look at the two tips alone; the 2 ms physics step was
chosen because a lighter wire at a longer step went through the wall (docs/simulator_changes.md,
problem 20), so the wall itself is the thing to watch, in every run. About 7 ms a step for 300
points. Without per-device collision (``instrument_collision`` off) the combined beam's nodes are
used with radius 0.

``WallPenetration`` (``env.labels``): the deepest penetration of the step, mm (0 when every device
is clear or just touching), the per-device minimum clearance in the details.
``WallBreach`` (``env.truncation``): the episode is truncated once a device's centerline is outside
(the simulation is wrong from here on, as with ``SimError``). ``WallInfo`` (``env.info``) reports it
as ``info["wall_breached"]``; the trainers drop that step and close the episode as ``breach`` (no
terminal flag, so the value of the step before is bootstrapped; no -1: a breach is the simulator's
failure, not the policy's, and "do not press on the wall" is taught by the force penalty; user
decision 2026-11-09). The referee's "a tip 6 mm outside every centerline tube" hard failure stays
as a backstop behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from eve.info import Info
from eve.truncation import Truncation
from simulator_modifications.device_contact import collision_node_name, instrument_radius

from ..labels.base import Label
from ..registry import register


@dataclass
class WallState:
    penetration_mm: float = 0.0            # deepest "device wall past the vessel wall" this step, >= 0
    min_distance_mm: float = 0.0           # closest centerline point to the wall, signed (+ inside)
    clearance_mm: Dict[str, float] = field(default_factory=dict)   # per device: min(distance - radius)
    breached: bool = False
    points: int = 0


class _MeshDistance:
    """Signed distance to a triangle mesh, positive inside once calibrated at a known inside point."""

    def __init__(self, mesh_path: str, inside_point) -> None:
        import pyvista as pv
        import vtk
        self.mesh_path = mesh_path
        self._function = vtk.vtkImplicitPolyDataDistance()
        self._function.SetInput(pv.read(mesh_path).triangulate())
        self.sign = 1.0
        if self._function.EvaluateFunction(np.asarray(inside_point, dtype=float).reshape(3).tolist()) < 0.0:
            self.sign = -1.0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return self.sign * np.fromiter((self._function.EvaluateFunction(p.tolist()) for p in pts),
                                       dtype=float, count=len(pts))


@register("monitor")
class WallDistance:
    """The devices' clearance to the vessel mesh after each step (module docstring)."""

    def __init__(self, intervention, breach_mm: float = 0.0) -> None:
        self.intervention = intervention
        self.breach_mm = float(breach_mm)
        self._distance: Optional[_MeshDistance] = None
        self.state = WallState()
        self.episode_max_penetration_mm = 0.0

    def reset(self, episode_nr: int = 0) -> None:
        tree = self.intervention.vessel_tree
        if self._distance is None or self._distance.mesh_path != tree.mesh_path:
            self._distance = _MeshDistance(tree.mesh_path, tree.insertion.position)
        self.episode_max_penetration_mm = 0.0
        self.step()

    def step(self) -> None:
        state = WallState(min_distance_mm=np.inf)
        for name, points, radius in self._device_points(self.intervention.simulation):
            if len(points) == 0 or not np.isfinite(points).all():
                continue
            distance = self._distance(points)
            state.clearance_mm[name] = float(np.min(distance - radius))
            state.min_distance_mm = min(state.min_distance_mm, float(distance.min()))
            state.points += len(points)
        if state.clearance_mm:
            state.penetration_mm = max(0.0, -min(state.clearance_mm.values()))
            state.breached = state.min_distance_mm < self.breach_mm
        else:
            state.min_distance_mm = 0.0
        self.episode_max_penetration_mm = max(self.episode_max_penetration_mm, state.penetration_mm)
        self.state = state

    @staticmethod
    def _device_points(simulation) -> Iterator[Tuple[str, np.ndarray, float]]:
        combined = simulation._instruments_combined           # pylint: disable=protected-access
        if getattr(simulation, "instrument_collision", False):
            for instrument in simulation.instruments:
                node = combined.getChild(collision_node_name(instrument.name))
                yield (instrument.name, np.asarray(node.CollisionDOFs.position.value, dtype=float)[:, :3],
                       instrument_radius(instrument))
        else:
            yield "combined", np.asarray(combined.DOFs.position.value, dtype=float)[:, :3], 0.0


@register("label")
class WallPenetration(Label):
    """Deepest penetration of a device wall past the vessel wall this step, mm (module docstring)."""

    def __init__(self, wall_distance: WallDistance, name: str = "wall_penetration") -> None:
        self.wall = wall_distance
        self.name = name

    def begin_step(self) -> None:
        pass

    def finish_step(self) -> Tuple[float, Dict[str, Any]]:
        state = self.wall.state
        return float(state.penetration_mm), {
            "min_distance_mm": float(state.min_distance_mm),
            "clearance_mm": {k: float(v) for k, v in state.clearance_mm.items()},
            "breached": bool(state.breached), "points": int(state.points)}


@register("truncation")
class WallBreach(Truncation):
    """Truncates the episode once a device's centerline is outside the vessel mesh."""

    def __init__(self, wall_distance: WallDistance) -> None:
        self.wall = wall_distance

    @property
    def truncated(self) -> bool:
        return bool(self.wall.state.breached)

    def step(self) -> None:
        pass

    def reset(self, episode_nr: int = 0) -> None:
        pass


@register("info")
class WallInfo(Info):
    """``info["wall_breached"]``: a device's centerline is outside the vessel mesh after this step
    (``WallDistance.state.breached``). The trainers drop such a step and close the episode as
    ``breach`` (module docstring)."""

    def __init__(self, wall_distance: WallDistance, name: str = "wall") -> None:
        super().__init__(name)
        self.wall = wall_distance

    @property
    def info(self) -> Dict[str, Any]:
        return {"wall_breached": bool(self.wall.state.breached)}
