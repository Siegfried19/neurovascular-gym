"""Episodes that start from a saved mid-episode state (work log 2026-11-09 22:05).

Why: the start-point curriculum hands the policy an episode after a scripted prefix (over the arch, the
hook placed: about 230 of a 520-step episode, a third of its time, more for short episodes cut off soon
after the handover), and every episode re-simulated that prefix. A snapshot library (``scripts/
make_snapshots.py``) holds the prefix's end states; ``FromSnapshot`` (``env.start``) restores one at reset.

What a state is. BeamAdapter lays the combined beam's nodes out afresh every substep from the devices'
inserted lengths (``InterventionalRadiologyController.xtip``), so the simulation carries no history
beyond: the controller's inserted lengths, rotations and ``indexFirstNode``; the beam nodes' positions
(with orientations) and velocities. The LCP's warm-start table is empty at every episode start anyway
(the scene is rebuilt per reset). Restoring = set the controller's data on a fresh scene, let it lay the
nodes at the start of the first substep, and overwrite the node positions and velocities before that
substep's collision detection (``_RestoreBeam``, a one-shot SOFA controller placed after the IR
controller and before the collision mappings); the substep then runs from the saved state (zero action,
one physics step). Measured (2026-11-09, seed 1000, saved at step 150): 191 nodes laid as saved, the tip
0.004 mm off after the substep, 30 replayed steps within 0.017 -> 0.001 mm of the reference, same forces.

The referee (``PhaseDetector``) is the one part with history (its phase is reached through transitions),
so its state is saved too and put back after its own reset: the start leaves it on the intervention as
``snapshot_referee_state`` and ``PhaseDetector.reset`` picks it up (stEVE resets the start before the
monitors, so the start cannot do it itself). Observation terms with memory (last actions, frame history)
start fresh: the policy is handed the episode there.

A snapshot taken before the wire leaves the catheter (referee phase <= 2) does not depend on the
episode's target (the scripted prefix does not look at it until then), so such snapshots serve any
target; ``max_phase`` keeps the library to those by default.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import Sofa
import Sofa.Core
from eve.start import Start

from ..registry import register
from .phases import PhaseDetector, PhaseState
from .start import WireFirst

SCHEMA = 1


class _RestoreBeam(Sofa.Core.Controller):
    """One shot: right after the IR controller laid the nodes for the restored inserted lengths (the
    first AnimateBeginEvent), before the collision mappings and the collision detection of that
    substep, overwrite the beam nodes' positions and velocities with the saved ones."""

    def __init__(self, dofs, position: np.ndarray, velocity: np.ndarray) -> None:
        super().__init__(name="snapshot_restore")
        self.dofs, self.position, self.velocity = dofs, position, velocity
        self.done, self.nodes_laid = False, None

    def onAnimateBeginEvent(self, _event) -> None:
        if self.done:
            return
        self.done = True
        self.nodes_laid = len(self.dofs.position.value)
        if self.nodes_laid != len(self.position):
            return                                  # the caller sees the mismatch and raises
        with self.dofs.position.writeable() as p:
            p[:] = self.position
        with self.dofs.velocity.writeable() as v:
            v[:] = self.velocity


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def referee_state(monitor: PhaseDetector) -> Dict[str, Any]:
    """The referee's history: its state, the phase candidate it is counting, its arming flags, the
    events reported to it that it has not counted yet."""
    return {"state": _jsonable(asdict(monitor.state)), "candidate": list(monitor._candidate),
            "armed": dict(monitor._armed), "pending_events": dict(monitor.pending_events)}


def apply_referee_state(monitor: PhaseDetector, saved: Dict[str, Any]) -> None:
    state = dict(saved["state"])
    for key in ("wire_tip", "cath_tip", "cath_tip_dir"):
        state[key] = np.asarray(state[key], dtype=float)
    state["cutoff"] = monitor.cutoff                 # the episode's own cutoff, not the saved one
    monitor.state = PhaseState(**state)
    monitor._candidate = tuple(saved["candidate"])
    monitor._armed = dict(saved["armed"])
    monitor.pending_events = {k: int(v) for k, v in saved["pending_events"].items()}


def capture(env, **meta) -> Dict[str, Any]:
    """The restorable state of ``env`` (a GuidewireEnv) now, plus ``meta`` (e.g. seed, step, stage)."""
    sim = env.simulation
    combined = sim._instruments_combined                     # pylint: disable=protected-access
    ctrl = combined.IRcontroller
    snap = {
        "schema": SCHEMA,
        "position": np.array(combined.DOFs.position.value, dtype=float),
        "velocity": np.array(combined.DOFs.velocity.value, dtype=float),
        "xtip": np.array(ctrl.xtip.value, dtype=float).reshape(-1),
        "rotation": np.array(ctrl.rotationInstrument.value, dtype=float).reshape(-1),
        "index_first_node": int(ctrl.indexFirstNode.value),
        "dt": float(sim.dt_simulation),
        "instruments": [[i.name, float(i.length)] for i in sim.instruments],
        "referee": referee_state(env.monitors["phase_detector"]) if "phase_detector" in env.monitors else None,
        "meta": dict(meta),
    }
    if snap["referee"] is not None:
        snap["meta"].setdefault("phase", int(snap["referee"]["state"]["phase"]))
    return snap


def save(path, snap: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, position=snap["position"], velocity=snap["velocity"], xtip=snap["xtip"],
                        rotation=snap["rotation"],
                        header=json.dumps({k: snap[k] for k in ("schema", "index_first_node", "dt", "instruments",
                                                                 "referee", "meta")}))


def load(path) -> Dict[str, Any]:
    with np.load(path) as z:
        header = json.loads(str(z["header"]))
        snap = dict(header, position=np.asarray(z["position"], dtype=float), velocity=np.asarray(z["velocity"], dtype=float),
                    xtip=np.asarray(z["xtip"], dtype=float), rotation=np.asarray(z["rotation"], dtype=float))
    if snap.get("schema") != SCHEMA:
        raise ValueError(f"snapshot {path}: schema {snap.get('schema')}, expected {SCHEMA}")
    return snap


def restore_simulation(sim, snap: Dict[str, Any]) -> Dict[str, Any]:
    """Put ``snap``'s beam state into the (freshly reset) simulation. One ordinary substep of physics
    runs (zero action): the controller lays the nodes at its start, the state is overwritten, and the
    substep carries it on by one physics step (a device moving at 40 mm/s travels 0.16 mm; nothing is
    gained by a shorter substep: the contact constraints correct positions, so a gap of 0.01 mm closed
    in 0.1 ms is a 100 mm/s kick, measured). Raises when the scene does not match the snapshot."""
    combined = sim._instruments_combined                     # pylint: disable=protected-access
    ctrl = combined.IRcontroller
    if abs(float(sim.dt_simulation) - float(snap["dt"])) > 1e-12:
        raise RuntimeError(f"snapshot taken at dt {snap['dt']} s, the scene runs {sim.dt_simulation} s")
    here = [[i.name, float(i.length)] for i in sim.instruments]
    if here != [list(x) for x in snap["instruments"]]:
        raise RuntimeError(f"snapshot instruments {snap['instruments']} != the scene's {here}")
    ctrl.xtip.value = snap["xtip"].tolist()
    ctrl.rotationInstrument.value = snap["rotation"].tolist()
    ctrl.indexFirstNode.value = int(snap["index_first_node"])
    if getattr(sim, "_sheaths", None):
        sim._update_sheathed_starts(snap["xtip"])            # pylint: disable=protected-access
    restorer = _RestoreBeam(combined.DOFs, snap["position"], snap["velocity"])
    combined.addObject(restorer)
    try:
        sim._sofa.Simulation.animate(sim.root, sim.root.dt.value)   # pylint: disable=protected-access
    finally:
        combined.removeObject(restorer)
    if restorer.nodes_laid != len(snap["position"]):
        raise RuntimeError(f"the controller laid {restorer.nodes_laid} nodes for the snapshot's inserted lengths, "
                           f"the snapshot has {len(snap['position'])}: different scene settings")
    sim._update_properties()                                 # pylint: disable=protected-access
    tip = np.asarray(sim.dof_positions[0], dtype=float)
    return {"nodes": int(restorer.nodes_laid),
            "tip_shift_mm": float(np.linalg.norm(tip - snap["position"][-1, :3]))}


@register("start")
class FromSnapshot(Start):
    """``env.start``: with probability ``fraction`` the episode starts from a snapshot of ``library``
    (a directory of ``.npz`` written by ``save``; those with referee phase <= ``max_phase``), picked
    with the simulation's seeded generator; otherwise the ordinary opening (``WireFirst``,
    ``wire_mm``). ``last`` tells which snapshot the current episode started from (None: the opening)."""

    def __init__(self, intervention, library: str, fraction: float = 1.0, max_phase: int = 2,
                 wire_mm: float = 40.0) -> None:
        self.intervention = intervention
        self.library = Path(library)
        self.fraction = float(fraction)
        self.max_phase = int(max_phase)
        self.fallback = WireFirst(intervention, wire_mm=wire_mm)
        self.files: List[Path] = []
        if self.library.exists():
            for path in sorted(self.library.glob("*.npz")):
                meta = load(path)["meta"]
                if meta.get("phase", 0) <= self.max_phase:
                    self.files.append(path)
        if self.fraction > 0.0 and not self.files:
            raise FileNotFoundError(f"FromSnapshot: no snapshot with phase <= {self.max_phase} in {self.library}")
        self.last: Optional[Dict[str, Any]] = None

    def reset(self, episode_nr: int = 0) -> None:
        self.intervention.snapshot_referee_state = None
        self.last = None
        rng = self.intervention.simulation._rng               # pylint: disable=protected-access
        if not self.files or rng.random() >= self.fraction:
            self.fallback.reset(episode_nr)
            return
        path = self.files[int(rng.integers(len(self.files)))]
        snap = load(path)
        self.intervention.reset_instruments()
        report = restore_simulation(self.intervention.simulation, snap)
        self.intervention.snapshot_referee_state = snap["referee"]
        self.intervention.last_action = np.zeros_like(np.asarray(self.intervention.last_action, dtype=np.float64))
        self.last = dict(path=str(path), **snap["meta"], **report)
