"""Distal guidewire contact force (G13's force label).

``DistalForceMonitor`` is copied from the reference
``contact_aware_tdmpc2/env/force_extraction.py``. Changes: it takes the
simulation object directly, reads the SOFA substep ``dt`` from the scene
instead of a constant, and the distal length and expected substep count are
parameters (G13: 3 mm, 22 substeps).

Which constraint solver: Wilson's eve uses ``GenericConstraintSolver`` named
``Generic`` (G13; read exactly as the reference does). Upstream stEVE on
SOFA 25.12 uses ``LCPConstraintSolver`` named ``LCP``. With
``computeConstraintForces`` on, LCP copies the multipliers it has just applied
into ``constraintForces`` (LCPConstraintSolver.cpp:233-240); the dynamics are
unchanged. It has no ``currentNumConstraints``, so its row count is the vector
length, which must be whole frictional triplets. Whether LCP's numbers equal
Generic's is docs/simulator_changes.md problems 11 and 12.

The collision DOFs' ``constraint`` data (contact Jacobian) is text in SOFA
23.06's Python bindings and a SciPy sparse matrix (row = constraint, columns
3*node + xyz) in SOFA 25.12's; ``jacobian_text`` turns the latter into the
former so the verbatim reference reconstruction reads both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import Sofa

from ..registry import register
from .base import Label
from .force_reconstruction import reconstruct_distal_contact_force


@dataclass(frozen=True)
class StepForce:
    maximum_N: float
    mean_N: float
    final_N: float
    integral_Ns: float
    contact_count: int
    contact_substeps: int
    max_vector_N: tuple[float, float, float]
    substep_count: int
    max_constraint_rows: int
    max_contact_groups: int


def jacobian_text(value) -> str:
    """SOFA's MapMapSparseMatrix rows as text: ``row count (node x y z)*`` per line."""
    if isinstance(value, str):
        return value
    matrix = value.tocsr()
    lines = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue
        nodes: Dict[int, np.ndarray] = {}
        for column, entry in zip(matrix.indices[start:end], matrix.data[start:end]):
            nodes.setdefault(int(column) // 3, np.zeros(3))[int(column) % 3] = float(entry)
        lines.append(" ".join([str(row), str(len(nodes))] + [
            f"{node} {float(vector[0])!r} {float(vector[1])!r} {float(vector[2])!r}"
            for node, vector in sorted(nodes.items())]))
    return "\n".join(lines)


SOLVER_NAMES = ("Generic", "LCP")


def constraint_solver(root):
    """The scene's Lagrangian constraint solver (see module docstring)."""
    for name in SOLVER_NAMES:
        solver = root.getObject(name)
        if solver is not None:
            return solver
    raise RuntimeError(f"No constraint solver named any of {SOLVER_NAMES} in the scene root")


def active_rows(solver, multipliers: np.ndarray) -> Tuple[int, int]:
    """(constraint rows, contact groups) of the current substep, checked."""
    if solver.findData("currentNumConstraints") is not None:          # Generic
        n_rows = int(solver.findData("currentNumConstraints").value)
        n_groups = int(solver.findData("currentNumConstraintGroups").value)
        if len(multipliers) != n_rows or n_rows != 3*n_groups:
            raise RuntimeError(
                f"Generic active-contact row mismatch: {len(multipliers)}, "
                f"{n_rows}, {n_groups}")
        return n_rows, n_groups
    n_rows = len(multipliers)                                             # LCP
    if n_rows % 3:
        raise RuntimeError(f"{solver.getName()}: {n_rows} constraint rows are not "
                           "whole frictional triplets")
    return n_rows, n_rows // 3


class DistalForceMonitor(Sofa.Core.Controller):
    """Read-only callback. ``begin_step``/``finish_step`` bracket one env.step."""

    def __init__(self, simulation, distal_length_mm: float,
                 expected_substeps: Optional[int]):
        super().__init__(name="g2_passive_distal_force_monitor")
        self.simulation = simulation
        self.distal_length_mm = float(distal_length_mm)
        self.expected_substeps = expected_substeps
        self.enabled = False
        self.samples = []
        self.error = None
        self.total_substeps = 0

    def begin_step(self) -> None:
        if self.enabled:
            raise RuntimeError("Previous control step was not finished")
        self.samples = []
        self.error = None
        self.enabled = True

    def abort_step(self) -> None:
        self.enabled = False
        self.samples = []
        self.error = None

    def onAnimateEndEvent(self, _event) -> None:
        if not self.enabled or self.error is not None:
            return
        try:
            sim = self.simulation
            solver = constraint_solver(sim.root)
            collision = sim._instruments_combined.getChild("CollisionModel")
            multipliers = np.asarray(solver.findData("constraintForces").value,
                                     dtype=float).reshape(-1)
            n_rows, n_groups = active_rows(solver, multipliers)
            dofs = collision.CollisionDOFs
            points = np.asarray(dofs.position.value, dtype=float)
            edges = np.asarray(collision.collisEdgeSet.findData("edges").value,
                               dtype=int)
            sample = reconstruct_distal_contact_force(
                multipliers, jacobian_text(dofs.findData("constraint").value),
                points, edges, float(sim.root.dt.value), self.distal_length_mm)
            if not np.isfinite(sample.magnitude_N) or sample.magnitude_N < 0:
                raise RuntimeError("Nonfinite or negative distal force")
            self.samples.append((sample, n_rows, n_groups))
            self.total_substeps += 1
        except Exception as exc:  # SOFA may log callback errors without raising.
            self.error = exc

    def finish_step(self) -> StepForce:
        self.enabled = False
        if self.error is not None:
            raise RuntimeError("Passive substep force reconstruction failed") from self.error
        if not self.samples:
            raise RuntimeError("No SOFA substep completed during the control step")
        if self.expected_substeps is not None and len(self.samples) != self.expected_substeps:
            raise RuntimeError(
                f"Expected {self.expected_substeps} SOFA substeps, got {len(self.samples)}")
        force = np.asarray([row[0].magnitude_N for row in self.samples], dtype=float)
        if not np.all(np.isfinite(force)):
            raise RuntimeError("Nonfinite per-substep force")
        argmax = int(np.argmax(force))
        max_sample = self.samples[argmax][0]
        dt = float(self.simulation.root.dt.value)
        return StepForce(
            maximum_N=float(np.max(force)), mean_N=float(np.mean(force)),
            final_N=float(force[-1]), integral_Ns=float(np.sum(force)*dt),
            contact_count=max(row[0].contact_count for row in self.samples),
            contact_substeps=sum(row[0].contact_count > 0 for row in self.samples),
            max_vector_N=tuple(float(x) for x in max_sample.vector_N),
            substep_count=len(self.samples),
            max_constraint_rows=max(row[1] for row in self.samples),
            max_contact_groups=max(row[2] for row in self.samples),
        )


STATISTICS = {
    "max": lambda f: f.maximum_N,
    "mean": lambda f: f.mean_N,
    "final": lambda f: f.final_N,
    "integral": lambda f: f.integral_Ns,
}


@register("label")
class DistalForce(Label):
    """Contact force on the distal ``distal_length_mm`` of the wire, summarized
    over the SOFA substeps of one control step (G13: ``statistic: max``)."""

    def __init__(self, name: str = "distal_force", statistic: str = "max",
                 distal_length_mm: float = 3.0,
                 expected_substeps: Optional[int] = None) -> None:
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {list(STATISTICS)}, got {statistic!r}")
        self.name = name
        self.statistic = statistic
        self.distal_length_mm = float(distal_length_mm)
        self.expected_substeps = expected_substeps
        self.monitor: Optional[DistalForceMonitor] = None

    def attach(self, env) -> None:
        if getattr(env.simulation, "instrument_collision", False):
            raise ValueError("DistalForce reads stEVE's shared collision model; with per-device "
                             "collision (instrument_collision) use InstrumentTipForce")
        root = env.simulation.root
        if self.monitor is not None and root.getObject(self.monitor.name.value) is self.monitor:
            return                                   # same scene as last reset
        solver = constraint_solver(root)
        if not solver.computeConstraintForces.value:
            solver.computeConstraintForces.value = True
        self.monitor = DistalForceMonitor(env.simulation, self.distal_length_mm,
                                          self.expected_substeps)
        env.simulation.root.addObject(self.monitor)

    def begin_step(self) -> None:
        self.monitor.begin_step()

    def finish_step(self) -> Tuple[float, Dict[str, Any]]:
        force = self.monitor.finish_step()
        details = {
            "max_N": force.maximum_N, "mean_N": force.mean_N,
            "integral_Ns": force.integral_Ns,
            "contact_substeps": force.contact_substeps,
            "substeps": force.substep_count,
        }
        return float(STATISTICS[self.statistic](force)), details

    def abort_step(self) -> None:
        if self.monitor is not None:
            self.monitor.abort_step()
