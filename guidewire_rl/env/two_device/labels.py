"""Training answers of the two-device task (labels: measured each step, never observed).

``InstrumentTipForce``: contact force on the last ``distal_length_mm`` (3 mm) of one instrument,
summarized over the SOFA substeps of a control step (``statistic``, max by default), as G13's
``DistalForce`` but for any instrument. The two devices are one combined beam in BeamAdapter
(coaxial, sharing nodes where they overlap), and its collision model is one chain whose last node
is the leading tip. An instrument whose tip is x mm behind the leading tip (x = difference of the
inserted lengths, read every substep) has its last 3 mm at [x, x + 3] mm from the leading tip
along that chain; contacts located there (point or line contacts, located exactly as the reference
reconstruction does: ``force_reconstruction.py``) are summed. For the leading instrument this is
G13's distal force. Where the wire is inside the catheter, its "tip force" is the force on the
catheter's wall at that place.

With per-device collision (``SofaBeamAdapter(instrument_collision=True)``, docs/simulator_changes.md
problem 8) each device has its own collision chain, ending at its own tip, and a contact belongs
to the device whose wall touched: the tip force is the contacts on that device's chain within
``distal_length_mm`` of its end. A wire inside the catheter then reads 0 (the catheter's wall keeps
the shared centerline 1 mm off the vessel; the wire's own 0.445 mm never gets there).

``CatheterTipForce`` is ``InstrumentTipForce`` with ``instrument: 1``. ``last_value`` keeps the
latest value (0 after a reset) for the safety shell's force limit.

``TipClearance``: one tip's signed distance to the vessel wall after the step, mm (positive
inside), from the centerline tubes (``route.py``; they follow the model's inner wall to about
0.2 mm, at most 3.7 mm near the route).

``SubstepMotion``: the largest move of any device node in one SOFA substep of the step, mm
(``statistic`` over the substeps, max by default), read from the simulation's ``substep_motion_mm``
(simulator_modifications/device_contact.py: a node's distance to the previous substep's polyline,
so BeamAdapter re-laying the nodes along the device does not count; nodes within 30 mm of the
insertion point are left out). The collision alarm distance is the look-ahead of the contact
handling; a node that moves farther than it plus its radius in one substep can pass a wall it was
not paired with. Needs ``stop_on_nan: true`` (that is the loop that reads the nodes each substep).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import Sofa
from simulator_modifications.device_contact import collision_node_name

from ..labels.base import Label
from ..labels.distal_force import STATISTICS, active_rows, constraint_solver
from ..labels.force_reconstruction import (_WEIGHT_TOL, _ZERO_TOL, _contact_arc_length,
                                           _parse_jacobian, _tip_arc_distances)
from ..registry import register
from .phases import CATH, WIRE, devices_of
from .route import Route


def jacobian_rows(value) -> Dict[int, Dict[int, np.ndarray]]:
    """Contact Jacobian rows {row: {node: xyz}}: SOFA 25.12 gives a SciPy sparse matrix (row =
    constraint, columns 3 * node + xyz), SOFA 23.06 text (parsed as the reference does)."""
    if isinstance(value, str):
        return _parse_jacobian(value)
    matrix = value.tocsr()
    rows: Dict[int, Dict[int, np.ndarray]] = {}
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if start == end:
            continue
        nodes: Dict[int, np.ndarray] = {}
        for column, entry in zip(matrix.indices[start:end], matrix.data[start:end]):
            nodes.setdefault(int(column) // 3, np.zeros(3))[int(column) % 3] = float(entry)
        rows[row] = nodes
    return rows


class _ChainEdges:
    """Edge lengths of the collision chain 0-1-2-...-n, looked up when asked (the mapping
    ``_contact_arc_length`` reads)."""

    def __init__(self, coords: np.ndarray) -> None:
        self.coords = coords

    def __contains__(self, edge) -> bool:
        a, b = sorted(edge)
        return b - a == 1 and a >= 0 and b < len(self.coords)

    def __getitem__(self, edge) -> float:
        a, b = sorted(edge)
        return float(np.linalg.norm(self.coords[a] - self.coords[b]))


def chain_from_tip(coords: np.ndarray, topology: np.ndarray) -> Optional[np.ndarray]:
    """mm from the tip for every collision node when the collision edges are the chain
    [[0, 1], [1, 2], ...] (node n-1 the tip): a cumulative sum. None for any other topology."""
    n = len(coords)
    if n >= 2 and topology.shape == (n - 1, 2) and \
            np.array_equal(np.sort(topology, axis=1), np.column_stack([np.arange(n - 1), np.arange(1, n)])):
        segment = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if not np.all(np.isfinite(segment)):
            raise ValueError("Non-finite guidewire collision edge")
        return np.r_[np.cumsum(segment[::-1])[::-1], 0.0]
    return None


def tip_arc_distances(coords: np.ndarray, topology: np.ndarray):
    """(mm from the tip for every collision node, edge lengths). The beam's collision edges are
    the chain [[0, 1], [1, 2], ...] (node n-1 the tip); then this is a cumulative sum. Any other
    topology goes through the reference walk (``force_reconstruction._tip_arc_distances``)."""
    from_tip = chain_from_tip(coords, topology)
    if from_tip is not None:
        return dict(enumerate(from_tip.tolist())), _ChainEdges(coords)
    distances = _tip_arc_distances(coords, topology)
    edges = {frozenset((int(a), int(b))): float(np.linalg.norm(coords[int(a)] - coords[int(b)]))
             for a, b in topology}
    return distances, edges


def located_contacts(multipliers: np.ndarray, rows, positions: np.ndarray, edges: np.ndarray,
                     dt: float, own_rows_only: bool = False) -> List[Tuple[float, np.ndarray]]:
    """Every active contact as (mm from the end of the collision chain, resultant N).
    The resultant is the sum over the contact's three rows of direction x multiplier / dt, the
    location that of its normal row (``force_reconstruction.reconstruct_distal_contact_force``,
    with the same checks). ``own_rows_only``: the scene has several collision models sharing the
    multipliers; only the contacts with rows in this model's Jacobian (``rows``) are its own."""
    lam = np.asarray(multipliers, dtype=float).reshape(-1)
    if len(lam) == 0:
        return []
    if not np.all(np.isfinite(lam)) or len(lam) % 3:
        raise ValueError("multipliers must be finite frictional row triplets")
    coords = np.asarray(positions, dtype=float)
    topology = np.asarray(edges, dtype=int)
    if own_rows_only:
        normal_rows = sorted(r for r in rows if r % 3 == 0 and r < len(lam))
        if not normal_rows:
            return []
    else:
        normal_rows = range(0, len(lam), 3)
    distances, active_edges = tip_arc_distances(coords, topology)
    out = []
    for normal_row_id in normal_rows:
        normal_lambda = float(lam[normal_row_id])
        if normal_lambda < -_ZERO_TOL:
            raise ValueError("Negative unilateral normal multiplier")
        if normal_lambda <= _ZERO_TOL:
            continue
        normal_row = rows.get(normal_row_id)
        if normal_row is None:
            raise ValueError("Active contact has no collision Jacobian row")
        arc_mm, _support = _contact_arc_length(normal_row, distances, active_edges)
        vector = np.zeros(3)
        for row_id in range(normal_row_id, normal_row_id + 3):
            for direction in rows.get(row_id, {}).values():
                vector += direction * float(lam[row_id]) / dt
        out.append((float(arc_mm), vector))
    return out


def window_force(contacts: List[Tuple[float, np.ndarray]], start_mm: float, length_mm: float) -> float:
    """Magnitude of the summed resultants of the contacts in [start, start + length] mm."""
    total = np.zeros(3)
    for arc, vector in contacts:
        if start_mm - 1e-9 <= arc <= start_mm + length_mm + 1e-9:
            total += vector
    return float(np.linalg.norm(total))


def window_force_fast(multipliers: np.ndarray, matrix, positions: np.ndarray, edges: np.ndarray,
                      dt: float, start_mm: float, length_mm: float,
                      own_rows_only: bool = False) -> Optional[float]:
    """``window_force(located_contacts(multipliers, jacobian_rows(matrix), ...), start, length)``
    without the Python loops over every Jacobian row, for the usual case (SOFA 25.12's sparse
    matrix, the collision edges a chain). Same arithmetic in the same order, so the same number bit
    for bit (tests/test_tip_force_fast.py): each contact's resultant is the sum, over the entries
    of its three rows in the matrix's stored order, of entry x multiplier / dt; the contact sits at
    its normal row's node (one node) or between two adjacent nodes weighted by the row's vectors
    (``force_reconstruction._contact_arc_length``). The same checks raise the same errors. Returns
    None when the inputs are not that case (the caller then takes the reference path)."""
    lam = np.asarray(multipliers, dtype=float).reshape(-1)
    if len(lam) == 0:
        return 0.0
    if not np.all(np.isfinite(lam)) or len(lam) % 3:
        raise ValueError("multipliers must be finite frictional row triplets")
    if not hasattr(matrix, "tocsr"):
        return None
    coords = np.asarray(positions, dtype=float)
    from_tip = chain_from_tip(coords, np.asarray(edges, dtype=int))
    if from_tip is None:
        return None
    csr = matrix.tocsr()
    n_rows, n_nodes = csr.shape[0], len(coords)
    counts = np.diff(csr.indptr)
    present = counts > 0
    row_of = np.repeat(np.arange(n_rows), counts)
    node_of, comp_of, data = csr.indices // 3, csr.indices % 3, np.asarray(csr.data, dtype=float)
    if own_rows_only:
        normal = np.flatnonzero(present[:len(lam):3]) * 3        # this model's normal rows
        if normal.size == 0:
            return 0.0
    else:
        normal = np.arange(0, len(lam), 3)
    if np.any(lam[normal] < -_ZERO_TOL):
        raise ValueError("Negative unilateral normal multiplier")
    active = normal[lam[normal] > _ZERO_TOL]
    if active.size == 0:
        return 0.0
    if np.any(active >= n_rows) or not present[active].all():
        raise ValueError("Active contact has no collision Jacobian row")
    # the normal rows' vectors per node
    contact_of_row = np.full(n_rows + 3, -1)
    contact_of_row[active] = np.arange(len(active))
    on_normal = contact_of_row[row_of] >= 0
    key = row_of[on_normal] * n_nodes + node_of[on_normal]
    unique_key, inverse = np.unique(key, return_inverse=True)
    vectors = np.zeros((len(unique_key), 3))
    vectors[inverse, comp_of[on_normal]] = data[on_normal]
    nonzero = np.linalg.norm(vectors, axis=1) > _ZERO_TOL
    k_contact = contact_of_row[unique_key[nonzero] // n_nodes]
    k_node, k_vec = unique_key[nonzero] % n_nodes, vectors[nonzero]
    per_contact = np.bincount(k_contact, minlength=len(active))
    if np.any(per_contact < 1) or np.any(per_contact > 2):
        raise ValueError("Contact normal must have one or two collision nodes")
    arc = np.empty(len(active))
    single = per_contact == 1
    arc[k_contact[single[k_contact]]] = from_tip[k_node[single[k_contact]]]
    pair = np.flatnonzero(per_contact == 2)
    if pair.size:
        order = np.lexsort((k_node, k_contact))                  # by contact, then node ascending
        kc, kn, kv = k_contact[order], k_node[order], k_vec[order]
        both = np.isin(kc, pair)
        lo, hi = kn[both][0::2], kn[both][1::2]
        v_lo, v_hi = kv[both][0::2], kv[both][1::2]
        if np.any(hi - lo != 1):
            raise ValueError("Line-contact support is not a current collision edge")
        if np.any(np.linalg.norm(coords[hi] - coords[lo], axis=1) <= _ZERO_TOL):
            raise ValueError("Active line contact lies on a degenerate collision edge")
        combined = v_lo + v_hi
        denominator = np.einsum("ij,ij->i", combined, combined)
        if np.any(denominator <= _ZERO_TOL ** 2):
            raise ValueError("Degenerate line-contact normal")
        w_lo = np.einsum("ij,ij->i", v_lo, combined) / denominator
        w_hi = np.einsum("ij,ij->i", v_hi, combined) / denominator
        if not np.allclose(w_lo + w_hi, 1.0, rtol=0, atol=_WEIGHT_TOL):
            raise ValueError("Line-contact interpolation weights do not sum to one")
        for w in (w_lo, w_hi):
            if np.any(w < -_WEIGHT_TOL) or np.any(w > 1.0 + _WEIGHT_TOL):
                raise ValueError("Line-contact interpolation is outside its edge")
        scale = _WEIGHT_TOL * np.linalg.norm(combined, axis=1)
        if np.any(np.linalg.norm(v_lo - w_lo[:, None] * combined, axis=1) > scale) or \
                np.any(np.linalg.norm(v_hi - w_hi[:, None] * combined, axis=1) > scale):
            raise ValueError("Line-contact Jacobian vectors are not collinear")
        arc[kc[both][0::2]] = w_lo * from_tip[lo] + w_hi * from_tip[hi]
    # each contact's resultant: its three rows' entries, in stored order, times multiplier / dt
    for k in (1, 2):
        contact_of_row[active + k] = np.arange(len(active))
    on_contact = contact_of_row[row_of] >= 0
    rows = row_of[on_contact]
    scaled = data[on_contact] * lam[rows] / dt
    resultant = np.zeros((len(active), 3))
    np.add.at(resultant, (contact_of_row[rows], comp_of[on_contact]), scaled)
    inside = (arc >= start_mm - 1e-9) & (arc <= start_mm + length_mm + 1e-9)
    total = np.zeros(3)
    for vector in resultant[inside]:
        total += vector
    return float(np.linalg.norm(total))


class TipForceMonitor(Sofa.Core.Controller):
    """Read-only SOFA callback: after every substep, each registered instrument's tip force.
    One per scene, shared by every ``InstrumentTipForce`` label."""

    NAME = "two_device_tip_force_monitor"

    def __init__(self, simulation) -> None:
        super().__init__(name=self.NAME)
        self.simulation = simulation
        self.windows: Dict[int, float] = {}         # instrument -> distal length, mm
        self.samples: Dict[int, List[float]] = {}
        self.peak_contact_N = 0.0       # largest single contact normal force in any substep of the step
        self.error: Optional[BaseException] = None
        self.enabled = False

    def register(self, instrument: int, distal_length_mm: float) -> None:
        self.windows[instrument] = float(distal_length_mm)
        self.samples.setdefault(instrument, [])

    def begin(self) -> None:
        if self.enabled:
            return                                   # another label already started this step
        self.samples = {k: [] for k in self.windows}
        self.peak_contact_N = 0.0
        self.error = None
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def onAnimateEndEvent(self, _event) -> None:
        if not self.enabled or self.error is not None:
            return
        try:
            sim = self.simulation
            solver = constraint_solver(sim.root)
            combined = sim._instruments_combined           # pylint: disable=protected-access
            multipliers = np.asarray(solver.findData("constraintForces").value, dtype=float).reshape(-1)
            active_rows(solver, multipliers)
            if multipliers.size:            # every contact's normal row is the first of its three
                self.peak_contact_N = max(self.peak_contact_N,
                                          float(np.abs(multipliers[0::3]).max()) / float(sim.root.dt.value))
            if getattr(sim, "instrument_collision", False):
                self._per_device(sim, combined, multipliers)
                return
            collision = combined.getChild("CollisionModel")
            dofs = collision.CollisionDOFs
            matrix = dofs.findData("constraint").value
            positions = np.asarray(dofs.position.value, dtype=float)
            edges = np.asarray(collision.collisEdgeSet.findData("edges").value, dtype=int)
            dt = float(sim.root.dt.value)
            contacts = None
            inserted = np.asarray(combined.IRcontroller.xtip.value, dtype=float).reshape(-1)
            lead = float(np.max(inserted))
            for instrument, length in self.windows.items():
                start = lead - float(inserted[instrument])
                force = self._window_force(multipliers, matrix, positions, edges, dt, start, length, False)
                if force is None:                       # not the fast path's case: the reference walk
                    if contacts is None:
                        contacts = located_contacts(multipliers, jacobian_rows(matrix), positions, edges, dt)
                    force = window_force(contacts, start, length)
                if not np.isfinite(force):
                    raise RuntimeError("Nonfinite tip force")
                self.samples[instrument].append(force)
        except Exception as exc:  # SOFA may log callback errors without raising
            self.error = exc

    # the vectorized evaluation (window_force_fast); tests swap it for a checked version
    _window_force = staticmethod(window_force_fast)

    def _per_device(self, sim, combined, multipliers: np.ndarray) -> None:
        """Each device's own collision chain (module docstring)."""
        dt = float(sim.root.dt.value)
        names = [instrument.name for instrument in sim._instruments]  # pylint: disable=protected-access
        for instrument, length in self.windows.items():
            node = combined.getChild(collision_node_name(names[instrument]))
            dofs = node.CollisionDOFs
            positions = np.asarray(dofs.position.value, dtype=float)
            force = 0.0
            if len(positions) >= 2:
                matrix = dofs.findData("constraint").value
                edges = np.asarray(node.collisEdgeSet.findData("edges").value, dtype=int)
                force = self._window_force(multipliers, matrix, positions, edges, dt, 0.0, length, True)
                if force is None:
                    contacts = located_contacts(multipliers, jacobian_rows(matrix), positions, edges, dt,
                                                own_rows_only=True)
                    force = window_force(contacts, 0.0, length)
            if not np.isfinite(force):
                raise RuntimeError("Nonfinite tip force")
            self.samples[instrument].append(force)


@register("label")
class InstrumentTipForce(Label):
    """Contact force on one instrument's last ``distal_length_mm`` (module docstring)."""

    def __init__(self, instrument: int = WIRE, name: Optional[str] = None, statistic: str = "max",
                 distal_length_mm: float = 3.0, expected_substeps: Optional[int] = None) -> None:
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {list(STATISTICS)}, got {statistic!r}")
        self.instrument = int(instrument)
        self.name = name or f"tip_force_{self.instrument}"
        self.statistic = statistic
        self.distal_length_mm = float(distal_length_mm)
        self.expected_substeps = expected_substeps
        self.monitor: Optional[TipForceMonitor] = None
        self.last_value = 0.0
        self._dt = None

    def attach(self, env) -> None:
        self.last_value = 0.0
        root = env.simulation.root
        monitor = root.getObject(TipForceMonitor.NAME)
        if monitor is None:
            solver = constraint_solver(root)
            if not solver.computeConstraintForces.value:
                solver.computeConstraintForces.value = True
            monitor = TipForceMonitor(env.simulation)
            root.addObject(monitor)
        self.monitor = monitor
        monitor.register(self.instrument, self.distal_length_mm)
        self._dt = float(root.dt.value)

    def begin_step(self) -> None:
        self.monitor.begin()

    def finish_step(self) -> Tuple[float, Dict[str, Any]]:
        monitor = self.monitor
        monitor.stop()
        if getattr(monitor.simulation, "simulation_error", False):
            # stEVE found NaN and threw the devices back to the start: the episode is truncated
            # (SimError); the substeps' numbers mean nothing, so no error is raised either
            self.last_value = 0.0
            return 0.0, {"max_N": 0.0, "mean_N": 0.0, "integral_Ns": 0.0, "contact_substeps": 0,
                         "substeps": len(monitor.samples.get(self.instrument, [])),
                         "simulation_error": True}
        if monitor.error is not None:
            raise RuntimeError("Tip force reconstruction failed") from monitor.error
        samples = monitor.samples.get(self.instrument, [])
        if not samples:
            raise RuntimeError("No SOFA substep completed during the control step")
        if self.expected_substeps is not None and len(samples) != self.expected_substeps:
            raise RuntimeError(f"Expected {self.expected_substeps} SOFA substeps, got {len(samples)}")
        force = np.asarray(samples, dtype=float)
        values = {"max": float(np.max(force)), "mean": float(np.mean(force)),
                  "final": float(force[-1]), "integral": float(np.sum(force) * self._dt)}
        value = values[self.statistic]
        self.last_value = value
        details = {"max_N": values["max"], "mean_N": values["mean"],
                   "integral_Ns": values["integral"],
                   "contact_substeps": int(np.sum(force > 0)), "substeps": len(samples)}
        return value, details

    def abort_step(self) -> None:
        if self.monitor is not None:
            self.monitor.stop()


@register("label")
class CatheterTipForce(InstrumentTipForce):
    """``InstrumentTipForce`` of the catheter (instrument 1)."""

    def __init__(self, name: str = "catheter_tip_force", statistic: str = "max",
                 distal_length_mm: float = 3.0, expected_substeps: Optional[int] = None,
                 instrument: int = CATH) -> None:
        super().__init__(instrument=instrument, name=name, statistic=statistic,
                         distal_length_mm=distal_length_mm, expected_substeps=expected_substeps)


@register("label")
class TipClearance(Label):
    """One tip's signed distance to the vessel wall after the step, mm (module docstring)."""

    def __init__(self, instrument: int = WIRE, name: Optional[str] = None) -> None:
        self.instrument = int(instrument)
        self.name = name or ("wire_tip_clearance" if self.instrument == WIRE else
                             f"tip_clearance_{self.instrument}")
        self.env = None

    def attach(self, env) -> None:
        self.env = env

    def begin_step(self) -> None:
        pass

    def finish_step(self) -> Tuple[float, Dict[str, Any]]:
        intervention = self.env.intervention
        route = Route.of(intervention.vessel_tree)
        tip = devices_of(intervention.simulation).tips[self.instrument]
        location = route.locate(tip)
        return float(location.clearance), {"branch": location.branch}


MOTION_STATISTICS = ("max", "p99", "mean")


@register("label")
class SubstepMotion(Label):
    """Largest node move in one substep of the step, mm (module docstring)."""

    def __init__(self, name: str = "substep_motion", statistic: str = "max") -> None:
        if statistic not in MOTION_STATISTICS:
            raise ValueError(f"statistic must be one of {list(MOTION_STATISTICS)}, got {statistic!r}")
        self.name = name
        self.statistic = statistic
        self.simulation = None

    def attach(self, env) -> None:
        self.simulation = env.simulation

    def begin_step(self) -> None:
        pass

    def finish_step(self) -> Tuple[float, Dict[str, Any]]:
        motion = np.asarray(getattr(self.simulation, "substep_motion_mm", []), dtype=float)
        if motion.size == 0:
            if getattr(self.simulation, "simulation_error", False):
                return 0.0, {"max_mm": 0.0, "p99_mm": 0.0, "mean_mm": 0.0, "substeps": 0,
                             "simulation_error": True}
            raise RuntimeError("SubstepMotion needs the simulation's per-substep loop "
                               "(SofaBeamAdapter with stop_on_nan: true)")
        values = {"max": float(motion.max()), "p99": float(np.percentile(motion, 99)),
                  "mean": float(motion.mean())}
        return values[self.statistic], {"max_mm": values["max"], "p99_mm": values["p99"],
                                        "mean_mm": values["mean"], "substeps": int(motion.size)}
