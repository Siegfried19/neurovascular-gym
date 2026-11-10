"""Devices that touch the wall with their own thickness, and two devices at their real masses
(docs/simulator_changes.md, problems 8 and 22).

Problem 8, contact ignores how thick a device is. stEVE puts every device on one collision model
(``MultiAdaptiveBeamMapping`` over the combined beam's centerline, contact distance 0), so only the
global ``LocalMinDistance.contactDistance`` (0.3 mm) keeps a centerline off the wall: the same for
a 0.89 mm wire and a 2 mm catheter, whose wall then sinks 0.7 mm into the vessel.
``instrument_collision: true`` gives each device its own collision node instead, written the way
stEVE writes the devices' visual models: an ``AdaptiveBeamMapping`` on that device's own
``Interpol_<name>`` (it maps the beams the controller gives this device, one point per node, so the
points end at this device's tip and follow every push and pull), a line and a point collision model
whose ``contactDistance`` is the device's outer radius, all in collision group 1 so the devices
never collide with each other; the global contact distance goes to 0. Measured on a flat wall: the
catheter's centerline stops at 1.000 mm, the wire at its radius (tests/test_device_contact.py).
A device's points follow its beams, which the controller changes at the start of a substep (a node
coming out at the base shifts the numbering of all the others); stock ``AdaptiveBeamMapping``
re-maps them only during the free motion, after the collision detection has used last substep's
points, so a contact's force went to another point (the neighbouring node in 2.8 % of the contacts
with force, measured 2026-11-03) or past the end of the new list. And the line model re-reads its
edges only when the topology's revision changes, which the mapping's rewrite of the edges does not
do, so the devices' segments never collided (only their nodes did). Our BeamAdapter build fixes
both for a listening mapping (``beamadapter/adaptive_mapping_refresh.patch``): it re-maps right
after the controller, and rebuilds the edges, bumping the revision, when their number changed. So
the mappings here listen.

Problem 22, two devices at their real masses go NaN. BeamAdapter lays the combined beam's nodes
out per device and drops a node only when it is closer than its ``threshold`` (0.01 mm) to the
previous one. When the trailing device's tip, or one of its section ends, lands 0.010-0.013 mm from
a node of the other device, a beam that short is simulated; bending stiffness goes with 1/L^3, so it
is ~10^5 times as stiff as a 0.5 mm beam. Measured 2026-11-03 (two devices, real masses, 2 ms,
random pushes and pulls): every divergence happened on such a substep while a device touched the
wall; the node re-sampling and the free motion were still finite, the contact force solve (LCP)
returned 10^234 or NaN. Without the wall: no divergence in 1384 steps with such beams in 468 of
them. ``min_beam_length`` sets ``minBeamLength`` of our BeamAdapter build
(``beamadapter/min_beam_length.patch``): when the nodes are laid out, a node closer than this to
the previous one is dropped, exactly as ``threshold`` does, while ``threshold``'s other uses
(re-interpolation, the push bookkeeping) keep 0.01 mm. The device whose node is dropped ends up to
that much short of its commanded tip (0.05 mm in configs/two_devices_real_mass.yaml).

``stop_on_nan``: stEVE notices NaN only after all the substeps of a control step, and a NaN scene
hangs in the next collision detection (one process went past 16 GB in 11 minutes, 2026-10-30).
With this on, every substep is checked and on the first non-finite node (or one blown 10 m away:
the contact solve once returned 10^234 instead of NaN) the step stops the way
stEVE stops it (``simulation_error`` and the devices back to the start): the episode is truncated
(``SimError``) instead of the worker hanging.

``densest_sampling``: where the devices overlap, BeamAdapter lays the nodes at the spacing of the
first device listed (the guidewire) and the catheter's own nodes are left out. Its head is laid every
1 mm, the wire's shaft every 10 mm; with the wire more than its 30 mm soft part out of the catheter,
the hook (130 degrees within ~26 mm) sat on the shaft and was 3-4 nodes. With a device's own 1 mm wall
radius pressed into the wall, the corner of that polygon was driven into the wall until the contact
force ran away (2026-11-03: 0.6 -> 3109 N within a control step, the wire thrown 15-21 mm out of the
vessel; 2 of 3 full scripted episodes). Our BeamAdapter build's ``densestSampling``
(``beamadapter/densest_sampling.patch``) lays each stretch at the densest spacing among the device
sections covering it (a tie keeps the first device, i.e. BeamAdapter's own layout).

``substep_motion_mm`` (read-only, filled by the ``stop_on_nan`` loop, which reads the nodes after
every substep anyway): for each substep of the last control step, the largest move of any node, mm,
measured as the node's distance to the previous substep's polyline, over the nodes farther than
``motion_skip_mm`` (30) from the insertion point. Measured that way because BeamAdapter re-lays the
nodes as the devices move: a node shifts by a spacing along the chain (1 mm where the catheter's
head overlaps the wire), the ones near the base jump 6 mm when they are renumbered (work log
2026-11-07 00:20); both are bookkeeping, not motion, and both stay on the old polyline. What counts
is the body moving off its previous line, and the leading tip's advance. The 2 ms step was chosen
because the real-mass wire went through the wall at 6 ms (problem 20); how far a node really moves in
a substep, against the alarm distance, is then something to measure in every run (the
``SubstepMotion`` label, guidewire_rl/env/two_device/labels.py) rather than infer from old data.

``trim_sheathed`` (problem 26): a device that runs inside another (the wire in the catheter; BeamAdapter
keeps them coaxial, so the thinner one is strictly inside the thicker one's collision body wherever they
overlap) gets no collision points on that part: each substep its chain's first point is moved to the
sheath's tip (our BeamAdapter build's ``AdaptiveBeamMapping.startCurvAbs``,
``beamadapter/collision_start_abscissa.patch``). Those points could never touch the wall, but every
substep they were tested against the mesh and, within the alarm distance, registered as constraint rows
that never carried force (docs/simulation_speed.md). The chain still ends at the device's tip, so the
force labels and the wall monitor read it as before; a device fully inside its sheath has an empty chain.

The five switches default off: an existing config builds the same scene as before.
"""

from __future__ import annotations

from typing import List

import numpy as np

from guidewire_rl.env.registry import register
from . import UPSTREAM

COLLISION_GROUP = 1
MAX_COORDINATE_MM = 1e4      # stop_on_nan: a node 10 m away has blown up even if it is still finite


def collision_node_name(instrument_name: str) -> str:
    """The child of ``InstrumentCombined`` that holds one device's collision model."""
    return "CollisionModel_" + instrument_name


def instrument_radius(instrument) -> float:
    """Contact radius of a device, mm: half its largest outer diameter."""
    sections = [s for s in (instrument.base_section, instrument.tip_section) if s is not None]
    return max(float(s.diameter_outer) for s in sections) / 2.0


if UPSTREAM:
    from .speedups import SofaBeamAdapter as SpeedupSofaBeamAdapter

    @register("simulation", "SofaBeamAdapter")
    class SofaBeamAdapter(SpeedupSofaBeamAdapter):
        """``speedups.SofaBeamAdapter`` plus per-device collision, a shortest beam and a NaN stop."""

        def __init__(self, friction: float = 0.1, dt_simulation: float = 0.006,
                     collision_tree_depth: int = 16, parallel_collision: bool = True,
                     collision_threads: int = 2, rebuild_every_resets: int = 1,
                     alarm_distance: float = 0.5, instrument_collision: bool = False,
                     min_beam_length: float = 0.0, stop_on_nan: bool = False,
                     densest_sampling: bool = False, motion_skip_mm: float = 30.0,
                     trim_sheathed: bool = False) -> None:
            super().__init__(friction=friction, dt_simulation=dt_simulation,
                             collision_tree_depth=collision_tree_depth,
                             parallel_collision=parallel_collision,
                             collision_threads=collision_threads,
                             rebuild_every_resets=rebuild_every_resets,
                             alarm_distance=alarm_distance)
            self.instrument_collision = bool(instrument_collision)
            self.min_beam_length = float(min_beam_length)
            self.stop_on_nan = bool(stop_on_nan)
            self.densest_sampling = bool(densest_sampling)
            self.motion_skip_mm = float(motion_skip_mm)
            self.substep_motion_mm: List[float] = []     # last control step, one entry per substep
            self.trim_sheathed = bool(trim_sheathed)
            if self.trim_sheathed and not (self.instrument_collision and self.stop_on_nan):
                raise ValueError("trim_sheathed needs instrument_collision (its own chain per device) and "
                                 "stop_on_nan (the per-substep loop that updates the chains' start)")
            self._sheaths: List[tuple] = []              # (device index, sheath index, device rest length)

        def _root_components(self, friction: float):
            super()._root_components(friction)
            if self.instrument_collision:
                # each device's own models carry its radius
                self.root.getObject("localmindistance").contactDistance.value = 0.0

        def _add_instruments(self, instruments, insertion_point, insertion_direction):
            super()._add_instruments(instruments, insertion_point, insertion_direction)
            combined = self._instruments_combined
            if self.min_beam_length > 0.0:
                data = combined.getObject("IRcontroller").findData("minBeamLength")
                if data is None:
                    raise RuntimeError(
                        "min_beam_length needs our BeamAdapter build with minBeamLength "
                        "(bash simulator_modifications/beamadapter/build.sh)")
                data.value = self.min_beam_length
            if self.densest_sampling:
                data = combined.getObject("IRcontroller").findData("densestSampling")
                if data is None:
                    raise RuntimeError(
                        "densest_sampling needs our BeamAdapter build with densestSampling "
                        "(bash simulator_modifications/beamadapter/build.sh)")
                data.value = True
            if self.instrument_collision:
                combined.removeChild(combined.getChild("CollisionModel"))
                for instrument in instruments:
                    radius = instrument_radius(instrument)
                    node = combined.addChild(collision_node_name(instrument.name))
                    node.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
                    node.addObject("EdgeSetTopologyModifier", name="collisEdgeModifier")
                    node.addObject("MechanicalObject", name="CollisionDOFs")
                    node.addObject("AdaptiveBeamMapping", name="collisMap",
                                   interpolation="@../Interpol_" + instrument.name,
                                   input="@../DOFs", output="@CollisionDOFs", useCurvAbs=True,
                                   nbPointsPerBeam=1.0, listening=True, printLog=False)
                    node.addObject("LineCollisionModel", contactDistance=radius,
                                   group=COLLISION_GROUP)
                    node.addObject("PointCollisionModel", contactDistance=radius,
                                   group=COLLISION_GROUP)
                    if self.trim_sheathed and combined.getChild(collision_node_name(instrument.name)).getObject(
                            "collisMap").findData("startCurvAbs") is None:
                        raise RuntimeError("trim_sheathed needs our BeamAdapter build with startCurvAbs "
                                           "(bash simulator_modifications/beamadapter/build.sh)")
            self._sheaths = self._sheath_pairs(instruments) if self.trim_sheathed else []

        @staticmethod
        def _sheath_pairs(instruments) -> List[tuple]:
            """(device, sheath, device's rest length) for every device that runs inside another: the
            sheath is the next thicker device (coaxial in BeamAdapter, so the thinner one is strictly
            inside it wherever they overlap)."""
            radii = [instrument_radius(i) for i in instruments]
            pairs = []
            for i, r in enumerate(radii):
                thicker = [(radii[j], j) for j in range(len(radii)) if radii[j] > r + 1e-9]
                if thicker:
                    pairs.append((i, min(thicker)[1], float(instruments[i].length)))
            return pairs

        def _update_sheathed_starts(self, x_tip) -> None:
            """Move each sheathed device's first collision point to its sheath's tip: in the device's own
            curvilinear abscissa (0 at its proximal end, its rest length at the tip; the inserted part is
            [length - inserted, length]) the sheath's tip sits at length - inserted_device + inserted_sheath.
            Nothing is left before it (a device fully inside its sheath has no collision point at all)."""
            combined = self._instruments_combined
            for device, sheath, length in self._sheaths:
                start = length - float(x_tip[device]) + float(x_tip[sheath])
                node = combined.getChild(collision_node_name(self._instruments[device].name))
                node.getObject("collisMap").startCurvAbs.value = max(0.0, start)

        def step(self, action: np.ndarray, duration: float):
            self.substep_motion_mm = []
            if not self.stop_on_nan:
                super().step(action, duration)
                return
            # stEVE's step (sofabeamadapter.py:85-108), with a finiteness check after each substep
            controller = self._instruments_combined.IRcontroller
            dofs = self._instruments_combined.DOFs
            n_steps = int(duration / self.dt_simulation)
            previous = np.asarray(dofs.position.value, dtype=float)[:, :3].copy()
            insertion = np.asarray(self._insertion_point, dtype=float).reshape(3)
            for _ in range(n_steps):
                inserted_lengths = self.inserted_lengths
                if len(inserted_lengths) > 1:
                    max_id = np.argmax(inserted_lengths)
                    new_length = inserted_lengths + action[:, 0] * self.dt_simulation
                    new_max_id = np.argmax(new_length)
                    if max_id != new_max_id:
                        if abs(action[max_id, 0]) > abs(action[new_max_id, 0]):
                            action[new_max_id, 0] = 0.0
                        else:
                            action[max_id, 0] = 0.0
                x_tip = controller.xtip
                tip_rot = controller.rotationInstrument
                for i in range(action.shape[0]):
                    x_tip[i] += float(action[i][0] * self.root.dt.value)
                    tip_rot[i] += float(action[i][1] * self.root.dt.value)
                controller.xtip = x_tip
                controller.rotationInstrument = tip_rot
                if self._sheaths:
                    self._update_sheathed_starts(x_tip)
                self._sofa.Simulation.animate(self.root, self.root.dt.value)
                position = np.asarray(dofs.position.value)
                if not (np.isfinite(position).all() and np.isfinite(dofs.velocity.value).all()
                        and np.abs(position[:, :3]).max() < MAX_COORDINATE_MM):
                    self.logger.warning("Non-finite or blown-up device nodes, resetting instruments")
                    self.simulation_error = True
                    self.reset_instruments()
                    return
                current = np.array(position[:, :3], dtype=float)      # a copy: SOFA updates in place
                self.substep_motion_mm.append(self._node_motion(previous, current, insertion))
                previous = current
            self._update_properties()

        MOTION_WINDOW = 3         # segments either side of a node's own index to compare against

        def _node_motion(self, previous: np.ndarray, current: np.ndarray, insertion: np.ndarray) -> float:
            """Largest move of a node in this substep, mm: each node's distance to the previous
            substep's polyline (its own and the neighbouring segments), over the nodes farther than
            ``motion_skip_mm`` from the insertion point. Off the polyline is what sweeps towards a
            wall; the leading tip's advance counts in full (it lies beyond the old polyline's end).
            A node re-laid along the device (BeamAdapter re-samples the nodes as the devices move,
            which shifts a node by a spacing along the chain) stays on the old polyline and does not
            count; the controller adds and drops nodes at the base end, so the lists are aligned at
            the tip end, and the node pile near the insertion point is left out."""
            n = min(len(previous), len(current))
            if n < 2:
                return 0.0
            before, after = previous[-n:], current[-n:]
            counted = np.linalg.norm(after - insertion, axis=1) > self.motion_skip_mm
            if not counted.any():
                return 0.0
            seg_a, seg_b = before[:-1], before[1:]
            index = np.arange(n)
            best = np.full(n, np.inf)
            for offset in range(-self.MOTION_WINDOW, self.MOTION_WINDOW + 1):
                k = np.clip(index + offset, 0, n - 2)
                a, ab = seg_a[k], seg_b[k] - seg_a[k]
                t = np.einsum("ij,ij->i", after - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
                foot = a + np.clip(t, 0.0, 1.0)[:, None] * ab
                best = np.minimum(best, np.linalg.norm(after - foot, axis=1))
            return float(best[counted].max())
