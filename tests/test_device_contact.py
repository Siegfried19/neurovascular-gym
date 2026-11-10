"""Problems 8 and 22 (simulator_modifications/device_contact.py, configs/two_devices_real_mass.yaml).

Without SOFA: the config differs from the first round only in the masses, the physics step, the
alarm distance, the device-contact switches, the label substeps and the opening. With SOFA (``sofa``), on scenes
built from that config's devices: each device's collision chain ends at its own tip and the devices
do not collide with each other; on a flat wall the wire's centerline stops at its radius and the
catheter's at its own (the stock shared model holds both at 0.3 mm); a beam of 0.0105 mm that the
stock node layout makes is not made with ``min_beam_length`` 0.02 (and the default is the stock
layout); with ``densest_sampling`` the catheter's head keeps its own 1 mm nodes on the wire's shaft;
each device's points are re-mapped before the collision detection, every contact's force goes to the
point the detection found, segments between nodes collide too, and the points' other vectors are not
read before they are recomputed (adaptive_mapping_refresh.patch); the tip force labels read each
device's own contacts; ``stop_on_nan`` ends a control step
at the first non-finite substep and changes nothing otherwise; on the phantom, a random push
sequence that goes NaN at real mass with the stock node layout runs clean with the config; and the
config's env opens with the wire 40 mm in and the catheter at the insertion point.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config
from two_device_scenes import (CONFIG, beam_lengths, chain, far_away_mesh, flat_wall_mesh, grid_wall_mesh, move_to,
                               real_mass_env, run_in_new_process, scene, scene_in_tmp, wall_distance)

pytestmark = pytest.mark.upstream

ROUND1 = WORKSPACE_ROOT / "configs" / "two_devices_round1.yaml"
NITINOL = 6.45e-9          # t/mm^3
PLASTIC = 1.2e-9           # t/mm^3, an estimate (the lab's catheter has no measured mass)
SIMMONS2_EI = 486.0        # N mm^2, measured 5F Simmons 2 (Qiu 2024)
REAL_CATHETER_E = 660.0    # MPa: that bending stiffness on the lab hook's 2 / 1 mm tube


def test_config_changes_only_masses_catheter_stiffness_step_device_contact_substeps_and_opening():
    config, base = load_config(CONFIG), load_config(ROUND1)
    second_moment = math.pi / 64 * (2.0 ** 4 - 1.0 ** 4)          # the hook's 2 / 1 mm tube
    assert REAL_CATHETER_E * second_moment == pytest.approx(SIMMONS2_EI, rel=0.01)
    scene, base_scene = config["env"]["scene"], base["env"]["scene"]
    wire, catheter = scene["instruments"]
    base_wire, base_catheter = base_scene["instruments"]
    assert wire == dict(base_wire, mass_density=NITINOL, flex_mass_density=NITINOL)
    assert catheter == dict(base_catheter, mass_density=PLASTIC, young_modulus=REAL_CATHETER_E)
    assert scene["simulation"] == dict(base_scene["simulation"], dt_simulation=0.004, alarm_distance=1.0,
                                       instrument_collision=True, min_beam_length=0.05, stop_on_nan=True,
                                       densest_sampling=True, trim_sheathed=True)
    assert [l.get("expected_substeps") for l in config["env"]["labels"][:4]] == [33, 33, None, None]
    assert [l.get("expected_substeps") for l in base["env"]["labels"]] == [22, 22, None, None]
    assert config["env"]["start"] == {"type": "WireFirst", "wire_mm": 40.0}
    assert base["env"]["start"] == {"type": "InsertionPoint"}
    # the wall monitors (two_device/wall.py), added 2026-11-07
    assert config["env"]["monitors"] == base["env"]["monitors"] + [{"name": "wall_distance", "type": "WallDistance"}]
    assert config["env"]["truncation"] == base["env"]["truncation"] + ["WallBreach", "TipForceBlowup"]
    assert config["env"]["info"] == base["env"]["info"] + ["WallInfo", "ForceBlowupInfo"]
    assert config["env"]["labels"][4:] == [{"name": "substep_motion", "type": "SubstepMotion"},
                                           {"name": "wall_penetration", "type": "WallPenetration"}]
    for c in (config, base):
        del c["run"], c["env"]["scene"]["instruments"], c["env"]["scene"]["simulation"]
        del c["env"]["start"], c["env"]["monitors"], c["env"]["truncation"], c["env"]["info"]
        c["env"]["labels"] = c["env"]["labels"][:4]
        for label in c["env"]["labels"]:
            label.pop("expected_substeps", None)
    assert config == base


# ------------------------------------------------------------------ problem 8 ----

@pytest.mark.sofa
def test_each_device_has_its_own_chain_ending_at_its_tip_and_they_do_not_collide():
    with scene_in_tmp(far_away_mesh) as sim:
        root, combined = sim.root, sim._instruments_combined
        assert combined.getChild("CollisionModel") is None
        assert root.getObject("localmindistance").contactDistance.value == 0.0
        radii = {}
        for device in sim._instruments:
            node = combined.getChild("CollisionModel_" + device.name)
            models = [o for o in node.objects if o.getClassName() in ("LineCollisionModel",
                                                                      "PointCollisionModel")]
            assert len(models) == 2 and all(str(m.group.value).strip("[]{} ") == "1" for m in models)
            radii[device.name] = {float(m.contactDistance.value) for m in models}
        assert radii == {"guidewire": {0.445}, "catheter": {1.0}}
        root.LCP.computeConstraintForces.value = True
        move_to(sim, [120.0, 80.0], step=0.25, settle=20)
        first = int(combined.IRcontroller.indexFirstNode.value)
        nodes = np.asarray(combined.DOFs.position.value)[first:, :3]
        for device in sim._instruments:
            points = chain(sim, device.name)[:, :3]
            beams = np.asarray(combined.getObject("Interpol_" + device.name).curvAbsList.value).reshape(-1, 2)
            assert beams[-1, 1] == pytest.approx(450.0, abs=1e-4)     # its last beam ends at its tip
            assert len(points) == len(beams) + 1                    # one point per node, base to tip
            np.testing.assert_allclose(points, nodes[:len(points)], atol=1e-9)
        assert len(chain(sim, "guidewire")) == len(nodes)           # the wire leads: all nodes
        assert len(chain(sim, "catheter")) < len(nodes)
        # the wire runs inside the catheter for 80 mm: no constraint between the two devices
        assert len(root.LCP.constraintForces.value) == 0


STALE_VECTORS = ("velocity", "free_position", "free_velocity", "force", "derivX")
NEEDS_THE_BUILD = ("no segment contact: the BeamAdapter in use predates the current "
                   "adaptive_mapping_refresh.patch (bash simulator_modifications/beamadapter/build.sh)")


def tip_distances(node) -> np.ndarray:
    """Each of a device's collision points named by its distance from that device's tip along the
    device. A node is a material point of its device, so the name does not depend on the numbering,
    which shifts when a node comes out at the base."""
    s = np.asarray(node.collisMap.segmentsCurvAbs.value, dtype=float).reshape(-1)
    return s[-1] - s if len(s) else s                  # a device not yet out has no points


def contact_rows(node, multipliers):
    """[(collision points of one contact row, |multiplier|)] on one device's collision points (one
    point: a point contact, two: a segment contact)."""
    matrix = node.CollisionDOFs.findData("constraint").value.tocsr()
    rows = []
    for row in range(matrix.shape[0]):
        span = slice(matrix.indptr[row], matrix.indptr[row + 1])
        points = sorted({int(c) // 3 for c, v in zip(matrix.indices[span], matrix.data[span]) if v != 0})
        if points:
            rows.append((points, abs(float(multipliers[row])) if row < len(multipliers) else 0.0))
    return rows


class PointProbe:
    """Per substep and device: the point list this substep's collision detection uses (read at
    AnimateBegin in a node visited after the collision nodes) and the one the contact forces are
    mapped with (AnimateEnd). Counts only: SOFA swallows exceptions raised in a controller.
    ``stale_to_nan``: right after the re-map, overwrite the points' other vectors with NaN."""

    def __init__(self, sim, stale_to_nan=False):
        import Sofa
        probe = self
        self.sim = sim
        self.nodes = {d.name: sim._instruments_combined.getChild("CollisionModel_" + d.name)
                      for d in sim._instruments}
        self.substeps = self.layout_changes = self.other_list = self.bad_count = self.nonfinite = 0
        self.contacts_with_force = self.segment_contacts = self.wrong_point = 0
        self.detected, self.last = {}, {}

        class Begin(Sofa.Core.Controller):
            def onAnimateBeginEvent(self, _event):
                for name, node in probe.nodes.items():
                    probe.detected[name] = tip_distances(node)
                    if stale_to_nan:
                        for field in STALE_VECTORS:
                            data = node.CollisionDOFs.findData(field)
                            if len(data.value):
                                with data.writeableArray() as array:
                                    array[...] = np.nan

        class End(Sofa.Core.Controller):
            def onAnimateEndEvent(self, _event):
                probe.end_of_substep()

        node = sim._instruments_combined.addChild("after_the_collision_nodes")  # visited after them
        node.addObject(Begin(name="begin_probe"))
        node.init()
        sim.root.addObject(End(name="end_probe"))
        sim.root.LCP.computeConstraintForces.value = True

    def end_of_substep(self):
        same = lambda a, b: len(a) == len(b) and np.allclose(a, b, rtol=0, atol=1e-9)
        self.substeps += 1
        multipliers = np.asarray(self.sim.root.LCP.constraintForces.value, dtype=float).reshape(-1)
        self.nonfinite += not np.isfinite(multipliers).all()
        for name, node in self.nodes.items():
            detected, now = self.detected[name], tip_distances(node)
            beams = np.asarray(self.sim._instruments_combined.getObject("Interpol_" + name).lengthList.value).size
            points = beams + 1 if beams else 0                 # a device not yet out has none
            self.bad_count += len(now) != points or len(node.collisEdgeSet.findData("edges").value) != beams
            self.other_list += not same(detected, now)
            self.layout_changes += name in self.last and not same(self.last[name], now)
            self.last[name] = now
            for points, force in contact_rows(node, multipliers):
                self.contacts_with_force += force > 0
                self.segment_contacts += len(points) > 1
                self.wrong_point += any(i >= len(detected) or i >= len(now) or abs(detected[i] - now[i]) > 1e-9
                                        for i in points)


@pytest.mark.sofa
@pytest.mark.parametrize("parallel", [True, False])
def test_device_points_follow_the_beams_before_the_collision_detection(parallel):
    """The controller changes the beams at the start of a substep (nodes come out at the base, the
    catheter's tip passes the wire's nodes). Each device's collision points must be re-mapped before
    that substep's collision detection (adaptive_mapping_refresh.patch): the detection must see the
    point list the contact forces are mapped with, same points in the same order. Stock BeamAdapter
    re-maps them only during the free motion, so the detection used last substep's list (measured
    2026-11-03, random pushes: a contact's force on a neighbouring node in 5.6-6.5 % of the
    substeps, a point past the end of the list in 2.8-2.9 %)."""
    with scene_in_tmp(far_away_mesh, parallel_collision=parallel) as sim:
        probe = PointProbe(sim)
        move_to(sim, [60.0, 40.0], step=0.25)
        move_to(sim, [30.0, 70.0], step=0.25)               # the catheter passes the wire's tip
        assert probe.substeps > 300 and probe.layout_changes > 300
        assert probe.other_list == 0 and probe.bad_count == 0


@pytest.mark.sofa
def test_on_a_flat_wall_each_contact_lands_on_the_point_the_detection_found():
    """The wire, then the catheter over it, pushed along the flat wall and pulled back while touching
    it: every contact's force goes to the collision point the detection found (named by its distance
    from its device's tip), and segments between nodes touch the wall's edge too (a LineCollisionModel
    re-reads its edges only when the topology revision changes; the patch bumps it)."""
    with scene_in_tmp(grid_wall_mesh) as sim:
        probe = PointProbe(sim)
        move_to(sim, [120.0, 0.0])
        move_to(sim, [100.0, 0.0])
        move_to(sim, [100.0, 90.0])
        move_to(sim, [100.0, 70.0])
        assert probe.contacts_with_force > 2000 and probe.layout_changes > 1000
        assert probe.wrong_point == 0 and probe.other_list == 0 and probe.bad_count == 0
        assert probe.nonfinite == 0
        assert probe.segment_contacts > 100, NEEDS_THE_BUILD


STALE_VECTORS_RUN = """
import json
import numpy as np
from two_device_scenes import grid_wall_mesh, move_to, scene_in_tmp
from test_device_contact import PointProbe
with scene_in_tmp(grid_wall_mesh) as sim:
    probe = PointProbe(sim, stale_to_nan={stale_to_nan!r})
    move_to(sim, [120.0, 0.0])
    move_to(sim, [100.0, 60.0])
    out = dict(nodes=np.asarray(sim._instruments_combined.DOFs.position.value).tolist(),
               counts=[probe.contacts_with_force, probe.segment_contacts, probe.nonfinite])
print("RESULT", json.dumps(out), flush=True)
"""


@pytest.mark.sofa
def test_the_points_other_vectors_are_not_read_before_they_are_recomputed():
    """The re-map at the start of a substep computes the points' positions only; their velocity, free
    position, free velocity, force and dx keep last substep's values (in the old numbering) until the
    free motion and the contact solve recompute them. Overwriting them with NaN right after the re-map
    changes nothing, bit for bit (segment contacts happen, so the contact mappers that read them run)."""
    poisoned = run_in_new_process(STALE_VECTORS_RUN, stale_to_nan=True)
    clean = run_in_new_process(STALE_VECTORS_RUN, stale_to_nan=False)
    contacts_with_force, segment_contacts, nonfinite = poisoned["counts"]
    assert contacts_with_force > 1000 and nonfinite == 0
    assert segment_contacts > 100, NEEDS_THE_BUILD
    assert poisoned["nodes"] == clean["nodes"]


@pytest.mark.sofa
def test_on_a_flat_wall_each_device_stops_at_its_own_radius():
    """The wire alone pushed along the wall, then the catheter over it (the wire 80 mm inside);
    nearest node (one collision point per node). The stock shared model holds both at 0.3 mm."""
    def nearest(instrument_collision):
        with scene_in_tmp(flat_wall_mesh, instrument_collision=instrument_collision, stop_on_nan=False) as sim:
            dofs = sim._instruments_combined.DOFs
            move_to(sim, [120.0, 0.0], settle=300)
            wire = float(np.min(wall_distance(dofs.position.value)))
            move_to(sim, [80.0, 100.0], settle=300)
            catheter = float(np.min(wall_distance(chain(sim, "catheter") if instrument_collision
                                                  else dofs.position.value)))
            return wire, catheter

    wire, catheter = nearest(True)
    assert wire == pytest.approx(0.445, abs=0.005)
    assert catheter == pytest.approx(1.0, abs=0.005)
    wire, catheter = nearest(False)
    assert wire == pytest.approx(0.3, abs=0.005) and catheter == pytest.approx(0.3, abs=0.005)


@pytest.mark.sofa
def test_tip_force_labels_read_each_devices_own_contacts():
    from guidewire_rl.env.two_device.labels import CatheterTipForce, InstrumentTipForce
    import Sofa
    with scene_in_tmp(flat_wall_mesh) as sim:
        env = SimpleNamespace(simulation=sim)
        # the whole device: which device a contact is booked to (the 3 mm tip window is the
        # shared window_force on a chain that ends at the device's tip, tested above)
        wire_label = InstrumentTipForce(instrument=0, distal_length_mm=1000.0)
        cath_label = CatheterTipForce(distal_length_mm=1000.0)
        wire_label.attach(env)
        cath_label.attach(env)
        ctrl, dt = sim._instruments_combined.IRcontroller, sim.root.dt.value

        def control_step(push):
            for label in (wire_label, cath_label):
                label.begin_step()
            for _ in range(66):
                ctrl.xtip.value = (np.asarray(ctrl.xtip.value) + np.asarray(push) * dt).tolist()
                Sofa.Simulation.animate(sim.root, dt)
            return wire_label.finish_step()[0], cath_label.finish_step()[0]

        move_to(sim, [100.0, 0.0], settle=100)
        wire, catheter = control_step([10.0, 0.0])          # the wire alone along the wall
        assert wire > 0.0 and catheter == 0.0
        move_to(sim, [80.0, 100.0], settle=100)             # the catheter over the wire's 80 mm
        wire, catheter = control_step([0.0, 10.0])
        assert catheter > 0.0 and wire == 0.0


# ----------------------------------------------------------------- problem 22 ----

@pytest.mark.sofa
def test_min_beam_length_drops_the_node_that_makes_a_0p0105_mm_beam():
    """Wire tip at 100 mm: its soft part has a node every 0.5 mm, one at 90 mm. The catheter's tip
    at 90.0105 mm makes a 0.0105 mm beam with the stock layout (merged only below 0.01 mm)."""
    def layout(min_beam_length):
        with scene_in_tmp(far_away_mesh, min_beam_length=min_beam_length) as sim:
            move_to(sim, [100.0, 89.75], step=0.25)
            move_to(sim, [100.0, 90.0105], step=0.5, settle=5)
            catheter = sim._instruments_combined.getObject("Interpol_catheter")
            end = float(np.asarray(catheter.curvAbsList.value).reshape(-1)[-1])
            return float(beam_lengths(sim).min()), end

    shortest, end = layout(0.0)
    assert shortest == pytest.approx(0.0105, abs=1e-4)   # the wire is 449.99998 mm: nodes 2e-5 off
    assert end == pytest.approx(450.0, abs=1e-9)                      # the catheter reaches its tip
    shortest, end = layout(0.02)
    assert shortest >= 0.02
    assert end == pytest.approx(450.0 - 0.0105, abs=1e-4)             # ends at the wire's node


@pytest.mark.sofa
def test_densest_sampling_keeps_the_hook_at_its_own_1_mm_spacing_on_the_wire_shaft():
    """Wire at 140 mm, catheter at 100 mm: the catheter's 46 mm head lies on the wire's shaft (its soft
    30 mm are out ahead). BeamAdapter's own layout takes the wire's 10 mm spacing there; with
    densest_sampling the head keeps its 1 mm nodes, and so do the catheter's collision points."""
    def head_beams(densest):
        with scene_in_tmp(far_away_mesh, densest_sampling=densest) as sim:
            move_to(sim, [140.0, 100.0], step=0.25)
            catheter = sim._instruments[1]
            interp = sim._instruments_combined.getObject("Interpol_catheter")
            ends = np.asarray(interp.curvAbsList.value, dtype=float).reshape(-1, 2)
            lengths = np.asarray(interp.lengthList.value, dtype=float).reshape(-1)
            head = ends[:, 0] >= catheter.length - catheter.tip_section.length - 1e-6
            points = chain(sim, "catheter")[:, :3]
            return lengths[head], points

    stock, stock_points = head_beams(False)
    assert stock.max() > 5.0 and len(stock) <= 6                  # 3-6 beams for 46 mm
    dense, dense_points = head_beams(True)
    assert dense.max() <= 1.0 + 1e-6 and len(dense) >= 46
    assert len(dense_points) > len(stock_points) + 35


@pytest.mark.sofa
def test_stop_on_nan_stops_at_the_first_bad_substep_and_changes_nothing_else():
    import Sofa

    class Counter(Sofa.Core.Controller):
        def __init__(self):
            super().__init__(name="substep_counter")
            self.n = 0

        def onAnimateEndEvent(self, _event):
            self.n += 1

    actions = [np.array([[20.0, 0.5], [15.0, -0.3]]), np.array([[-5.0, 0.0], [10.0, 0.2]])] * 3

    def run(stop_on_nan):
        with scene_in_tmp(far_away_mesh, stop_on_nan=stop_on_nan) as sim:
            track = []
            for action in actions:
                sim.step(action.copy(), 1 / 7.5)
                track.append(np.array(sim._instruments_combined.DOFs.position.value))
            return track

    # two scenes in one process differ in the last bits (work log 2026-11-02 14:42), so a tolerance
    for a, b in zip(run(True), run(False)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    with scene_in_tmp(far_away_mesh) as sim:
        counter = Counter()
        sim.root.addObject(counter)
        sim.step(actions[0].copy(), 1 / 7.5)
        assert counter.n == 33 and not sim.simulation_error                # 1/7.5 s of 4 ms substeps
        with sim._instruments_combined.DOFs.velocity.writeableArray() as velocity:
            velocity[-3, 1] = np.nan
        sim.step(actions[0].copy(), 1 / 7.5)
        assert counter.n == 34 and sim.simulation_error          # stopped after one substep
        assert np.isfinite(sim._instruments_combined.DOFs.position.value).all()   # reset


RANDOM_PUSH_RUN = """
import json
from two_device_scenes import CONFIG, random_pushes
from guidewire_rl.config import load_config
from guidewire_rl.env import build_env
config = load_config(CONFIG)["env"]
config["scene"]["simulation"]["min_beam_length"] = {min_beam!r}
config["scene"]["simulation"]["dt_simulation"] = 0.002   # measured at 2 ms (the config moved to 4 ms on 2026-11-09)
for label in config["labels"]:
    if "expected_substeps" in label:
        label["expected_substeps"] = 66
config["start"] = {{"type": "InsertionPoint"}}          # measured before the config had the opening
env = build_env(config)
print("RESULT", json.dumps(random_pushes(env, {steps!r}, seed={seed!r})), flush=True)
"""


@pytest.mark.sofa
def test_real_mass_random_pushes_go_nan_with_the_stock_layout_and_not_with_the_config():
    """Random seed 1 goes NaN within 150 control steps at real mass with the stock node layout (step
    138, measured 2026-11-03 with this config before it had the opening, which the run turns off);
    with the config's min_beam_length the same 150 steps run without NaN. Since the wall monitors
    (2026-11-07) the blow-up ends the episode on that same step as a wall breach (a device thrown out
    of the mesh) before the NaN is seen, so either end counts. Fresh processes, because one that
    simulated other scenes first does not repeat the trajectory bit for bit, so where it fails would
    depend on the test order."""
    kind, step = run_in_new_process(RANDOM_PUSH_RUN, min_beam=0.0, seed=1, steps=150)
    assert kind in ("nan", "ended") and step == 138, (kind, step)
    kind, step = run_in_new_process(RANDOM_PUSH_RUN, seed=1, steps=150,
                                    min_beam=load_config(CONFIG)["env"]["scene"]["simulation"]["min_beam_length"])
    assert kind is None and step == 150, (kind, step)


@pytest.mark.sofa
def test_real_mass_env_opens_with_the_wire_40_mm_in():
    """The config's opening (WireFirst) leaves the wire 40 mm in and the catheter at the insertion
    point, with no action shown as taken; the policy's first steps then start from there."""
    with real_mass_env() as env:
        _, info = env.reset(seed=0)
        sim = env.intervention.simulation
        np.testing.assert_allclose(sim.inserted_lengths, [40.0, 0.0], atol=1e-6)
        assert info["phase"] == 0 and not info["simulation_error"]
        assert np.all(np.asarray(env.intervention.last_action) == 0.0)
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 1.0, 0.0], np.float32))
        np.testing.assert_allclose(sim.inserted_lengths, [40.0, 25.0 * 0.132], atol=1e-6)  # 33 x 4 ms
        assert not (terminated or truncated or info["simulation_error"])
