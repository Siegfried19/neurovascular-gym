"""The wall monitors (guidewire_rl/env/two_device/wall.py, labels.SubstepMotion; work log 2026-11-07).

Without SOFA (needs VTK): on an open cylinder, ``WallDistance`` counts the insertion point as inside
whichever way the mesh's normals point; the clearance is the signed distance minus the device's
radius; a point past the wall is a breach; the label reports the deepest penetration, the
truncation fires and ``WallInfo`` says so in ``info["wall_breached"]``; with per-device collision
each device is measured with its own radius. With SOFA
(``sofa``): on a far-away mesh, one control step at the wire's full speed records one substep move per
substep (33 at the config's 4 ms) of about 40 mm/s x the step; the real-mass two-device env from the insertion point for a dozen random
steps keeps every device within 0.1 mm of the wall or inside, breaches nothing, and reports a
substep motion well below the alarm distance.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from two_device_scenes import far_away_mesh, move_to, random_steps, real_mass_env, scene_in_tmp

pytestmark = pytest.mark.upstream

RADIUS = 5.0


def cylinder_obj(directory, flip: bool = False) -> str:
    """An open tube of radius RADIUS along +x, 100 mm long, centred on the origin; ``flip`` reverses
    the triangles (normals the other way)."""
    import pyvista as pv
    tube = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), radius=RADIUS, height=100.0,
                       resolution=64, capping=False).triangulate()
    faces = np.asarray(tube.faces).reshape(-1, 4)[:, 1:]
    if flip:
        faces = faces[:, ::-1]
    path = os.path.join(directory, "tube_flipped.obj" if flip else "tube.obj")
    with open(path, "w") as fh:
        for v in tube.points:
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        for f in faces:
            fh.write("f %d %d %d\n" % tuple(f + 1))
    return path


def fake_scene(mesh_path, points, per_device=None):
    """An intervention with the given mesh, insertion on the axis, and a simulation whose nodes are
    ``points`` (shared model), or ``per_device`` = {name: (points, radius)} (per-device collision)."""
    tree = SimpleNamespace(mesh_path=mesh_path, insertion=SimpleNamespace(position=np.array([-40.0, 0.0, 0.0])))
    if per_device is None:
        combined = SimpleNamespace(DOFs=SimpleNamespace(position=SimpleNamespace(value=np.asarray(points))))
        simulation = SimpleNamespace(instrument_collision=False, _instruments_combined=combined, instruments=[])
    else:
        from simulator_modifications.device_contact import collision_node_name
        children = {collision_node_name(n): SimpleNamespace(CollisionDOFs=SimpleNamespace(
            position=SimpleNamespace(value=np.asarray(p)))) for n, (p, _r) in per_device.items()}
        combined = SimpleNamespace(getChild=children.get)
        instruments = [SimpleNamespace(name=n, base_section=SimpleNamespace(diameter_outer=2 * r), tip_section=None)
                       for n, (_p, r) in per_device.items()]
        simulation = SimpleNamespace(instrument_collision=True, _instruments_combined=combined,
                                     instruments=instruments)
    return SimpleNamespace(vessel_tree=tree, simulation=simulation)


@pytest.mark.parametrize("flip", [False, True])
def test_wall_distance_clearance_and_breach_on_a_tube(tmp_path, flip):
    from guidewire_rl.env.two_device.wall import WallBreach, WallDistance, WallInfo, WallPenetration
    mesh = cylinder_obj(tmp_path, flip)
    points = np.array([[0.0, 0.0, 0.0], [10.0, 4.5, 0.0], [20.0, 0.0, -4.0]])
    scene = fake_scene(mesh, points)
    wall = WallDistance(scene)
    label, cut, told = WallPenetration(wall), WallBreach(wall), WallInfo(wall)
    wall.reset()
    state = wall.state
    assert state.points == 3 and not state.breached and state.penetration_mm == 0.0
    assert told.info == {"wall_breached": False}
    assert state.min_distance_mm == pytest.approx(0.5, abs=0.02)         # the point 0.5 mm off the wall
    assert state.clearance_mm == {"combined": pytest.approx(0.5, abs=0.02)}
    assert label.finish_step()[0] == 0.0 and not cut.truncated

    scene.simulation._instruments_combined.DOFs.position.value = np.array([[0.0, 0.0, 0.0], [10.0, 5.6, 0.0]])
    wall.step()
    assert wall.state.breached and cut.truncated and told.info == {"wall_breached": True}
    assert wall.state.penetration_mm == pytest.approx(0.6, abs=0.02)
    value, details = label.finish_step()
    assert value == pytest.approx(0.6, abs=0.02) and details["breached"] and details["points"] == 2
    assert wall.episode_max_penetration_mm == pytest.approx(0.6, abs=0.02)
    wall.reset()                                                           # a new episode starts clean
    assert wall.episode_max_penetration_mm == pytest.approx(0.6, abs=0.02)   # (the points did not move)


def test_wall_distance_uses_each_devices_own_radius(tmp_path):
    from guidewire_rl.env.two_device.wall import WallDistance
    mesh = cylinder_obj(tmp_path)
    wire = (np.array([[0.0, 0.0, 0.0], [5.0, 4.6, 0.0]]), 0.445)          # wall 0.4 mm off: 0.045 mm in
    cath = (np.array([[0.0, 0.0, 0.0], [-5.0, 0.0, 3.9]]), 1.0)           # wall 1.1 mm off: 0.1 mm clear
    wall = WallDistance(fake_scene(mesh, None, per_device={"guidewire": wire, "catheter": cath}))
    wall.reset()
    state = wall.state
    assert state.points == 4 and not state.breached
    assert state.clearance_mm["guidewire"] == pytest.approx(-0.045, abs=0.02)
    assert state.clearance_mm["catheter"] == pytest.approx(0.1, abs=0.02)
    assert state.penetration_mm == pytest.approx(0.045, abs=0.02)


# ------------------------------------------------------------------------- SOFA ----

@pytest.mark.sofa
def test_substep_motion_of_a_free_push_is_the_speed_times_the_step():
    with scene_in_tmp(far_away_mesh, stop_on_nan=True) as sim:
        move_to(sim, [100.0, 60.0], settle=300)                     # 0.6 s: the damping's lag gone
        sim.step(np.array([[40.0, 0.0], [0.0, 0.0]]), 1 / 7.5)        # the wire at 40 mm/s
        motion = np.asarray(sim.substep_motion_mm)
        n_sub = int((1 / 7.5) / sim.dt_simulation)
        assert motion.shape == (n_sub,) and np.isfinite(motion).all()
        # 40 mm/s x the substep (0.08 mm at 2 ms, 0.16 at 4 ms) commanded; the damping (rayleighStiffness
        # 0.2 s) makes the nodes ramp up to about half of it within the step (measured 0.005 -> 0.041 mm at 2 ms)
        assert 0.02 < motion.max() < 0.3
        assert motion[-1] > 2 * motion[0] and 0.01 < np.median(motion) < 0.1
        sim.step(np.zeros((2, 2)), 1 / 7.5)                           # standing still: the lag runs out
        sim.step(np.zeros((2, 2)), 1 / 7.5)
        # the relaxation left per substep scales with the substep (0.02 mm at 2 ms, 0.04 at 4 ms)
        assert np.asarray(sim.substep_motion_mm).max() < 0.03 * sim.dt_simulation / 0.002


@pytest.mark.sofa
def test_real_mass_env_reports_wall_clearance_and_substep_motion():
    with real_mass_env() as env:
        env.reset(seed=3)
        wall = env.monitors["wall_distance"]
        assert wall.state.points > 0 and not wall.state.breached
        assert wall.state.clearance_mm["guidewire"] > -0.1        # the catheter has no points yet (at 0 mm)
        for info in random_steps(env, seed=3):
            labels, details = info["labels"], info["label_details"]
            assert 0.0 <= labels["wall_penetration"] < 0.1
            assert not details["wall_penetration"]["breached"]
            assert 0.0 < labels["substep_motion"] < 1.0                   # alarm distance 2 mm
            assert details["substep_motion"]["substeps"] == int((1 / 7.5) / env.simulation.dt_simulation)
