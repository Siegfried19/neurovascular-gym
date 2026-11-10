"""simulator_modifications/ on upstream stEVE + SOFA 25.12 (docs/simulator_changes.md).

Run in the conda env ``guidewire_rl``; skipped with Wilson's eve.
"""

import copy
import inspect
import math
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

from guidewire_rl.env import build_env, registry

pytestmark = pytest.mark.upstream

ACTIONS = np.asarray([[0.9, 0.1], [0.8, -0.3], [1.0, 0.5], [0.6, -0.9],
                      [0.9, 0.2], [1.0, -0.1], [0.7, 0.8], [0.9, -0.5]] * 2, dtype=np.float32)


def far_away_mesh(directory) -> str:
    """A vessel 'tree' of one triangle far from the wire: no contact at all."""
    path = os.path.join(directory, "far.obj")
    with open(path, "w") as fh:
        fh.write("v -5000 -5000 -5000\nv -5001 -5000 -5000\nv -5000 -5001 -5000\nf 1 2 3\n")
    return path


def test_scene_names_resolve_to_the_fixed_classes():
    import eve
    from simulator_modifications import device_contact, hollow_catheter, observations, seeds, speedups
    assert registry.resolve("observation", "Tracking2D") is observations.Tracking2D
    assert (registry.resolve("observation_wrapper", "NormalizeTracking2DEpisode")
            is observations.NormalizeTracking2DEpisode)
    assert registry.resolve("simulation", "SofaBeamAdapter") is device_contact.SofaBeamAdapter
    assert issubclass(device_contact.SofaBeamAdapter, speedups.SofaBeamAdapter)
    from simulator_modifications import pathfinder
    assert registry.resolve("pathfinder", "BruteForceBFS") is pathfinder.BruteForceBFS
    assert issubclass(speedups.SofaBeamAdapter, hollow_catheter.SofaBeamAdapter)
    assert registry.resolve("target", "CenterlineRandom") is seeds.CenterlineRandom
    assert registry.resolve("target", "Manual") is seeds.Manual
    assert registry.resolve("vessel_tree", "AorticArchRandom") is seeds.AorticArchRandom
    assert registry.resolve("instrument", "Angled") is eve.intervention.instrument.Angled
    assert registry.resolve("device", "Angled") is eve.intervention.instrument.Angled


# ---------------------------------------------------------------- problem 7 ----

def test_the_class_a_config_gets_does_not_depend_on_import_order():
    """Only device_contact registers ``SofaBeamAdapter``; importing speedups (its base) first must not
    leave the name pointing at the base, which does not take the device-contact switches."""
    code = ("import simulator_modifications.speedups\n"
            "from guidewire_rl.env import registry\n"
            "print('CLASS', registry.resolve('simulation', 'SofaBeamAdapter').__module__)\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "CLASS simulator_modifications.device_contact" in run.stdout


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python < 3.12 only warns")
def test_numpy_seed_breaks_stock_stEVE_on_python_312():
    """The bug the fix is for: stEVE's own MonoPlaneStatic hands numpy seeds down."""
    import random
    with pytest.raises(TypeError):
        random.Random(np.int64(7))


def test_aortic_arch_random_accepts_numpy_seeds():
    from simulator_modifications.seeds import AorticArchRandom
    first, second = AorticArchRandom(), AorticArchRandom()
    first.reset(0, np.int64(3))
    second.reset(0, np.int64(3))
    np.testing.assert_array_equal(first.centerline_coordinates, second.centerline_coordinates)


@pytest.mark.sofa
def test_seeded_reset_is_reproducible(upstream_config):
    """Two fresh envs, same seed: same target and same first observation.
    (Within one env the first reset also builds the scene, which draws the
    initial wire rotation once more, so only resets at the same position in
    the sequence match; that is stEVE's behaviour, also in the old env.)"""
    observations, targets = [], []
    for _ in range(2):
        env = build_env(upstream_config["env"])
        try:
            observations.append(env.reset(seed=7)[0])
            targets.append(np.array(env.intervention.target.coordinates3d))
        finally:
            env.close()
    np.testing.assert_array_equal(*observations)
    np.testing.assert_array_equal(*targets)


@pytest.mark.sofa
def test_manual_target_from_yaml_lists_runs_with_the_target_observation(upstream_config):
    """Problems 7 and 19: fixed targets written as YAML lists, observed through Target2D
    (g13.yaml's observation), reset with a seed (MonoPlaneStatic hands it a numpy int)."""
    from eve.util.coordtransform import vessel_cs_to_tracking3d
    points = [[0.0, 0.0, 50.0], [10.0, 0.0, 80.0]]
    config = copy.deepcopy(upstream_config)
    config["env"]["scene"]["target"] = {"type": "Manual", "threshold": 5.0, "targets_vessel_cs": points}
    env = build_env(config["env"])
    try:
        target, fluoroscopy = env.intervention.target, env.intervention.fluoroscopy
        assert type(target).__module__ == "simulator_modifications.seeds"
        allowed = [vessel_cs_to_tracking3d(np.asarray(p), fluoroscopy.image_rot_zx,
                                           fluoroscopy.image_center, fluoroscopy.field_of_view)
                   for p in points]
        chosen = []
        for _ in range(2):
            env.reset(seed=3)
            chosen.append(np.array(target.coordinates3d))
            np.testing.assert_array_equal(target.coordinates2d, np.delete(target.coordinates3d, 1, -1))
            for action in ACTIONS[:4]:
                env.step(action)
        np.testing.assert_array_equal(*chosen)
        assert any(np.array_equal(chosen[0], p) for p in allowed)
    finally:
        env.close()


# ---------------------------------------------------------------- problem 2 ----

def cantilever_deflection(simulation_class, od, idd, young, length, steps=1500):
    """Tip deflection (mm) of a uniform tube pushed ``length`` mm into empty space,
    clamped by BeamAdapter's base, under a lateral tip force; and the force (N).
    Half the difference between +F and -F cancels any residual rest offset."""
    import Sofa
    import eve

    class Probe(simulation_class):
        def _load_plugins(self):
            super()._load_plugins()
            self.root.addObject("RequiredPlugin", pluginName="Sofa.Component.MechanicalLoad")

        def _add_instruments(self, instruments, insertion_point, insertion_direction):
            super()._add_instruments(instruments, insertion_point, insertion_direction)
            inst = instruments[0]
            self.tip = inst.tip_section.collis_edges + inst.base_section.collis_edges
            self.load = self._instruments_combined.addObject(
                "ConstantForceField", name="tipLoad", template="Rigid3d", indices=[self.tip],
                forces=[[0.0] * 6])

    tube = eve.intervention.instrument.Straight(
        name="tube", length=length + 150.0, diameter_outer=od, diameter_inner=idd,
        young_modulus=young, mass_density=2.1e-11, collis_edges_per_mm=1.0,
        flex_length=40.0, flex_diameter_outer=od, flex_diameter_inner=idd,
        flex_young_modulus=young, flex_mass_density=2.1e-11, flex_collis_edges_per_mm=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        sim = Probe(friction=0.0)
        sim.reset(insertion_point=np.zeros(3), insertion_direction=np.array([1.0, 0.0, 0.0]),
                  mesh_path=far_away_mesh(tmp), instruments=[tube],
                  coords_high=np.full(3, 10.0), coords_low=np.full(3, -10.0))
        controller = sim._instruments_combined.IRcontroller
        dofs = sim._instruments_combined.DOFs
        controller.rotationInstrument.value = [0.0]
        dt = sim.root.dt.value
        x = float(controller.xtip.value[0])
        while x < length:                                # 0.25 mm steps, never exactly 0.01
            x = min(x + 0.25, length)
            controller.xtip.value = [x]
            Sofa.Simulation.animate(sim.root, dt)
        for _ in range(steps):
            Sofa.Simulation.animate(sim.root, dt)
        free = np.array(dofs.position.value)[int(controller.indexFirstNode.value):, :3]
        free_length = float(np.sum(np.linalg.norm(np.diff(free, axis=0), axis=1)))
        second_moment = math.pi / 64 * (od ** 4 - idd ** 4)
        force = 0.01 * free_length * 3 * young * second_moment / free_length ** 3
        tips = {}
        for sign in (1.0, -1.0):
            sim.load.forces.value = [[0.0, sign * force, 0.0, 0.0, 0.0, 0.0]]
            for _ in range(steps):
                Sofa.Simulation.animate(sim.root, dt)
            tips[sign] = float(dofs.position.value[sim.tip][1])
        sim.close()
    return 0.5 * (tips[1.0] - tips[-1.0]), force, free_length


def cantilever_theory(force, length, young, od, idd):
    return force * length ** 3 / (3 * young * math.pi / 64 * (od ** 4 - idd ** 4))


@pytest.mark.sofa
def test_catheter_bends_like_a_hollow_tube():
    import eve
    from simulator_modifications.hollow_catheter import SofaBeamAdapter
    od, idd, young, length = 2.0, 1.73, 2180.0, 60.0          # 6F aspiration catheter
    fixed, force, free = cantilever_deflection(SofaBeamAdapter, od, idd, young, length)
    hollow = cantilever_theory(force, free, young, od, idd)
    assert fixed == pytest.approx(hollow, rel=0.01)
    stock, force, free = cantilever_deflection(
        eve.intervention.simulation.SofaBeamAdapter, od, idd, young, length)
    solid = cantilever_theory(force, free, young, od, 0.0)
    assert stock == pytest.approx(solid, rel=0.01)              # the bug: stEVE bends a solid rod


@pytest.mark.sofa
def test_guidewire_is_unchanged_by_the_hollow_fix():
    from simulator_modifications.hollow_catheter import SofaBeamAdapter
    od, young, length = 0.89, 80e3, 60.0
    deflection, force, free = cantilever_deflection(SofaBeamAdapter, od, 0.0, young, length)
    assert deflection == pytest.approx(cantilever_theory(force, free, young, od, 0.0), rel=0.01)


# ---------------------------------------------------------------- problem 1 ----

CRASH_PROBE = textwrap.dedent("""
    import numpy as np, Sofa, eve, tempfile, os
    mesh = os.path.join(tempfile.mkdtemp(), "far.obj")
    open(mesh, "w").write("v -5000 -5000 -5000\\nv -5001 -5000 -5000\\nv -5000 -5001 -5000\\nf 1 2 3\\n")
    sim = eve.intervention.simulation.SofaBeamAdapter(friction=0.0)
    sim.reset(insertion_point=np.zeros(3), insertion_direction=np.array([1.0, 0.0, 0.0]),
              mesh_path=mesh, instruments=[eve.intervention.instrument.Straight()],
              coords_high=np.full(3, 10.0), coords_low=np.full(3, -10.0))
    print("BeamAdapter", sorted({l.split()[-1] for l in open("/proc/self/maps") if "libBeamAdapter" in l}))
    ctrl, dt = sim._instruments_combined.IRcontroller, sim.root.dt.value
    x = float(ctrl.xtip.value[0])
    while x < 200.0:
        x = min(x + 0.25, 200.0)
        ctrl.xtip.value = [x]
        Sofa.Simulation.animate(sim.root, dt)
    ctrl.xtip.value = [200.01]          # exactly the controller's 0.01 mm threshold
    for _ in range(5):
        Sofa.Simulation.animate(sim.root, dt)
    print("SURVIVED")
""")


@pytest.mark.sofa
def test_exact_threshold_push_does_not_crash():
    """200.0 -> 200.01 mm read past the end of BeamAdapter's node list and
    segfaulted with the released plugin; our build has the upstream fix."""
    run = subprocess.run([sys.executable, "-c", CRASH_PROBE], capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, f"exit {run.returncode}: {run.stderr[-2000:]}"
    assert "SURVIVED" in run.stdout
    loaded = [line for line in run.stdout.splitlines() if line.startswith("BeamAdapter")]
    assert loaded and "BeamAdapter-v25.12-dev" in loaded[0], (
        f"not our BeamAdapter build: {loaded}; is SOFA_PLUGIN_PATH set (simulator_modifications/beamadapter/build.sh)?")



# --------------------------------------------------------------- problem 17 ----

def steve_without_straight_tip_check(monkeypatch):
    """stEVE refuses a straight soft tip after a shaft because of this very bug; drop that check
    (only its `raise`) for one test."""
    from eve.intervention.simulation import sofabeamadapter
    source = textwrap.dedent(inspect.getsource(sofabeamadapter.SofaBeamAdapter._add_instruments))
    assert source.count("raise ValueError(") == 1, "stEVE's check changed"
    namespace = {}
    exec(compile(source.replace("raise ValueError(", "print(", 1), sofabeamadapter.__file__, "exec"),
         vars(sofabeamadapter), namespace)
    monkeypatch.setattr(sofabeamadapter.SofaBeamAdapter, "_add_instruments", namespace["_add_instruments"])
    return sofabeamadapter.SofaBeamAdapter


@pytest.mark.sofa
def test_straight_soft_tip_after_a_shaft_keeps_its_rest_shape(monkeypatch, tmp_path):
    """Shaft 420 mm + soft straight tip 30 mm, pushed ~58 mm through free space. With the tip as a
    second RodStraightSection, released BeamAdapter v25.12 puts its rest shape 420 mm too far and the
    tip shoots ~415 mm ahead (figure: visual/rod_straight_section/ in commit f7b7440); our build has the
    upstream fix, so the tip lands where the same tip written as a mesh section (stEVE's `Straight`) does."""
    from eve.intervention.instrument.instrument import (Instrument, MeshSection, StraightMeshElement,
                                                        StraightSection)
    SofaBeamAdapter = steve_without_straight_tip_check(monkeypatch)
    soft = dict(diameter_outer=0.7, diameter_inner=0.0, poisson_ratio=0.49, young_modulus=17e3,
                mass_density=2.1e-5, visu_edges_per_mm=0.5, collis_edges_per_mm=2)
    mesh = far_away_mesh(str(tmp_path))

    def tip_after_push(tip) -> float:
        shaft = StraightSection(length=420.0, diameter_outer=0.89, diameter_inner=0.0, poisson_ratio=0.49,
                                young_modulus=80e3, mass_density=2.1e-5, visu_edges_per_mm=0.5,
                                collis_edges_per_mm=0.1)
        wire = Instrument(name="guidewire", velocity_limit=(50, 3.14), tip_section=tip, base_section=shaft)
        sim = SofaBeamAdapter(friction=0.0)
        sim.reset(insertion_point=np.zeros(3), insertion_direction=np.array([1.0, 0.0, 0.0]),
                  mesh_path=mesh, instruments=[wire],
                  coords_high=np.full(3, 1000.0), coords_low=np.full(3, -1000.0))
        for _ in range(20):                    # 30 mm/s for 2 s: the joint is ~28 mm in
            sim.step(np.array([[30.0, 0.0]]), duration=0.1)
        tip_x = float(sim.dof_positions[0][0])
        sim.close()
        return tip_x

    straight = tip_after_push(StraightSection(length=30.0, **soft))
    meshed = tip_after_push(MeshSection(mesh_elements=[StraightMeshElement(30.0)], **soft))
    loaded = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libBeamAdapter" in line})
    assert abs(straight - meshed) < 1.0, (
        f"tip {straight:.1f} mm (straight section) vs {meshed:.1f} mm (mesh section) with {loaded}; "
        "is SOFA_PLUGIN_PATH set (simulator_modifications/beamadapter/build.sh)?")


# ------------------------------------------------------ force label on LCP ----

@pytest.mark.sofa
def test_force_label_does_not_change_the_dynamics(upstream_config):
    """Turning on LCP's constraint-force output only copies numbers out."""
    def rollout(config):
        env = build_env(config["env"])
        try:
            steps = [env.reset(seed=1000)[0]]
            forces = []
            for action in ACTIONS:
                observation, reward, _, _, info = env.step(action)
                steps.append((observation, reward))
                forces.append(info["labels"].get("distal_force"))
            return steps, forces
        finally:
            env.close()

    with_label, forces = rollout(upstream_config)
    plain = copy.deepcopy(upstream_config)
    plain["env"]["labels"] = []
    without_label, _ = rollout(plain)
    np.testing.assert_array_equal(with_label[0], without_label[0])
    for (o1, r1), (o2, r2) in zip(with_label[1:], without_label[1:]):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
    assert all(np.isfinite(f) and f >= 0 for f in forces)
    assert any(f > 0 for f in forces)                      # these actions do touch the wall


# ------------------------------------------------------- problems 14 and 15 ----

def bare_tracking2d(cls, n_points, resolution):
    term = cls.__new__(cls)
    term.n_points, term.resolution = n_points, resolution
    return term


def test_tracking2d_spreads_points_along_a_long_segment():
    import eve
    from simulator_modifications.observations import Tracking2D
    wire = np.asarray([[0, 0], [5, 0], [5, 5]], dtype=np.float32)     # first segment 5 mm
    stock = bare_tracking2d(eve.observation.Tracking2D, 3, 2.0)._evenly_distributed_tracking(wire)
    fixed = bare_tracking2d(Tracking2D, 3, 2.0)._evenly_distributed_tracking(wire)
    np.testing.assert_array_equal(stock, [[0, 0], [2, 0], [2, 0]])     # the bug
    np.testing.assert_array_equal(fixed, [[0, 0], [2, 0], [4, 0]])


def test_tracking2d_fix_is_bit_identical_when_nodes_are_closer_than_the_spacing():
    """The usual case (Angled's tip nodes are ~0.5 mm apart): nothing changes."""
    import eve
    from simulator_modifications.observations import Tracking2D
    rng = np.random.default_rng(0)
    for _ in range(50):
        steps = rng.normal(size=(40, 2))
        steps *= rng.uniform(0.1, 0.95, (40, 1)) / np.linalg.norm(steps, axis=1, keepdims=True)
        wire = np.cumsum(steps, axis=0).astype(np.float32)
        for n_points, resolution in ((5, 1.0), (3, 2.0)):
            np.testing.assert_array_equal(
                bare_tracking2d(Tracking2D, n_points, resolution)._evenly_distributed_tracking(wire),
                bare_tracking2d(eve.observation.Tracking2D, n_points, resolution)
                ._evenly_distributed_tracking(wire))


def phantom_scene():
    import eve
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    tree = BiroPhantomArch()
    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(simulation=None, vessel_tree=tree,
                                                            image_rot_zx=[20, 5])
    return tree, fluoroscopy


def lumen_outline(tree, fluoroscopy):
    """Image-plane points on the wall of every lumen tube (centerline ± radius)."""
    from eve.util.coordtransform import tracking3d_to_2d, vessel_cs_to_tracking3d
    points = np.concatenate([b.coordinates for b in tree.branches])
    radii = np.concatenate([b.radii for b in tree.branches])[:, None]
    centre = tracking3d_to_2d(vessel_cs_to_tracking3d(points, fluoroscopy.image_rot_zx,
                                                      fluoroscopy.image_center, None))
    return np.concatenate([centre + radii * np.asarray(d, np.float32)
                           for d in ([1, 0], [-1, 0], [0, 1], [0, -1])])


def normalized(points, space):
    return 2 * (points - space.low) / (space.high - space.low) - 1


def test_normalization_range_covers_the_whole_lumen():
    from simulator_modifications.observations import lumen_bounds_2d
    tree, fluoroscopy = phantom_scene()
    wall = lumen_outline(tree, fluoroscopy)
    stock = normalized(wall, fluoroscopy.tracking2d_space_episode)
    fixed = normalized(wall, lumen_bounds_2d(tree, fluoroscopy))
    assert np.abs(stock).max() > 1.05           # the bug: a wire at the wall leaves [-1, 1]
    assert np.abs(fixed).max() <= 1 + 1e-5
    for axis in range(2):                        # and the range is no wider than the lumen
        assert fixed[:, axis].min() == pytest.approx(-1, abs=1e-5)
        assert fixed[:, axis].max() == pytest.approx(1, abs=1e-5)


# ---------------------------------------------------------- problems 4 and 5 ----

def with_simulation(config, **simulation):
    config = copy.deepcopy(config)
    config["env"]["scene"]["simulation"].update(simulation)
    return config


def tip_track(config, actions=ACTIONS, seed=1000):
    """Wire tip in vessel coordinates after each action, and the scene's root components."""
    env = build_env(config["env"])
    try:
        env.reset(seed=seed)
        root = env.intervention.simulation.root
        components = {o.getClassName(): o for o in root.objects}
        setup = {
            "depth": int(components["CollisionPipeline"].depth.value),
            "parallel": bool(components["FreeMotionAnimationLoop"]
                             .parallelCollisionDetectionAndFreeMotion.value),
            "threads": (int(components["ParallelBVHNarrowPhase"].nbThreads.value)
                        if "ParallelBVHNarrowPhase" in components else 0),
            "alarm": float(components["LocalMinDistance"].alarmDistance.value),
        }
        tips = []
        for action in actions:
            env.step(action)
            tips.append(np.array(env.intervention.simulation.dof_positions[0], dtype=np.float64))
        return np.asarray(tips), setup
    finally:
        env.close()


@pytest.mark.sofa
def test_speedups_are_on_by_default_and_parallel_changes_nothing(upstream_config):
    fast, setup = tip_track(upstream_config)
    assert setup == {"depth": 16, "parallel": True, "threads": 2, "alarm": 0.5}   # stEVE's alarm distance
    serial, setup = tip_track(with_simulation(upstream_config, parallel_collision=False))
    assert setup == {"depth": 16, "parallel": False, "threads": 0, "alarm": 0.5}
    np.testing.assert_array_equal(fast, serial)          # lab: bit for bit


# --------------------------------------------------------------- problem 20 ----

@pytest.mark.sofa
def test_alarm_distance_is_a_parameter_that_reaches_the_collision_detection(upstream_config):
    """2 mm: every wire point is paired with the wall triangles within 2 mm instead of 0.5 mm. The
    pairs beyond 0.3 mm (the contact distance) only act when the point would reach the plane, so a
    wire that never comes that close in one substep moves the same; here the heavy stock wire does."""
    stock, _ = tip_track(with_simulation(upstream_config, parallel_collision=False))
    wide, setup = tip_track(with_simulation(upstream_config, parallel_collision=False, alarm_distance=2.0))
    assert setup["alarm"] == 2.0
    assert np.max(np.linalg.norm(wide - stock, axis=1)) < 1.0


@pytest.mark.sofa
def test_deeper_collision_tree_barely_moves_the_wire(upstream_config):
    deep, _ = tip_track(with_simulation(upstream_config, parallel_collision=False))
    stock, setup = tip_track(with_simulation(upstream_config, parallel_collision=False,
                                             collision_tree_depth=6))
    assert setup["depth"] == 6
    assert np.max(np.linalg.norm(deep - stock, axis=1)) < 1.0    # lab: max 0.62 mm


# --------------------------------------------------------------- problem 16 ----

@pytest.mark.sofa
def test_scene_is_rebuilt_every_reset_and_the_force_label_follows(upstream_config):
    env = build_env(upstream_config["env"])
    try:
        sim = env.intervention.simulation
        for episode in range(3):
            env.reset(seed=episode)
            assert sim.builds == episode + 1
            monitor = env.labels[0].monitor
            assert sim.root.getObject(monitor.name.value) is monitor      # attached to this scene
            forces = []
            for action in ACTIONS:
                _, _, terminated, truncated, info = env.step(action)
                forces.append(info["labels"]["distal_force"])
                if terminated or truncated:
                    break
            assert max(forces) > 0.0                                       # still reads contacts
        rebuilt = with_simulation(upstream_config, rebuild_every_resets=0)
    finally:
        env.close()
    env = build_env(rebuilt["env"])
    try:
        for episode in range(3):
            env.reset(seed=episode)
        assert env.intervention.simulation.builds == 1                    # stEVE's own behaviour
    finally:
        env.close()


def test_pathfinder_without_deepcopy_finds_the_same_paths():
    """Problem 18: same path lengths and points as stEVE's BruteForceBFS, on the phantom with
    every vessel (75 branches) and tips spread over the whole tree."""
    import types

    import eve
    from eve.util.coordtransform import vessel_cs_to_tracking3d

    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    from simulator_modifications.pathfinder import BruteForceBFS

    tree = BiroPhantomArch(centerline="all_vessels")
    rng = np.random.default_rng(0)
    points = np.concatenate([b.coordinates for b in tree.branches])
    tips = points[rng.choice(len(points), 40, replace=False)]
    targets = np.concatenate([tree[name].coordinates for name in ("rcca", "lsa", "bct")])
    rot, centre = [20, 5], [0.0, 0.0, 0.0]
    to3d = lambda p: vessel_cs_to_tracking3d(np.asarray(p, float), rot, centre, None)
    for i, tip in enumerate(tips):
        fluoroscopy = types.SimpleNamespace(tracking3d=to3d(tip[None]), image_rot_zx=rot,
                                            image_center=centre, field_of_view=None)
        target = types.SimpleNamespace(coordinates3d=to3d(targets[i % len(targets)]))
        intervention = types.SimpleNamespace(fluoroscopy=fluoroscopy, target=target, vessel_tree=tree)
        ours, theirs = BruteForceBFS(intervention), eve.pathfinder.BruteForceBFS(intervention)
        ours.reset(), theirs.reset()
        assert ours.path_length == theirs.path_length
        np.testing.assert_array_equal(ours.path_points3d, theirs.path_points3d)


# --------------------------------------------------------------- problem 21 ----
# TrackingOnly's tracking space turned by two corners only (simulator_modifications/fluoroscopy.py).

def _c_arm_angles():
    """The C-arm angles training uses on the phantom: G13's, the two-device run's, and every row of
    the per-target angle tables (guidewire_rl/env/c_arm/data/)."""
    import json
    from guidewire_rl.env.c_arm import choose_angle
    angles = {"G13": [20.0, 5.0], "two devices, round 1": [61.0, -21.0]}
    for which in ("rebuilt_cut", "all_vessels"):
        table = json.loads(choose_angle.table_path(which).read_text())
        angles.update({f"{which} {key}": row["image_rot_zx"] for key, row in table["angles"].items()})
    return angles


def _corners(space):
    import itertools
    low, high = np.asarray(space.low, np.float64), np.asarray(space.high, np.float64)
    return np.array([[(low, high)[p][a] for a, p in enumerate(ps)] for ps in itertools.product((0, 1), repeat=3)])


@pytest.mark.parametrize("version", ["rebuilt_cut", "lab"])
def test_rotated_tracking_space_is_the_box_around_every_rotated_corner(version):
    """stEVE's box is not a box at some of these angles (gymnasium refuses it); ours is, at every one,
    and it is the smallest box holding the vessel box's eight corners turned into the image frame."""
    import eve
    from eve.util.coordtransform import vessel_cs_to_tracking3d
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    from simulator_modifications.fluoroscopy import TrackingOnly
    assert registry.resolve("fluoroscopy", "TrackingOnly") is TrackingOnly
    tree = BiroPhantomArch(version=version)
    stock_fails = []
    for name, angle in _c_arm_angles().items():
        stock = eve.intervention.fluoroscopy.TrackingOnly(simulation=None, vessel_tree=tree, image_rot_zx=angle)
        try:
            stock.tracking3d_space
        except ValueError:
            stock_fails.append(name)
        fixed = TrackingOnly(simulation=None, vessel_tree=tree, image_rot_zx=angle)
        box = fixed.tracking3d_space
        rotated = vessel_cs_to_tracking3d(_corners(tree.coordinate_space), angle, [0, 0, 0], None)
        np.testing.assert_allclose(box.low, rotated.min(0), rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(box.high, rotated.max(0), rtol=1e-6, atol=1e-4)
        assert np.all(box.low <= box.high)
        fixed.tracking2d_space                  # what Tracking2D / Target2D read: builds too
    assert "G13" not in stock_fails             # the existing single-device configs build either way
    assert stock_fails                           # the bug


def test_tracking_space_fix_leaves_the_observation_values_alone():
    """Tracking2D and Target2D under NormalizeTracking2DEpisode (the single-device configs' 2D
    observations) give the same values with stEVE's TrackingOnly and ours: only the declared space
    changes. G13's angle, where stEVE's box is valid; a wire and a target spread over the phantom."""
    import types

    import eve
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    from simulator_modifications.fluoroscopy import TrackingOnly
    from simulator_modifications.observations import NormalizeTracking2DEpisode, Tracking2D
    tree = BiroPhantomArch(centerline="all_vessels")
    wire = np.concatenate([tree["rcca"].coordinates[::-1], tree["bct"].coordinates[-2::-1]])   # tip first
    simulation = types.SimpleNamespace(dof_positions=wire)
    values, spaces = [], []
    for cls in (eve.intervention.fluoroscopy.TrackingOnly, TrackingOnly):
        fluoroscopy = cls(simulation=simulation, vessel_tree=tree, image_rot_zx=[20, 5])
        target = types.SimpleNamespace(coordinates2d=fluoroscopy.tracking2d[3])
        intervention = types.SimpleNamespace(fluoroscopy=fluoroscopy, vessel_tree=tree, target=target)
        terms = [NormalizeTracking2DEpisode(Tracking2D(intervention, n_points=5), intervention),
                 NormalizeTracking2DEpisode(eve.observation.Target2D(intervention), intervention)]
        for term in terms:
            term.reset()
        values.append([term.obs.copy() for term in terms])
        spaces.append([term.wrapped_obs.space for term in terms])
    for stock, fixed in zip(*values):
        np.testing.assert_array_equal(stock, fixed)
    for stock, fixed in zip(*spaces):
        assert stock.shape == fixed.shape
    assert not np.allclose(spaces[0][1].low, spaces[1][1].low)      # the declared space did change


def test_each_stEVE_name_is_registered_by_one_module_only():
    """Two modules registering the same name would make a config's class the last one imported. In a
    fresh process, every simulator_modifications module imported (last to first), each registration
    is seen once, and ``TrackingOnly`` resolves to the problem-21 fix."""
    code = textwrap.dedent("""
        import collections, importlib, pkgutil
        from guidewire_rl.env import registry
        seen = collections.defaultdict(set)
        real = registry.register
        def spy(kind, name=None):
            decorate = real(kind, name)
            def wrap(cls):
                seen[(kind, name or cls.__name__)].add(cls.__module__)
                return decorate(cls)
            return wrap
        registry.register = spy
        import simulator_modifications
        for module in sorted(pkgutil.iter_modules(simulator_modifications.__path__), key=lambda m: m.name)[::-1]:
            importlib.import_module("simulator_modifications." + module.name)
        twice = {key: sorted(mods) for key, mods in seen.items() if len(mods) > 1}
        print("TWICE", twice)
        print("TRACKING", sorted(seen[("fluoroscopy", "TrackingOnly")]))
        print("CLASS", registry.resolve("fluoroscopy", "TrackingOnly").__module__)
    """)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "TWICE {}" in run.stdout
    assert "TRACKING ['simulator_modifications.fluoroscopy']" in run.stdout
    assert "CLASS simulator_modifications.fluoroscopy" in run.stdout
