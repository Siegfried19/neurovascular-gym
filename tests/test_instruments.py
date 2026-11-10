"""Our instruments (guidewire_rl/env/instruments/): the Simmons 2 catheter traced from a real one."""

import copy
import hashlib
import os

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config

TRACE = WORKSPACE_ROOT / "guidewire_rl" / "env" / "instruments" / "data" / "simmons.csv"
TRACE_MD5 = "6978de93499064bc16f3c3eb75378b62"      # Zenodo 11203294, 2.5D Aortic Arch/Herniation/simmons.csv

GUIDEWIRE = {"type": "Angled", "name": "guidewire", "length": 450, "diameter_outer": 0.89,
             "young_modulus": 24900.0, "flex_length": 30.0, "flex_diameter_outer": 0.89,
             "flex_young_modulus": 325.0, "tip_radius": 12.1, "tip_angle_deg": 72.0,
             "mass_density": 6.45e-9, "flex_mass_density": 6.45e-9}


def test_trace_is_the_unchanged_zenodo_file():
    assert hashlib.md5(TRACE.read_bytes()).hexdigest() == TRACE_MD5


@pytest.mark.upstream
def test_yaml_name_resolves_to_simmons2():
    from guidewire_rl.env import registry
    from guidewire_rl.env.instruments import Simmons2
    assert registry.resolve("instrument", "Simmons2") is Simmons2


@pytest.mark.upstream
def test_shaft_first_then_a_mesh_head():
    """docs/simulator_changes.md problem 17: a straight section only as the first one."""
    from eve.intervention.instrument import MeshSection, StraightSection
    from guidewire_rl.env.instruments import Simmons2
    catheter = Simmons2()
    assert type(catheter.base_section) is StraightSection and isinstance(catheter.tip_section, MeshSection)
    assert catheter.base_section.length == 500
    for section in (catheter.base_section, catheter.tip_section):
        assert (section.diameter_outer, section.diameter_inner) == (1.67, 1.22)
        assert section.young_modulus == 1.78e3 and section.mass_density == 1.2e-9


@pytest.mark.upstream
def test_head_follows_the_trace_point_by_point():
    """The head stEVE lays out (the polyline BeamAdapter reads as the rest shape) against every
    traced point of the head: mean <= 0.3 mm, max <= 2.5 mm (todo_list/two_devices_measurements.md)."""
    from eve.intervention.instrument.util.meshing import create_shape_point_cloud
    from scipy.spatial import cKDTree
    from guidewire_rl.env.instruments import Simmons2, simmons2
    catheter = Simmons2()
    shape, length = create_shape_point_cloud(catheter.tip_section.mesh_elements)
    trace = simmons2.head_trace_in_mesh_frame(TRACE)
    to_shape, _ = cKDTree(shape).query(trace)
    to_trace, _ = cKDTree(trace).query(shape)
    assert to_shape.mean() <= 0.3 and to_shape.max() <= 2.5
    assert to_trace.max() <= 2.5                       # and the head does not run past the trace
    assert length == pytest.approx(simmons2.arc_length(trace[:, :2])[-1], abs=1.0)
    assert np.allclose(shape[:, 2], 0.0)              # one plane, as the photo
    turns = np.array([turn for _, turn in simmons2.head_pieces()])
    assert np.abs(np.cumsum(turns)).max() > 150        # the reversed curve


def two_device_config():
    """phantom_real_mass.yaml (real mass, 2 ms, friction 0.1) on the rebuilt phantom with a
    guidewire and the Simmons 2."""
    config = copy.deepcopy(load_config(WORKSPACE_ROOT / "configs" / "phantom_real_mass.yaml"))
    env = config["env"]
    env["scene"]["vessel_tree"] = {"type": "BiroPhantomArch", "version": "rebuilt_cut"}
    env["scene"]["instruments"] = [GUIDEWIRE, {"type": "Simmons2"}]
    env["scene"]["target"] = {"type": "CenterlineRandom", "threshold": 5.0, "branches": ["rcca"]}
    env.pop("pathfinder", None)
    env["observation"] = [
        {"name": "position", "type": "Tracking2D", "n_points": 5, "wrappers": ["NormalizeTracking2DEpisode"]},
        {"name": "rotation", "type": "Rotations"},
        {"name": "inserted", "type": "InsertionLengths"}]
    env["reward"] = [{"type": "TargetReached", "factor": 1.0}]
    return config


@pytest.fixture(scope="module")
def two_devices():
    from guidewire_rl.env import build_env
    env = build_env(two_device_config()["env"])
    yield env
    env.close()


@pytest.mark.sofa
@pytest.mark.upstream
def test_runs_with_a_guidewire_in_the_phantom(two_devices):
    """Builds, and 32 steps (guidewire leads, then both pushed) run without a simulation error."""
    env = two_devices
    env.reset(seed=3)
    for step in range(32):
        action = np.array([0.6, 0.2, 0.6 if step >= 4 else 0.0, -0.2], dtype=np.float32)
        observation, _, _, _, info = env.step(action)
        assert np.all(np.isfinite(observation)) and not info["simulation_error"]
        assert np.isfinite(info["labels"]["distal_force"])
    wire, catheter = env.intervention.instrument_lengths_inserted
    assert 100 < catheter < wire                       # both went in, the guidewire ahead


@pytest.mark.sofa
@pytest.mark.upstream
def test_inner_diameter_reaches_beamadapter(two_devices):
    """simulator_modifications/hollow_catheter.py sets innerRadius only when diameter_inner > 0
    (test_simulator_modifications.py shows that innerRadius is what makes the tube bend as hollow)."""
    env = two_devices
    env.reset(seed=3)
    root = env.simulation.root
    for prefix in ("base_section_", "tip_section_"):
        rod = root.getChild("topolines_simmons2").getObject(prefix + "simmons2")
        assert rod.radius.value == pytest.approx(1.67 / 2)
        assert rod.innerRadius.value == pytest.approx(1.22 / 2)
        assert root.getChild("topolines_guidewire").getObject(prefix + "guidewire").innerRadius.value == 0.0


@pytest.mark.sofa
@pytest.mark.upstream
def test_beamadapter_holds_the_traced_head_in_free_space(tmp_path):
    """Pushed into empty space, the head takes the traced shape (rest shape read correctly)."""
    from scipy.spatial import cKDTree
    from guidewire_rl.env.instruments import Simmons2, simmons2
    from simulator_modifications.speedups import SofaBeamAdapter
    mesh = os.path.join(tmp_path, "far.obj")
    with open(mesh, "w") as fh:
        fh.write("v -5000 -5000 -5000\nv -5001 -5000 -5000\nv -5000 -5001 -5000\nf 1 2 3\n")
    catheter = Simmons2()
    sim = SofaBeamAdapter(friction=0.0, dt_simulation=0.002, alarm_distance=2.0)
    try:
        sim.reset(insertion_point=np.zeros(3), insertion_direction=np.array([1.0, 0.0, 0.0]),
                  mesh_path=mesh, instruments=[catheter],
                  coords_high=np.full(3, 1000.0), coords_low=np.full(3, -1000.0))
        for _ in range(32):                             # 30 mm/s for 4.3 s: the whole head is out
            sim.step(np.array([[30.0, 0.0]]), duration=1 / 7.5)
        for _ in range(5):
            sim.step(np.array([[0.0, 0.0]]), duration=1 / 7.5)
        nodes = np.array(sim._instruments_combined.DOFs.position.value)[:, :3]
    finally:
        sim.close()
    s = simmons2.arc_length(nodes)
    head = nodes[s >= s[-1] - catheter.tip_section.length - 1e-6]

    def even(points, n=400):
        t = simmons2.arc_length(points)
        u = np.linspace(0.0, t[-1], n)
        return np.stack([np.interp(u, t, points[:, k]) for k in range(3)], axis=1)

    a = even(head)                                       # rigid best fit onto the traced head
    b = even(simmons2.head_trace_in_mesh_frame(TRACE))
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    distance, _ = cKDTree((a - ca) @ rotation.T + cb).query(b)
    assert distance.mean() <= 0.3 and distance.max() <= 2.5
