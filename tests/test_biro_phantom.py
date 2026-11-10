"""The lab phantom as a vessel tree (guidewire_rl/env/vessels/biro_phantom.py), its aorta
centerline extended to the bottom of the root (biro_phantom_root.py), a centerline for every
vessel (biro_phantom_all.py), the phantom model's own inner wall with centerlines measured to it
(biro_phantom_lumen.py, biro_phantom_lumen_centerlines.py), the phantom versions (the rebuilt one
from the model alone, biro_rebuild.py; the inscribed-sphere wall on its centerlines, biro_spheres.py;
each without the right arm, biro_cut.py), and the configs
that train G13's task in it (configs/phantom_g13.yaml, phantom_friction.yaml, phantom_root.yaml,
phantom_all_vessels.yaml)."""

import copy
import hashlib
import json

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config
from guidewire_rl.env.vessels import biro_phantom

SHA256 = {  # also in guidewire_rl/env/vessels/biro_phantom/README.md
    "phantom_arch.json": "70595a2e046f914ba397624751d24b54a8231673dc3f8b5d0cab346429dcea8a",
    "phantom_arch.obj": "e23430efa986bd93734e7ec613d97a676c35f9e163fb9b86f6731c2906a5db4a",
}
ROOT_SHA256 = "2a2507f0f0088cc1110340f20d90038ed0a5a5f44fb44b8d2fd91db894753063"  # phantom_arch_root.json
ALL_SHA256 = "acdd043a931a1365a3236ba0be44ed004d3774d34b23cd926755c856c5c447e8"  # phantom_arch_all.json
TREE_SHA256 = "0c5bebc39cfdcf37057e97301e53feafb1e33a38ccf87e1851ff4b336c01abc6"  # vessel_tree.npz
LAB_TREE = WORKSPACE_ROOT / "reference" / "surgery-all-in-one" / "19-phantom-sofa" / "derived" / "tree.npz"
LAB_COPY = WORKSPACE_ROOT / "reference" / "surgery-all-in-one" / "47-steve-agent-phantom" / "derived"
PHANTOM_CONFIG = WORKSPACE_ROOT / "configs" / "phantom_g13.yaml"
FRICTION_CONFIG = WORKSPACE_ROOT / "configs" / "phantom_friction.yaml"
REAL_MASS_CONFIG = WORKSPACE_ROOT / "configs" / "phantom_real_mass.yaml"
ROOT_CONFIG = WORKSPACE_ROOT / "configs" / "phantom_root.yaml"
ALL_CONFIG = WORKSPACE_ROOT / "configs" / "phantom_all_vessels.yaml"
ACTIONS = np.asarray([[0.9, 0.1], [0.8, -0.3], [1.0, 0.5], [0.6, -0.9],
                      [0.9, 0.2], [1.0, -0.1], [0.7, 0.8], [0.9, -0.5]] * 4, dtype=np.float32)


@pytest.mark.parametrize("name", sorted(SHA256))
def test_vendored_files_are_the_lab_files(name):
    data = (biro_phantom.DATA_DIR / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256[name]
    lab = LAB_COPY / name
    if lab.is_file():                       # reference/ is only a local checkout
        assert lab.read_bytes() == data


def test_tree_is_what_the_lab_hands_to_FromMesh():
    """Field by field against phantom_arch.json, as the lab's make_phantom_env passes it."""
    spec = json.loads(biro_phantom.SPEC_PATH.read_text())
    tree = biro_phantom.BiroPhantomArch()
    assert [b.name for b in tree.branches] == list(spec["branches"]) == [
        "aorta", "bct", "rsa", "rcca", "lcca", "lsa"]
    for branch in tree.branches:
        given = spec["branches"][branch.name]
        np.testing.assert_allclose(branch.coordinates, np.asarray(given["coordinates"], np.float32),
                                   rtol=0, atol=1e-6)
        np.testing.assert_array_equal(branch.radii, np.asarray(given["radii"], np.float32))
    np.testing.assert_allclose(tree.insertion.position, spec["insertion"]["position"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(tree.insertion.direction, spec["insertion"]["direction"], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(tree.centerline_coordinates,
                                  np.concatenate([b.coordinates for b in tree.branches]))


def test_phantom_config_changes_only_the_vessels_and_the_geometry_frame(upstream_config):
    config = load_config(PHANTOM_CONFIG)
    assert config["env"]["scene"]["vessel_tree"] == {"type": "BiroPhantomArch"}
    geometry = next(t for t in config["env"]["observation"] if t["name"] == "geometry")
    assert geometry["frame"] == "vessel"
    ours, base = copy.deepcopy(config), copy.deepcopy(upstream_config)
    for c in (ours, base):
        del c["run"], c["env"]["scene"]["vessel_tree"]
        next(t for t in c["env"]["observation"] if t["name"] == "geometry").pop("frame")
    assert ours == base


@pytest.mark.sofa
@pytest.mark.upstream
@pytest.mark.parametrize("config_path", [PHANTOM_CONFIG, ROOT_CONFIG, ALL_CONFIG], ids=lambda p: p.stem)
def test_phantom_env_runs_with_every_observation_inside_its_range(config_path):
    from guidewire_rl.env import build_env
    env = build_env(load_config(config_path)["env"])
    try:
        observation, _ = env.reset(seed=1000)
        assert observation.shape == (22,)
        assert type(env.term("position")).__module__ == "simulator_modifications.observations"
        assert type(env.term("position").wrapped_obs).__module__ == "simulator_modifications.observations"
        for action in ACTIONS:
            _, _, terminated, truncated, info = env.step(action)
            raw = np.concatenate([np.asarray(env.term(name).obs, np.float32).reshape(-1)
                                  for name, _ in env.observation_layout])
            assert np.all(np.isfinite(raw)) and np.all(np.abs(raw) <= 1.0)   # before clipping
            assert info["labels"]["distal_force"] >= 0.0
            if terminated or truncated:
                break
        assert env.intervention.instrument_lengths_inserted[0] > 50.0      # the wire went in
    finally:
        env.close()


def test_friction_config_changes_only_the_friction():
    config, base = load_config(FRICTION_CONFIG), load_config(PHANTOM_CONFIG)
    assert config["env"]["scene"]["simulation"] == {"type": "SofaBeamAdapter", "friction": 0.1}
    assert base["env"]["scene"]["simulation"]["friction"] == 0.001
    for c in (config, base):
        del c["run"], c["env"]["scene"]["simulation"]["friction"]
    assert config == base


@pytest.mark.sofa
@pytest.mark.upstream
def test_friction_config_reaches_the_constraint_solver():
    from guidewire_rl.env import build_env
    env = build_env(load_config(FRICTION_CONFIG)["env"])
    try:
        env.reset(seed=1000)
        assert env.intervention.simulation.root.getObject("LCP").mu.value == pytest.approx(0.1)
    finally:
        env.close()


# ---- the aorta's centerline extended to the bottom of the aortic root (biro_phantom_root.py)

def _root_specs():
    lab = json.loads(biro_phantom.SPEC_PATH.read_text())
    ext = json.loads(biro_phantom.CENTERLINES["root_extended"].read_text())
    return lab, ext, ext.pop("root_extension")


def test_root_extended_file_is_the_recorded_one():
    data = biro_phantom.CENTERLINES["root_extended"].read_bytes()
    assert hashlib.sha256(data).hexdigest() == ROOT_SHA256
    _, _, info = _root_specs()
    assert info["from"] == f"phantom_arch.json (sha256 {SHA256['phantom_arch.json']})"
    assert info["wall"] == f"phantom_arch.obj (sha256 {SHA256['phantom_arch.obj']})"


def test_root_extension_changes_only_the_end_of_the_aorta():
    lab, ext, info = _root_specs()
    kept = info["kept_lab_points"]
    old, new = lab["branches"].pop("aorta"), ext["branches"].pop("aorta")
    assert new["coordinates"][:kept] == old["coordinates"][:kept]
    assert new["radii"][:kept] == old["radii"][:kept]
    assert kept + info["replaced_lab_points"] == len(old["coordinates"])
    assert len(new["coordinates"]) == len(new["radii"]) == kept + info["new_points"]
    assert lab["lengths_mm"].pop("aorta") == 250.6 and ext["lengths_mm"].pop("aorta") > 260
    assert ext == lab                                   # every other field and branch as the lab's
    coordinates = np.asarray(new["coordinates"])
    # the collision mesh's root ends at z 4.4: the line stops 1 mm above it (wire tips get no closer)
    assert coordinates[-1, 2] < 6.0 and new["radii"][-1] == pytest.approx(1.0, abs=0.05)
    steps = np.diff(coordinates[kept - 3:], axis=0)
    assert np.all(np.linalg.norm(steps, axis=1) > 1.2)
    unit = steps / np.linalg.norm(steps, axis=1)[:, None]
    assert np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))).max() < 25


def test_root_extension_lies_on_the_ridge_of_the_lumen():
    """Each new point's radius is its distance to the collision mesh, no point of its
    cross-section is much farther from the wall, and the line ends at the wall."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from guidewire_rl.env.vessels import biro_phantom_root as root
    _, ext, info = _root_specs()
    kept = info["kept_lab_points"]
    coordinates = np.asarray(ext["branches"]["aorta"]["coordinates"])
    radii = np.asarray(ext["branches"]["aorta"]["radii"])
    end = coordinates[kept - 1]
    low, high = np.r_[end[:2] - 25, 0.0], np.r_[end[:2] + 25, 45.0]
    wall = root.Wall(trimesh.load(biro_phantom.MESH_PATH, process=True), low, high, coordinates[150])
    np.testing.assert_allclose(wall.distance(coordinates[kept:]), radii[kept:], atol=0.01)
    offsets = np.array([(x, y) for x in np.linspace(-2, 2, 9) for y in np.linspace(-2, 2, 9)])
    for i in range(kept, len(coordinates) - 1):
        tangent = coordinates[i + 1] - coordinates[i - 1]
        tangent /= np.linalg.norm(tangent)
        u = np.cross(tangent, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        section = coordinates[i] + offsets[:, :1] * u + offsets[:, 1:] * v
        assert wall.distance(section).max() - radii[i] < 0.4, i
    past = coordinates[-1] + 1.5 * (coordinates[-1] - coordinates[-2]) / np.linalg.norm(
        coordinates[-1] - coordinates[-2])
    assert wall.distance(past[None])[0] < 0                 # 1.5 mm further on is outside the lumen


@pytest.mark.upstream
def test_root_extension_moves_only_the_end_of_the_vessel():
    """Branching points, target candidates, the 2D range and the coordinate space are as
    before; stEVE now stops pushing at the bottom of the root instead of 16 mm above it."""
    import types

    from eve.intervention.target import CenterlineRandom
    from eve.intervention.vesseltree.vesseltree import at_tree_end
    from simulator_modifications.observations import lumen_bounds_2d

    old = biro_phantom.BiroPhantomArch()
    new = biro_phantom.BiroPhantomArch(centerline="root_extended")
    points = lambda tree: sorted((tuple(np.round(p.coordinates, 4)), round(float(p.radius), 4),
                                  tuple(sorted(b.name for b in p.connections)))
                                 for p in tree.branching_points)
    assert points(new) == points(old)

    def targets(tree):
        target = CenterlineRandom(tree, None, 5.0, ["lcca", "rcca", "lsa", "rsa", "bct", "co"])
        target._init_centerline_point_cloud()
        return target._potential_targets
    np.testing.assert_array_equal(targets(new), targets(old))
    fluoroscopy = types.SimpleNamespace(image_rot_zx=[20, 5], image_center=[0, 0, 0], field_of_view=None)
    for a, b in ((lumen_bounds_2d(new, fluoroscopy), lumen_bounds_2d(old, fluoroscopy)),
                 (new.coordinate_space, old.coordinate_space)):
        np.testing.assert_array_equal(a.low, b.low)
        np.testing.assert_array_equal(a.high, b.high)
    for point in ([5.0, -72.5, 17.0], [7.0, -74.0, 10.0]):          # inside the root, below z 20.6
        assert at_tree_end(np.asarray(point), old) and not at_tree_end(np.asarray(point), new)
    assert at_tree_end(np.asarray([7.385, -74.642, 5.0]), new)       # past the new end


def test_root_config_changes_only_the_centerline():
    config, base = load_config(ROOT_CONFIG), load_config(FRICTION_CONFIG)
    assert config["env"]["scene"]["vessel_tree"] == {"type": "BiroPhantomArch", "centerline": "root_extended"}
    for c in (config, base):
        del c["run"], c["env"]["scene"]["vessel_tree"]
    assert config == base


def test_unknown_centerline_is_refused():
    with pytest.raises(ValueError, match="centerline"):
        biro_phantom.BiroPhantomArch(centerline="lab")


# ---- a centerline for every vessel of the phantom (biro_phantom_all.py)

def _all_specs():
    root = json.loads(biro_phantom.CENTERLINES["root_extended"].read_text())
    full = json.loads(biro_phantom.CENTERLINES["all_vessels"].read_text())
    return root, full, full.pop("all_vessels")


def test_all_vessels_files_are_the_recorded_ones():
    assert hashlib.sha256(biro_phantom.CENTERLINES["all_vessels"].read_bytes()).hexdigest() == ALL_SHA256
    tree = (biro_phantom.DATA_DIR / "vessel_tree.npz").read_bytes()
    assert hashlib.sha256(tree).hexdigest() == TREE_SHA256
    if LAB_TREE.is_file():                     # reference/ is only a local checkout
        assert LAB_TREE.read_bytes() == tree


def test_all_vessels_adds_branches_and_keeps_the_six():
    root, full, info = _all_specs()
    root_extension = root.pop("root_extension")
    assert info["root_extension"] == root_extension and info["left_out_not_in_mesh"] == {}
    six = list(root["branches"])
    assert full.pop("task_branches") == six
    assert list(full["branches"])[:6] == six
    for name in six:
        assert full["branches"][name] == root["branches"][name]
    added = {k: v for k, v in full["branches"].items() if k not in six}
    assert set(added) == set(info["added"]) and len(added) == 69
    assert {"v0_inlet", "v6_rest", "v7_rest", "v8_rest", "v10_rest"} <= set(added)
    assert all(min(b["radii"]) >= 1.0 and len(b["coordinates"]) >= 2 for b in added.values())
    full["branches"] = {k: full["branches"][k] for k in six}
    assert full == root                                # every other field as before
    # a whole lab vessel, moved with the lab's transform, radius at least 1 mm
    tree = np.load(biro_phantom.DATA_DIR / "vessel_tree.npz")
    theta = np.radians(root["rotation_z_deg"])
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    moved = (tree["vp_v12"] - np.asarray(root["origin_in_phantom_mm"])) @ rotation.T
    np.testing.assert_allclose(added["v12"]["coordinates"], moved, atol=6e-4)
    np.testing.assert_allclose(added["v12"]["radii"], np.maximum(tree["vr_v12"], 1.0), atol=6e-4)


def test_all_vessels_fill_the_collision_mesh():
    """Random points inside the collision mesh: the six branch tubes hold about 72 % of them,
    all branches nearly all (the mesh is the union of balls along these lines)."""
    trimesh = pytest.importorskip("trimesh")
    from scipy.spatial import cKDTree

    mesh = trimesh.load(biro_phantom.MESH_PATH, process=True)
    samples, faces = trimesh.sample.sample_surface(mesh, int(mesh.area * 20), seed=0)
    normals = mesh.face_normals[faces]
    nearest = cKDTree(samples)
    points = np.random.default_rng(0).uniform(*mesh.bounds, size=(800_000, 3))
    _, j = nearest.query(points, workers=-1)
    depth = -np.einsum("ij,ij->i", points - samples[j], normals[j])      # mesh normals point inward
    flip = 1.0 if np.mean(depth > 0) < 0.5 else -1.0                     # the lumen is the small part
    points = points[flip * depth > 0.3]

    def covered(spec):
        s = np.concatenate([np.asarray(b["coordinates"][:-1]) for b in spec["branches"].values()])
        e = np.concatenate([np.asarray(b["coordinates"][1:]) for b in spec["branches"].values()])
        r0 = np.concatenate([np.asarray(b["radii"][:-1]) for b in spec["branches"].values()])
        r1 = np.concatenate([np.asarray(b["radii"][1:]) for b in spec["branches"].values()])
        v, best = e - s, np.full(len(points), -np.inf)
        for i in range(0, len(points), 500):
            x = points[i:i + 500, None]
            t = np.clip(np.einsum("pij,ij->pi", x - s, v) / np.einsum("ij,ij->i", v, v), 0, 1)
            best[i:i + 500] = (r0 + t * (r1 - r0) - np.linalg.norm(x - (s + t[..., None] * v), axis=-1)).max(1)
        return float(np.mean(best >= 0))

    root, full, _ = _all_specs()
    assert len(points) > 2000
    assert covered(root) < 0.8 and covered(full) > 0.97


@pytest.mark.upstream
def test_all_vessels_change_only_the_lumen_not_the_task():
    """Targets and the 2D range come from the six task branches as before; stEVE no longer
    stops the push at the cut ends of rsa, rcca, lcca, lsa; the root's bottom is still an end."""
    import types

    from eve.intervention.vesseltree.vesseltree import at_tree_end
    from simulator_modifications.observations import lumen_bounds_2d
    from simulator_modifications.seeds import CenterlineRandom

    root = biro_phantom.BiroPhantomArch(centerline="root_extended")
    full = biro_phantom.BiroPhantomArch(centerline="all_vessels")
    assert full.task_branch_names == tuple(b.name for b in root.branches) == root.task_branch_names
    assert len(full.branches) == 75

    def targets(tree):
        target = CenterlineRandom(tree, None, 5.0, ["lcca", "rcca", "lsa", "rsa", "bct", "co"])
        target._init_centerline_point_cloud()
        return target._potential_targets
    np.testing.assert_array_equal(targets(full), targets(root))
    fluoroscopy = types.SimpleNamespace(image_rot_zx=[20, 5], image_center=[0, 0, 0], field_of_view=None)
    a, b = lumen_bounds_2d(full, fluoroscopy), lumen_bounds_2d(root, fluoroscopy)
    np.testing.assert_array_equal(a.low, b.low)
    np.testing.assert_array_equal(a.high, b.high)
    for name in ("rsa", "rcca", "lcca", "lsa"):
        c = root[name].coordinates.astype(float)
        past = c[-1] + 3.0 * (c[-1] - c[-2]) / np.linalg.norm(c[-1] - c[-2])
        assert at_tree_end(past, root) and not at_tree_end(past, full), name
    assert at_tree_end(np.asarray([7.385, -74.642, 5.0]), full)


def test_all_vessels_config_changes_only_the_centerline():
    config, base = load_config(ALL_CONFIG), load_config(ROOT_CONFIG)
    assert config["env"]["scene"]["vessel_tree"] == {"type": "BiroPhantomArch", "centerline": "all_vessels"}
    for c in (config, base):
        del c["run"], c["env"]["scene"]["vessel_tree"]
    assert config == base


# ---- the phantom model's own inner wall (biro_phantom_lumen.py) and centerlines measured to it
# (biro_phantom_lumen_centerlines.py); not wired into BiroPhantomArch yet

LUMEN_SHA256 = {"phantom_lumen.obj": "993a760af08dc0ffead6421389c4e53491952f6403dac5ea520842855858bf61", "phantom_lumen_all.json": "4c642ffa38a337e90204332a1da73a3c316bcfe5b9fcc36097bd1690459f3ca7"}


@pytest.mark.parametrize("name", sorted(LUMEN_SHA256))
def test_lumen_files_are_the_recorded_ones(name):
    assert hashlib.sha256((biro_phantom.DATA_DIR / name).read_bytes()).hexdigest() == LUMEN_SHA256[name]
    record = json.loads((biro_phantom.DATA_DIR / "phantom_lumen.json").read_text())
    assert record["sha256"] == LUMEN_SHA256["phantom_lumen.obj"]


def test_lumen_centerlines_keep_the_lines_and_change_the_radii():
    old = json.loads(biro_phantom.CENTERLINES["all_vessels"].read_text())
    new = json.loads((biro_phantom.DATA_DIR / "phantom_lumen_all.json").read_text())
    assert new["task_branches"] == old["task_branches"]
    kept = new["all_vessels"]["root_extension"]["kept_lab_points"]
    lab = json.loads(biro_phantom.SPEC_PATH.read_text())["branches"]["aorta"]["coordinates"]
    assert new["branches"]["aorta"]["coordinates"][:kept] == lab[:kept]
    assert set(old["branches"]) - set(new["branches"]) == set(new["all_vessels"]["left_out_not_in_mesh"])
    for name in set(new["branches"]) - {"aorta"}:
        assert new["branches"][name]["coordinates"] == old["branches"][name]["coordinates"], name
    radii = np.concatenate([b["radii"] for b in new["branches"].values()])
    assert radii.min() >= 0.3


def test_lumen_wall_is_closed_and_the_radii_reach_it():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    from guidewire_rl.env.vessels.biro_phantom_root import Wall

    mesh = trimesh.load(biro_phantom.DATA_DIR / "phantom_lumen.obj", process=True)
    assert mesh.is_watertight and abs(mesh.volume) / 1000 == pytest.approx(255.9, abs=0.1)
    new = json.loads((biro_phantom.DATA_DIR / "phantom_lumen_all.json").read_text())
    points = np.concatenate([b["coordinates"] for b in new["branches"].values()])
    radii = np.concatenate([b["radii"] for b in new["branches"].values()])
    pick = np.random.default_rng(0).choice(len(points), 200, replace=False)
    wall = Wall(mesh, mesh.bounds[0], mesh.bounds[1], inside_point=new["branches"]["aorta"]["coordinates"][100])
    np.testing.assert_allclose(np.maximum(wall.distance(points[pick]), 0.3), radii[pick], atol=0.002)


# ---- the guidewire at its real mass: 2 ms physics step, 2 mm alarm distance (problem 20)

NITINOL = 6.45e-9      # t/mm^3; stEVE's 2.1e-5 is platinum written in kg/mm^3


def test_real_mass_config_changes_only_mass_step_alarm_and_substeps():
    config, base = load_config(REAL_MASS_CONFIG), load_config(FRICTION_CONFIG)
    scene, base_scene = config["env"]["scene"], base["env"]["scene"]
    assert scene["instruments"] == [{"type": "Angled", "mass_density": NITINOL, "flex_mass_density": NITINOL}]
    assert base_scene["instruments"] == [{"type": "Angled"}]
    assert scene["simulation"] == {"type": "SofaBeamAdapter", "friction": 0.1,
                                   "dt_simulation": 0.002, "alarm_distance": 2.0}
    assert [l["expected_substeps"] for l in config["env"]["labels"]] == [66]
    assert [l["expected_substeps"] for l in base["env"]["labels"]] == [22]
    for c in (config, base):
        del c["run"], c["env"]["scene"]["instruments"], c["env"]["scene"]["simulation"]
        for label in c["env"]["labels"]:
            del label["expected_substeps"]
    assert config == base


@pytest.mark.sofa
@pytest.mark.upstream
def test_real_mass_config_reaches_the_scene_and_a_control_step_has_66_substeps():
    from guidewire_rl.env import build_env
    env = build_env(load_config(REAL_MASS_CONFIG)["env"])
    try:
        env.reset(seed=1000)
        root = env.intervention.simulation.root
        assert root.dt.value == pytest.approx(0.002)
        assert root.getObject("localmindistance").alarmDistance.value == pytest.approx(2.0)
        assert root.getObject("localmindistance").contactDistance.value == pytest.approx(0.3)
        sections = [o for o in root.topolines_guidewire.objects if o.getClassName().startswith("Rod")]
        assert len(sections) == 2 and all(o.massDensity.value == pytest.approx(NITINOL) for o in sections)
        for action in ACTIONS[:6]:                      # strong pushes, 0.8 s of simulated time
            obs, _, _, _, info = env.step(action)
            assert np.all(np.isfinite(obs))
            assert info["label_details"]["distal_force"]["substeps"] == 66
        assert not env.intervention.simulation.simulation_error
    finally:
        env.close()


# ---- phantom versions: lab algorithm, rebuilt and spheres, whole and cut (biro_rebuild.py, biro_spheres.py, biro_cut.py)

VERSION_SHA256 = {  # also in guidewire_rl/env/vessels/biro_phantom/README.md
    "rebuilt_full.obj": "e3d22f5b375749a15fca0366968d8f0cfafd369bac1c5d566bae5ef37e4f300c",
    "rebuilt_full.json": "e36bcf3f37605ea29c3d4f7ac2f6f0b172027d203444c24c576486039efede33",
    "rebuilt_cut.obj": "a358f8f5714679f6f1062119cf262e8d33b5a9cec241da8c6c17039f09c399a1",
    "rebuilt_cut.json": "177a796db47ca1a3827629f1bd6c86cc08480fffbc664559d0240cdf35cb53a9",
    "lab_cut.obj": "bca7b783970b180307052740f1d9a3257e513f54d8cb962675c00d94575468b5",
    "lab_cut.json": "945de2eca4c078494bff7d2943c06da933abbf67efdd15e5ad7be3e6a6c280ce",
    "spheres_full.obj": "e336d97d24b9a29bf2c9735335b4981f109a8e59c461f0ba1f030f63753bcafe",
    "spheres_full.json": "2116e87bf6b289151ec68984727cef4a3ed4e1fb107e065b034d4a73efef9b58",
    "spheres_cut.obj": "97aea5548cd0fa9853c0120b40f208b965db512618dbdcd4622f2683121caa64",
    "spheres_cut.json": "f5e96faba09d7674f0a068d4e2e6e0287f83f1dfc3bad78d172c0a3dd4f144e0",
}
VOLUME_ML = {"lab_cut": 196.1, "rebuilt": 256.6, "rebuilt_cut": 238.7, "spheres": 235.4, "spheres_cut": 219.3}


def version_spec(version):
    return json.loads(biro_phantom.VERSIONS[version][1].read_text())


def signed_distance(mesh_path, points, inside_point):
    """Distance to the wall, positive on the side of ``inside_point`` (in the lumen)."""
    pv = pytest.importorskip("pyvista")
    mesh = pv.read(str(mesh_path))
    query = np.vstack([np.asarray(points, float), np.asarray(inside_point, float)])
    d = np.asarray(pv.PolyData(query).compute_implicit_distance(mesh)["implicit_distance"])
    return d[:-1] * np.sign(d[-1])


@pytest.mark.parametrize("name", sorted(VERSION_SHA256))
def test_version_files_are_the_recorded_ones(name):
    assert hashlib.sha256((biro_phantom.DATA_DIR / name).read_bytes()).hexdigest() == VERSION_SHA256[name]


def test_rebuilt_reads_only_the_model_and_the_fixed_task_numbers():
    from guidewire_rl.env.vessels import biro_rebuild

    spec, lab = version_spec("rebuilt"), json.loads(biro_phantom.SPEC_PATH.read_text())
    assert spec["source"]["model"].endswith(f"(sha256 {biro_rebuild.SOURCE[1]})")
    assert spec["frame_transform"] == {"rotation_z_deg": lab["rotation_z_deg"],
                                       "origin_in_phantom_mm": lab["origin_in_phantom_mm"]}
    assert spec["task_definition"]["first_ostium_from_start_mm"] == lab["first_ostium_from_start_mm"]
    for name, length in spec["task_definition"]["branch_lengths_mm"].items():
        assert length == lab["paper_lengths_mm"][name]
    assert spec["task_branches"] == ["aorta", "bct", "rsa", "rcca", "lcca", "lsa"]


@pytest.mark.parametrize("version", sorted(VOLUME_ML))
def test_version_walls_are_closed(version):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(biro_phantom.VERSIONS[version][0], process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert abs(mesh.volume) / 1000 == pytest.approx(VOLUME_ML[version], abs=0.1)


def test_rebuilt_task_branches_follow_lab_47s_definition():
    spec = version_spec("rebuilt")
    br = {n: np.asarray(b["coordinates"]) for n, b in spec["branches"].items()}
    length = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    for name, want in spec["task_definition"]["branch_lengths_mm"].items():
        assert length(br[name]) == pytest.approx(want, abs=1.0), name   # points every 0.73 mm cut corners
    # the aorta starts 146.7 mm before the left subclavian leaves it; its second point is the insertion
    aorta = np.vstack([br["aorta_inlet"], br["aorta"][1:]])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(aorta, axis=0), axis=1))]
    s_lsa = arc[np.argmin(np.linalg.norm(aorta - br["lsa"][0], axis=1))]
    s_start = length(br["aorta_inlet"])
    assert s_lsa - s_start == pytest.approx(spec["task_definition"]["first_ostium_from_start_mm"], abs=1.0)
    np.testing.assert_allclose(spec["insertion"]["position"], br["aorta"][1], atol=1e-3)
    # the trunk ends where the right common carotid and subclavian begin; the carotid rises more
    np.testing.assert_allclose(br["rcca"][0], br["bct"][-1], atol=1e-3)
    np.testing.assert_allclose(br["rsa"][0], br["bct"][-1], atol=1e-3)
    assert br["rcca"][-1, 2] - br["rcca"][0, 2] > br["rsa"][-1, 2] - br["rsa"][0, 2]
    # the aorta ends at the bottom of its root, below where the coronaries leave
    assert br["aorta"][-1, 2] < 10.0


def test_rebuilt_centerlines_are_inside_and_their_radius_is_the_distance_to_the_wall():
    spec = version_spec("rebuilt")
    mesh = biro_phantom.VERSIONS["rebuilt"][0]
    inner = []                      # leave out the ends: next to a cap the radius is held (open_end_radii)
    for b in spec["branches"].values():
        c, r = np.asarray(b["coordinates"]), np.asarray(b["radii"])
        keep = slice(len(c) // 5, len(c) - len(c) // 5) if len(c) >= 10 else slice(0, 0)
        inner.append((c[keep], r[keep]))
    points = np.concatenate([c for c, _ in inner])
    radii = np.concatenate([r for _, r in inner])
    distance = signed_distance(mesh, points, spec["branches"]["aorta"]["coordinates"][60])
    assert distance.min() > 0                                     # every point inside the lumen
    close = np.abs(np.maximum(distance, 0.3) - radii) < 0.01
    assert close.mean() > 0.99


def test_rebuilt_junctions_join_every_branch_to_the_aorta():
    spec = version_spec("rebuilt")
    br = {n: np.asarray(b["coordinates"]) for n, b in spec["branches"].items()}
    reached, todo = {"aorta"}, ["aorta"]
    while todo:
        name = todo.pop()
        for j in spec["junctions"]:
            if name in j["branches"]:
                for other in j["branches"]:
                    if other not in reached:
                        reached.add(other)
                        todo.append(other)
    assert reached == set(br)
    for j in spec["junctions"]:                                    # a junction lies on its branches
        for name in j["branches"]:
            assert np.linalg.norm(br[name] - j["point"], axis=1).min() < 1.0, (name, j["point"])
    # the left common carotid and subclavian leave the aorta at junctions of their own
    lcca = next(j for j in spec["junctions"] if "lcca" in j["branches"] and "aorta" in j["branches"])
    assert "lsa" not in lcca["branches"]


def test_spheres_are_built_on_the_rebuilt_centerlines_only():
    from guidewire_rl.env.vessels import biro_spheres

    spec, rebuilt = version_spec("spheres"), version_spec("rebuilt")
    sha = hashlib.sha256(biro_phantom.VERSIONS["rebuilt"][1].read_bytes()).hexdigest()
    assert spec["source"]["centerlines"].startswith(f"rebuilt_full.json (sha256 {sha})")
    for key in ("insertion", "task_branches", "junctions", "frame_transform", "task_definition"):
        assert spec[key] == rebuilt[key], key
    assert set(spec["branches"]) == set(rebuilt["branches"])
    for name, b in spec["branches"].items():
        assert b["coordinates"] == rebuilt["branches"][name]["coordinates"]
        np.testing.assert_allclose(b["radii"], np.maximum(rebuilt["branches"][name]["radii"], biro_spheres.R_MIN_MM),
                                   atol=1e-3)


def test_spheres_wall_lies_just_inside_the_model_wall_on_the_task_branches():
    trimesh = pytest.importorskip("trimesh")
    from scipy.spatial import cKDTree

    spec = version_spec("spheres")
    wall = trimesh.load(biro_phantom.VERSIONS["spheres"][0], process=False)
    names = np.concatenate([[n] * len(b["coordinates"]) for n, b in spec["branches"].items()])
    lines = np.vstack([b["coordinates"] for b in spec["branches"].values()])
    _, nearest = cKDTree(lines).query(wall.vertices)
    on_task = np.isin(names[nearest], spec["task_branches"])
    inside = signed_distance(biro_phantom.VERSIONS["rebuilt"][0], wall.vertices[on_task],
                             spec["branches"]["aorta"]["coordinates"][60])
    assert inside.min() > -0.1                                   # never past the model's own wall
    assert np.median(inside) < 0.4                               # and close to it (the lab's: 0.7 mm)


@pytest.mark.parametrize("prefix", ["lab", "rebuilt", "spheres"])
def test_cut_versions_lose_the_arm_and_nothing_else(prefix):
    trimesh = pytest.importorskip("trimesh")
    full = version_spec(prefix)
    cut = version_spec(f"{prefix}_cut")
    record = cut["cut"]
    point, normal = np.asarray(record["point"]), np.asarray(record["normal"])
    assert set(full["branches"]) - set(cut["branches"]) == set(record["dropped"])
    for name in full["task_branches"]:                             # the task is untouched
        assert cut["branches"][name] == full["branches"][name]
    left = np.vstack([full["branches"]["lsa"]["coordinates"],
                      full["branches"]["v6_rest" if prefix == "lab" else "lsa_rest"]["coordinates"][1:]])
    assert record["left_subclavian_mm"] == pytest.approx(np.linalg.norm(np.diff(left, axis=0), axis=1).sum(), abs=0.1)
    shortened = np.asarray(cut["branches"][record["shortened"]]["coordinates"])
    assert (shortened - point) @ normal == pytest.approx(np.r_[(shortened[:-1] - point) @ normal, 0.0], abs=1e-6)
    assert np.all((shortened[:-1] - point) @ normal < 0)
    wall = trimesh.load(biro_phantom.VERSIONS[f"{prefix}_cut"][0], process=False)
    beyond = wall.vertices[(wall.vertices - point) @ normal > 1e-3]
    # what is left beyond the plane is the rest of the body, not the arm next to the cut
    assert np.linalg.norm(beyond - point, axis=1).min() > 20.0
    assert wall.vertices[:, 0].min() > -100.0                      # the arm reached x -290


@pytest.mark.sofa
@pytest.mark.upstream
@pytest.mark.parametrize("version", sorted(biro_phantom.VERSIONS))
def test_every_version_runs_in_the_simulator(version):
    from guidewire_rl.env import build_env

    config = load_config(FRICTION_CONFIG)
    config["env"]["scene"]["vessel_tree"] = {"type": "BiroPhantomArch", "version": version}
    env = build_env(config["env"])
    try:
        obs, _ = env.reset(seed=2)
        tree = env.intervention.vessel_tree
        assert tree.task_branch_names == ("aorta", "bct", "rsa", "rcca", "lcca", "lsa")
        spec = version_spec(version)
        if "junctions" in spec:                                    # listed, not guessed from the radii
            assert len(tree.branching_points) == len(spec["junctions"])
        for action in ACTIONS:
            obs, _, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)) and not info.get("simulation_error")
            if terminated or truncated:
                env.reset(seed=3)
    finally:
        env.close()
