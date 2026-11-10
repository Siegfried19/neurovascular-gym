"""One C-arm angle per target branch in training (``env.scene.fluoroscopy.image_rot_zx: per_target``,
guidewire_rl/env/c_arm/per_target.py).

Without SOFA: the switch, the table lookup and its errors (no table, a table from another version of
the centerline file, a target off the five branches), continuations; on the scene alone (vessel tree,
fluoroscopy, target; no simulation reset): the angle follows the target's branch from episode to
episode with G13's five target branches on the lab phantom, the right carotid's angle for the
two-device target on both pieces of the carotid, and the target lands where a C-arm fixed at that
angle puts it, to the bit; the vessel geometry's 2D tangents follow the angle. With SOFA: the
two-device env with the switch and the same env with the right carotid's angle written in give the
same observations (target and junction map to the bit, the rest to 1e-6), rewards and task box, step
by step; a config without the switch is built as before; a single-device phantom config with the switch
builds (problem 21's fix) and runs, each episode at its target branch's angle.

Each branch's angle is the one chosen for its whole route from the aorta (choose_angle.route; the
right carotid's: aorta 60 mm + trunk + carotid, LAO 15, cranial 7.5 on the model-wall phantom).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config
from guidewire_rl.env.c_arm import choose_angle as ca
from guidewire_rl.env.c_arm import per_target

CONFIGS = WORKSPACE_ROOT / "configs"
TWO_DEVICES = CONFIGS / "two_devices_round1.yaml"
ICA = CONFIGS / "two_devices_ica.yaml"
ALL_VESSELS = CONFIGS / "phantom_all_vessels.yaml"
SWITCH = "env.scene.fluoroscopy.image_rot_zx=per_target"


def _table(which):
    return json.loads(ca.table_path(which).read_text())


def _fixed(angle):
    return "env.scene.fluoroscopy.image_rot_zx=" + json.dumps([float(a) for a in angle])


# ------------------------------------------------------------------ no SOFA ----
def test_the_switch_is_one_value():
    assert per_target.is_per_target({"type": "TrackingOnly", "image_rot_zx": "per_target"})
    assert not per_target.is_per_target({"type": "TrackingOnly", "image_rot_zx": [61.0, -21.0]})
    assert not per_target.is_per_target({"type": "TrackingOnly"})
    assert not per_target.is_per_target("TrackingOnly")
    with pytest.raises(ValueError, match="per_target"):
        per_target.is_per_target({"image_rot_zx": "per-target"})
    config = load_config(TWO_DEVICES, [SWITCH])
    assert config["env"]["scene"]["fluoroscopy"] == {"type": "TrackingOnly", "image_frequency": 7.5,
                                                     "image_rot_zx": "per_target"}
    # the round-two config has the switch on (2026-11-09); round one keeps its fixed angle
    assert load_config(ICA)["env"]["scene"]["fluoroscopy"] == config["env"]["scene"]["fluoroscopy"]
    assert load_config(TWO_DEVICES)["env"]["scene"]["fluoroscopy"]["image_rot_zx"] == [61.0, -21.0]


def test_table_rows_for_branches():
    table = _table("rebuilt_cut")
    for target in ca.TARGETS:
        assert per_target.table_key(table, target) == target
        assert per_target.table_key(table, f"{target}_rest" if target != "bct" else target) == target
    for off in ("aorta", "aorta_inlet", "v7", "co"):
        with pytest.raises(ValueError, match="C-arm angle table has angles for"):
            per_target.table_key(table, off)
    assert per_target.initial_angle(table) == table["angles"]["all"]["image_rot_zx"]


def test_missing_or_stale_tables_are_errors_that_say_what_to_run(tmp_path, monkeypatch):
    phantom = ca.DATA["rebuilt_cut"].parent
    found = per_target.table_for(SimpleNamespace(spec_path=ca.DATA["rebuilt_cut"]))
    assert found == _table("rebuilt_cut")
    assert per_target.table_for(SimpleNamespace(spec_path=ca.DATA["all_vessels"])) == _table("all_vessels")
    with pytest.raises(FileNotFoundError, match="choose_angle --centerline original --write-table"):
        per_target.table_for(SimpleNamespace(spec_path=ca.DATA["original"]))      # the chooser has it
    with pytest.raises(FileNotFoundError, match="no input for that file"):
        per_target.table_for(SimpleNamespace(spec_path=phantom / "lab_cut.json"))  # it does not
    with pytest.raises(ValueError, match="BiroPhantomArch"):
        per_target.table_for(SimpleNamespace())                                   # stEVE's AorticArch
    stale = _table("rebuilt_cut")
    stale["centerline"]["sha256"] = "0" * 64
    (tmp_path / "angles_rebuilt_cut.json").write_text(json.dumps(stale))
    monkeypatch.setattr(per_target, "TABLE_DIR", tmp_path)
    with pytest.raises(ValueError, match="write it again: .*--centerline rebuilt_cut --write-table"):
        per_target.table_for(SimpleNamespace(spec_path=ca.DATA["rebuilt_cut"]))


def test_phantom_vessel_trees_name_their_centerline_file():
    pytest.importorskip("eve")
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    assert BiroPhantomArch(version="rebuilt_cut").spec_path == ca.DATA["rebuilt_cut"]
    assert BiroPhantomArch(version="lab").spec_path == ca.DATA["all_vessels"]
    assert BiroPhantomArch(centerline="all_vessels").spec_path == ca.DATA["all_vessels"]
    assert BiroPhantomArch().spec_path == ca.DATA["original"]


def test_vessel_geometry_tangents_follow_the_angle():
    from guidewire_rl.env.observations.vessel_geometry import GeometryCore
    branch = SimpleNamespace(name="a", coordinates=np.array([[0, 0, 0], [10, 0, 0], [10, 10, 5]], float),
                             radii=np.array([2.0, 2.0, 2.0]))
    fluoroscopy = SimpleNamespace(image_rot_zx=[20.0, 5.0], image_center=[0, 0, 0], field_of_view=None)
    intervention = SimpleNamespace(vessel_tree=SimpleNamespace(branches=[branch]), fluoroscopy=fluoroscopy)
    core = GeometryCore(frame="vessel")
    core._cache_vessel(intervention)
    before = core._tangent_2d.copy()
    fluoroscopy.image_rot_zx = [61.0, -21.0]
    core._cache_vessel(intervention)
    fresh = GeometryCore(frame="vessel")
    fresh._cache_vessel(intervention)
    np.testing.assert_array_equal(core._tangent_2d, fresh._tangent_2d)
    assert not np.allclose(core._tangent_2d, before)


# ------------------------------------------------------------- scene only ----
def _scene(config_path, overrides):
    from guidewire_rl.env import builder
    config = load_config(config_path, overrides)
    deps = builder._build_scene(config["env"]["scene"])
    deps["vessel_tree"].reset(0, None)
    return config, deps


def _same_target_at_fixed_angle(config, deps, angle, seed):
    """The episode's target as a C-arm fixed at ``angle`` puts it (a fresh target, same seed)."""
    from guidewire_rl.env import registry
    scene = config["env"]["scene"]
    fixed = registry.instantiate("fluoroscopy", {**scene["fluoroscopy"], "image_rot_zx": angle}, deps)
    target = registry.instantiate("target", scene["target"], {**deps, "fluoroscopy": fixed})
    target.reset(0, seed)
    return target


@pytest.mark.upstream
def test_angle_follows_the_target_branch_from_episode_to_episode():
    """phantom_all_vessels.yaml (G13's target branches on the lab phantom, CenterlineRandom) with the
    switch: every episode gets its target branch's angle from angles_all_vessels.json, and the target is
    where a C-arm fixed at that angle puts it."""
    from eve.util.coordtransform import tracking3d_to_vessel_cs
    config, deps = _scene(ALL_VESSELS, [SWITCH])
    vessel_tree, fluoroscopy, target, c_arm = (deps[k] for k in ("vessel_tree", "fluoroscopy", "target", "c_arm"))
    table = _table("all_vessels")
    assert c_arm.table == table
    assert fluoroscopy.image_rot_zx == table["angles"]["all"]["image_rot_zx"]     # until the first reset
    keys = []
    for seed in range(12):
        target.reset(0, seed)
        c_arm.reset()
        angle = table["angles"][c_arm.key]["image_rot_zx"]
        assert fluoroscopy.image_rot_zx == angle
        assert c_arm.key == c_arm.branch            # G13's targets lie on the five branches themselves
        point = tracking3d_to_vessel_cs(target.coordinates3d, angle, fluoroscopy.image_center)
        assert np.min(np.linalg.norm(vessel_tree[c_arm.branch].coordinates - point, axis=1)) < 1e-3
        again = _same_target_at_fixed_angle(config, deps, angle, seed)
        np.testing.assert_array_equal(target.coordinates3d, again.coordinates3d)
        np.testing.assert_array_equal(target.coordinates2d, again.coordinates2d)
        keys.append(c_arm.key)
    assert set(keys) == set(ca.TARGETS)
    assert sum(a != b for a, b in zip(keys, keys[1:])) >= 5        # the angle changes between episodes
    # unseeded resets go on drawing; the angle still follows
    for _ in range(5):
        target.reset(0, None)
        c_arm.reset()
        assert fluoroscopy.image_rot_zx == table["angles"][c_arm.key]["image_rot_zx"]


@pytest.mark.upstream
def test_two_device_target_gets_the_right_carotids_angle_on_both_pieces():
    """RouteWindowTarget (35-55 mm into the right carotid) lies on ``rcca`` or ``rcca_rest``; both take
    the rcca angle of angles_rebuilt_cut.json, [52.0, -12.2] (its whole route: aorta, trunk, carotid)."""
    from guidewire_rl.env.two_device.route import Route
    config, deps = _scene(TWO_DEVICES, [SWITCH, "env.scene.target.split=all"])
    fluoroscopy, target, c_arm = deps["fluoroscopy"], deps["target"], deps["c_arm"]
    route = Route.of(deps["vessel_tree"])
    angle = _table("rebuilt_cut")["angles"]["rcca"]["image_rot_zx"]
    assert np.round(angle, 1).tolist() == [52.0, -12.2]
    branches = set()
    for seed in range(12):
        target.reset(0, seed)
        c_arm.reset()
        assert c_arm.key == "rcca" and fluoroscopy.image_rot_zx == angle
        branches.add(c_arm.branch)
        on_rest = route.s_rcca + target.depth_mm >= route.offsets["rcca_rest"]   # rcca: 37.7 mm
        assert c_arm.branch == target.branch == ("rcca_rest" if on_rest else "rcca")
        again = _same_target_at_fixed_angle(config, deps, angle, seed)
        np.testing.assert_array_equal(target.coordinates3d, again.coordinates3d)
    assert branches == {"rcca", "rcca_rest"}


@pytest.mark.upstream
def test_a_config_without_the_switch_builds_its_scene_as_before():
    _, deps = _scene(TWO_DEVICES, [])
    assert "c_arm" not in deps
    assert deps["fluoroscopy"].image_rot_zx == [61.0, -21.0]
    deps["target"].reset(0, 3)
    assert deps["fluoroscopy"].image_rot_zx == [61.0, -21.0]


# --------------------------------------------------------------------- SOFA ----
@pytest.mark.sofa
@pytest.mark.upstream
def test_two_device_env_sees_everything_at_the_episodes_angle():
    """The two-device env with the switch against the same env with the right carotid's angle written
    in: the same observations (tips, shapes, target, junction map: all projected), rewards and task box,
    reset and step by step. The config without the switch keeps stEVE's intervention and its angle.

    The target and the junction map (no physics in them) bit for bit; the rest to 1e-6: SOFA scenes in
    one process drift apart by ~1e-11 in the reward, later in the observations' last float32 digit,
    whatever the angle (measured 2026-11-03: a third env with the angle written in, built after these
    two, drifts from the first by the same 3.8e-11 as the switched one; in the full test run, after
    other SOFA tests, already in the first episode). A wrong angle moves the normalized observations
    by ~0.1."""
    from guidewire_rl.env import build_env
    from guidewire_rl.env.two_device.route import Route, task_box_2d
    angle = _table("rebuilt_cut")["angles"]["rcca"]["image_rot_zx"]
    switched = build_env(load_config(TWO_DEVICES, [SWITCH])["env"])
    fixed = build_env(load_config(TWO_DEVICES, [_fixed(angle)])["env"])
    try:
        assert type(fixed.intervention).__name__ == "MonoPlaneStatic"
        assert switched.intervention.c_arm.table["centerline"]["file"] == "rebuilt_cut.json"
        actions = [[0.8, 0.2, 0.0, 0.0]] * 3 + [[0.6, -0.3, 0.6, 0.4]] * 3
        start = dict(zip([n for n, _ in switched.observation_layout],
                         np.cumsum([0] + [k for _, k in switched.observation_layout])))
        no_physics = np.concatenate([np.arange(start[n], start[n] + k) for n, k in switched.observation_layout
                                     if n in ("target", "junctions")])
        assert len(no_physics) == 2 + 8
        for seed in (4, 11):
            obs_s, _ = switched.reset(seed=seed)
            obs_f, _ = fixed.reset(seed=seed)
            fluoroscopy = switched.intervention.fluoroscopy
            assert fluoroscopy.image_rot_zx == angle and switched.intervention.c_arm.key == "rcca"
            assert fixed.intervention.fluoroscopy.image_rot_zx == angle
            np.testing.assert_array_equal(obs_s[no_physics], obs_f[no_physics])
            np.testing.assert_allclose(obs_s, obs_f, atol=1e-6)
            box = task_box_2d(Route.of(switched.intervention.vessel_tree), fluoroscopy)
            np.testing.assert_array_equal(switched.term("target").box.get(), box)
            np.testing.assert_array_equal(switched.term("wire_tip").box.get(), box)
            at_all_5 = task_box_2d(Route.of(switched.intervention.vessel_tree),
                                   SimpleNamespace(image_rot_zx=[61.0, -21.0], image_center=[0, 0, 0]))
            assert not np.allclose(box, at_all_5)
            for action in actions:
                step_s = switched.step(np.asarray(action, np.float32))
                step_f = fixed.step(np.asarray(action, np.float32))
                np.testing.assert_allclose(step_s[0], step_f[0], atol=1e-6)
                assert step_s[1] == pytest.approx(step_f[1], abs=1e-6)
    finally:
        switched.close()
        fixed.close()


@pytest.mark.sofa
@pytest.mark.upstream
def test_single_device_phantom_config_runs_with_the_switch():
    """phantom_all_vessels.yaml (stEVE's Tracking2D / Target2D, which read the fluoroscopy's tracking
    space) with the switch: builds (stEVE's two-corner box is no box at most of these angles; problem 21),
    and over a few episodes each one sees its target branch's angle, with valid observation spaces and
    observations inside the clip."""
    from guidewire_rl.env import build_env
    table = _table("all_vessels")
    env = build_env(load_config(ALL_VESSELS, [SWITCH])["env"])
    try:
        assert type(env.intervention.fluoroscopy).__module__ == "simulator_modifications.fluoroscopy"
        keys = []
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            c_arm = env.intervention.c_arm
            assert env.intervention.fluoroscopy.image_rot_zx == table["angles"][c_arm.key]["image_rot_zx"]
            for name in ("position", "target"):
                space = env.term(name).wrapped_obs.space
                assert np.all(space.low <= space.high)
            for action in ([0.9, 0.1], [0.8, -0.3], [1.0, 0.5]):
                obs, *_ = env.step(np.asarray(action, np.float32))
                assert np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= 1.0)
            keys.append(c_arm.key)
        assert len(set(keys)) >= 2            # the angle changed between episodes
    finally:
        env.close()
