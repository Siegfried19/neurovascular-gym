"""Round two of the two-device task, the env side, without SOFA (todo_list/two_devices_round2_changes.md).

The route's junction list and the ICA/ECA split, the target window counted from the split, the
single reward formula and its materials, the rule-free action map, the full junction map, the first-
tier flag, the catheter's goal F in the image, the whole-tree box, and the round-two config.
"""

from __future__ import annotations

from types import SimpleNamespace

from functools import partial

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config

ROUND2 = WORKSPACE_ROOT / "configs" / "two_devices_ica.yaml"
ANGLE = SimpleNamespace(image_rot_zx=[52.03, -12.25], image_center=[0.0, 0.0, 0.0], field_of_view=None)


@pytest.fixture(scope="module")
def tree():
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    return BiroPhantomArch(version="rebuilt_cut")


@pytest.fixture(scope="module")
def route(tree):
    from guidewire_rl.env.two_device.route import Route
    return Route(tree)


# ------------------------------------------------------------------------------- route ----
def test_route_lists_every_junction_in_order_and_knows_the_split(route):
    names = [n for n, _, _, _ in route.all_junctions]
    # the six of the route (user decision 2026-11-08), then the three past the target zone
    assert names[:6] == ["lsa", "lcca", "bct", "rcca", "v7", "v8"]
    assert names[6:] == ["v9", "v10", "v11"]
    s = [s for _, _, _, s in route.all_junctions]
    assert s == sorted(s) and s[0] > route.s_insertion
    assert s[2] == pytest.approx(route.s_O, abs=1e-6)                 # the trunk leaves the aorta at O
    assert s[3] == pytest.approx(route.s_rcca, abs=1e-6)              # the trunk's end
    split = route.s_of_fork("v8")
    assert split == pytest.approx(route.s_rcca + 90.9, abs=0.3)       # 90.9 mm past the carotid's opening
    assert route.s_of_fork("v7") == pytest.approx(route.s_rcca + 63.8, abs=0.5)
    for _, mouth, direction, _ in route.all_junctions:
        assert mouth.shape == (3,) and np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(KeyError):
        route.s_of_fork("v39")                                        # not off a route vessel


def test_depth_past_the_split_and_F_from_the_split(route):
    split = route.s_of_fork("v8")
    route.set_target(route.point_at(split + 40.0), catheter_goal_s=split + 5.0)
    assert route.s_F == pytest.approx(split + 5.0)
    np.testing.assert_allclose(route.F, route.point_at(split + 5.0))
    on = route.locate(route.point_at(split + 12.0))
    assert route.depth_past(split, on) == pytest.approx(12.0, abs=1e-6)
    before = route.locate(route.point_at(split - 30.0))
    assert route.depth_past(split, before) == pytest.approx(-30.0, abs=1e-6)
    eca = route.locate(route.graph.point("v8", 6.0))                   # in the external carotid
    assert eca.branch == "v8" and route.depth_past(split, eca) == pytest.approx(-6.0, abs=1e-6)
    assert route.path_to_F(None, on) == 0.0                           # past F: done
    assert route.path_to_F(None, before) == pytest.approx(35.0, abs=1e-6)
    assert route.path_to_F(None, eca) == pytest.approx(11.0, abs=1e-6)  # back to the split and 5 on


def test_the_external_carotid_is_a_wrong_vessel_and_no_ica_tip_reads_as_one(route):
    """User 2026-11-11: past the split, its other branch counts as a wrong vessel (8 mm in, like the
    arch's). The ICA and ECA lumens overlap just past the split; no point of the ICA lumen (out to
    95 % of its radius, all around) may read 8 mm into the ECA, or an ICA tip would be charged."""
    from guidewire_rl.env.two_device.phases import ENTER_MM, WRONG_VESSELS, PhaseDetector
    from guidewire_rl.env.two_device.route import unit

    assert PhaseDetector(None).wrong_vessels == WRONG_VESSELS + ("v8",)
    assert PhaseDetector(None, split_branch=None, catheter_goal_depth_mm=96.0).wrong_vessels == WRONG_VESSELS
    for arc in (9.0, 12.0, 20.0):                                    # really in the ECA: counted
        assert route.depth(None, "v8", route.locate(route.graph.point("v8", arc))) >= ENTER_MM
    split, deepest = route.s_of_fork("v8"), -np.inf
    for ds in np.arange(0.0, 60.1, 1.0):
        centre = route.point_at(split + ds)
        tangent = unit(route.point_at(split + ds + 0.5) - route.point_at(split + ds - 0.5))
        a = unit(np.cross(tangent, [1.0, 0.0, 0.0]))
        b = np.cross(tangent, a)
        here = route.locate(centre)
        radius = route.graph.radius(here.branch, here.arc)
        for frac in (0.0, 0.5, 0.8, 0.95):
            for angle in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
                p = centre + frac * radius * (np.cos(angle) * a + np.sin(angle) * b)
                deepest = max(deepest, route.depth(None, "v8", route.locate(p)))
    assert deepest < ENTER_MM                                        # measured 6.7 mm, 5 mm past the split


def test_whole_tree_box_holds_the_task_box(route):
    from guidewire_rl.env.two_device.route import task_box_2d
    task_low, task_high = task_box_2d(route, ANGLE, "task")
    tree_low, tree_high = task_box_2d(route, ANGLE, "tree")
    assert np.all(tree_low <= task_low + 1e-9) and np.all(tree_high >= task_high - 1e-9)
    assert np.any(tree_high - tree_low > task_high - task_low)
    with pytest.raises(ValueError):
        task_box_2d(route, ANGLE, "everything")


def test_target_window_counts_from_the_split(tree):
    from guidewire_rl.env.two_device.route import Route
    from guidewire_rl.env.two_device.targets import RouteWindowTarget
    target = RouteWindowTarget(tree, ANGLE, split="train", past_fork="v8", depth_min_mm=35.0, depth_max_mm=45.0)
    assert target.candidate_depths == [35.0, 37.0, 39.0, 41.0, 43.0, 45.0]
    route = Route.of(tree)
    split = route.s_of_fork("v8")
    for point, depth in zip(target.candidates(), target.candidate_depths):
        assert route.depth_past(split, route.locate(point)) == pytest.approx(depth, abs=1e-4)
    target.reset(seed=3)
    assert target.depth_mm in target.candidate_depths and target.branch == "rcca_rest"
    legacy = RouteWindowTarget(tree, ANGLE, split="eval")                 # the first round's window
    assert legacy.candidate_depths == [36.0 + 2 * i for i in range(10)]
    assert legacy.reference_s(route) == pytest.approx(route.s_rcca)


# ----------------------------------------------------------------------------- formula ----
def test_formula_is_the_users_numbers():
    from guidewire_rl.env.two_device import reward_formula as rf
    c = rf.Coefficients()
    assert (c.progress, c.follow, c.step_penalty, c.wrong_vessel, c.tier1, c.tier2, c.failure) == \
        (0.01, 0.01, 0.002, 0.1, 1.0, 1.0, 1.0)
    assert (c.force_cap, c.cath_knee_N, c.cath_width_N, c.wire_knee_N, c.wire_width_N) == (0.1, 2.0, 3.0, 0.5, 1.0)

    def m(**fields):
        v = np.zeros(len(rf.FIELDS))
        for k, x in fields.items():
            v[rf.INDEX[k]] = x
        return v

    assert rf.compute(m(), c) == pytest.approx(-0.002)
    assert rf.compute(m(progress_mm=5.0), c) == pytest.approx(0.05 - 0.002)
    assert rf.compute(m(follow_mm=-3.0), c) == pytest.approx(-0.03 - 0.002)
    # the force penalty: 0 at the knee, (excess / width)^2 x cap, capped
    assert rf.compute(m(cath_force_N=2.0), c) == pytest.approx(-0.002)
    assert rf.compute(m(cath_force_N=3.5), c) == pytest.approx(-0.002 - 0.1 * 0.25)
    assert rf.compute(m(cath_force_N=4.0), c) == pytest.approx(-0.002 - 0.1 * (2 / 3) ** 2)
    assert rf.compute(m(cath_force_N=30.0), c) == pytest.approx(-0.002 - 0.1)
    assert rf.compute(m(wire_force_N=1.0), c) == pytest.approx(-0.002 - 0.1 * 0.25)
    assert rf.compute(m(wire_force_N=1.5), c) == pytest.approx(-0.002 - 0.1)
    assert rf.compute(m(wrong_vessel=1.0), c) == pytest.approx(-0.102)
    assert rf.compute(m(tier1=1.0), c) == pytest.approx(0.998)
    assert rf.compute(m(success=1.0), c) == pytest.approx(0.998)
    assert rf.compute(m(hard_failure=1.0), c) == pytest.approx(-1.002)
    assert rf.compute(m(sim_error=1.0, progress_mm=50.0), c) == 0.0        # a dropped step pays nothing
    # batches: any leading shape
    batch = np.stack([m(progress_mm=1.0), m(progress_mm=2.0)])
    np.testing.assert_allclose(rf.compute(batch, c), [0.008, 0.018])
    # the reserves
    reserve = rf.Coefficients(arch_ruler=True)
    assert rf.compute(m(progress_mm=5.0, hook_progress_mm=1.0), reserve) == pytest.approx(0.01 - 0.002)
    soft = rf.Coefficients(lead_soft_min_mm=30.0, lead_soft_factor=0.8)
    assert rf.compute(m(success=1.0, lead_mm=20.0), soft) == pytest.approx(0.8 - 0.002)
    assert rf.compute(m(success=1.0, lead_mm=35.0), soft) == pytest.approx(1.0 - 0.002)
    with pytest.raises(KeyError):
        rf.Coefficients.from_config({"shield": 0.05})
    assert rf.Coefficients.from_config({"type": "TwoDeviceFormula", "progress_ruler": "nearest",
                                        "progress": 0.02}).progress == 0.02


def made_state(**fields):
    from guidewire_rl.env.two_device.phases import PhaseState
    base = dict(d=20.0, L_wire=100.0, L_cath=80.0, depth_bct_c=-50.0, depth_rcca_c=-90.0,
                depth_rcca_w=-70.0, path_to_target_w=300.0, path_to_target_c=320.0, path_to_F_c=280.0,
                to_hook_w=10.0, to_hook_c=30.0, hook_to_target=200.0, hook_to_F=150.0)
    base.update(fields)
    return PhaseState(**base)


def test_formula_part_builds_the_materials_from_the_referee_and_the_labels():
    from guidewire_rl.env.two_device import reward_formula as rf
    referee = SimpleNamespace(state=made_state())
    part = rf.TwoDeviceFormula(referee, progress_ruler="nearest")
    part.reset()
    referee.state = made_state(path_to_target_w=290.0, path_to_target_c=325.0, path_to_F_c=276.0,
                               to_hook_w=4.0, events={"wrong_vessel": 1, "dislodged": 0, "tier1": 0})
    reward, m = part.step({"distal_force": 0.3, "catheter_tip_force": 3.5})
    assert m[rf.INDEX["progress_mm"]] == pytest.approx(10.0)         # the nearer tip (the wire) got 10 mm closer
    assert m[rf.INDEX["follow_mm"]] == pytest.approx(4.0)
    assert m[rf.INDEX["hook_progress_mm"]] == pytest.approx(6.0)     # the reserve's materials ride along
    assert m[rf.INDEX["wire_force_N"]] == pytest.approx(0.3) and m[rf.INDEX["cath_force_N"]] == pytest.approx(3.5)
    assert m[rf.INDEX["wrong_vessel"]] == 1.0 and m[rf.INDEX["lead_mm"]] == 20.0
    assert reward == pytest.approx(rf.compute(m, part.coefficients))
    assert reward == pytest.approx(0.1 + 0.04 - 0.1 * 0.25 - 0.002 - 0.1)
    # the catheter takes the lead: min() is continuous, the sum telescopes
    referee.state = made_state(path_to_target_w=290.0, path_to_target_c=285.0, path_to_F_c=276.0)
    _, m = part.step({})
    assert m[rf.INDEX["progress_mm"]] == pytest.approx(5.0)
    # the other ruler: the wire alone once its tip is in the trunk
    wire_ruler = rf.TwoDeviceFormula(SimpleNamespace(state=made_state()), progress_ruler="wire_after_trunk")
    wire_ruler.reset()
    wire_ruler.phase_detector.state = made_state(path_to_target_w=295.0, path_to_target_c=200.0, depth_bct_w=5.0)
    _, m = wire_ruler.step({})
    assert m[rf.INDEX["progress_mm"]] == pytest.approx(5.0)          # not the catheter's 120
    # a simulation-error step: nothing moves, nothing is paid
    referee.state = made_state(path_to_target_w=900.0, sim_error=True)
    reward, m = part.step({"distal_force": 9.0})
    assert reward == 0.0 and m[rf.INDEX["sim_error"]] == 1.0 and m[rf.INDEX["wire_force_N"]] == 0.0
    referee.state = made_state(path_to_target_w=289.0, path_to_target_c=285.0, path_to_F_c=276.0)
    _, m = part.step({})
    assert m[rf.INDEX["progress_mm"]] == pytest.approx(0.0)          # 285 both before and after the error
    with pytest.raises(ValueError):
        rf.TwoDeviceFormula(referee, progress_ruler="catheter")


def test_best_mode_pays_only_new_progress_and_old_materials_derive_the_same():
    """User 2026-11-12: only new progress pays (progress_mode 'best'): backing off, overshooting a
    mouth and coming back, or pulling the wire into the catheter cost nothing; the episode's progress
    terms add up to at most the whole path; the live materials and the ones derived for an episode
    recorded before (upgrade_materials) agree."""
    from guidewire_rl.env.two_device import reward_formula as rf
    best = rf.Coefficients(progress_mode="best")
    referee = SimpleNamespace(state=made_state(path_to_target_w=300.0, path_to_target_c=320.0, path_to_F_c=280.0))
    part = rf.TwoDeviceFormula(referee, progress_ruler="nearest", progress_mode="best")
    part.reset()
    # the nearer tip's path to the target, and the catheter's path to F, step by step: forward, back
    # (overshoot / pull the wire in), forward again past the old best, a simulation error, on
    paths = [(290.0, 270.0), (285.0, 268.0), (300.0, 285.0), (295.0, 280.0), (280.0, 266.0), (280.0, 266.0),
             (270.0, 260.0)]
    rows, rewards = [], []
    for k, (to_target, to_F) in enumerate(paths):
        referee.state = made_state(path_to_target_w=to_target, path_to_target_c=to_target + 30.0, path_to_F_c=to_F,
                                   sim_error=(k == 5))
        reward, m = part.step({})
        rows.append(m.copy()); rewards.append(reward)
    rows = np.asarray(rows)
    assert rows[:, rf.INDEX["progress_new_mm"]] == pytest.approx([10.0, 5.0, 0.0, 0.0, 5.0, 0.0, 10.0])
    assert rows[:, rf.INDEX["follow_new_mm"]] == pytest.approx([10.0, 2.0, 0.0, 0.0, 2.0, 0.0, 6.0])
    assert rows[:, rf.INDEX["progress_mm"]].sum() == pytest.approx(30.0)        # the plain terms still telescope
    assert min(rewards) >= -best.step_penalty - 1e-9                             # going back costs only the step
    assert rewards[2] == pytest.approx(-best.step_penalty) and rewards[5] == 0.0  # back off; the error step pays 0
    assert rows[:, rf.INDEX["progress_new_mm"]].sum() == pytest.approx(300.0 - 270.0)   # bounded by the way covered
    # an episode recorded with the round-two materials (12 fields) derives the same new progress
    old = rows[:, :rf.ROUND_TWO_FIELDS]
    still = np.zeros((len(rows) + 1, 4))                      # no executed action: action_change 0, as stepped
    upgraded = rf.upgrade_materials(old, observations=still, action_slice=(0, 4))
    np.testing.assert_allclose(upgraded, rows, atol=1e-5)
    assert rf.upgrade_materials(rows) is not None and rf.upgrade_materials(rows).shape == rows.shape
    np.testing.assert_allclose(rf.compute(upgraded, best), rewards, atol=1e-6)
    # the default is unchanged: potential differences, back-off charged
    plain = rf.compute(rows, rf.Coefficients())
    assert plain[2] == pytest.approx(-0.15 - 0.17 - 0.002)                # 285 -> 300 and 268 -> 285
    with pytest.raises(ValueError):
        rf.Coefficients(progress_mode="best", arch_ruler=True)
    with pytest.raises(ValueError):
        rf.Coefficients(progress_mode="newest")
    with pytest.raises(ValueError):
        rf.upgrade_materials(np.zeros((3, 5)))


def test_replay_upgrades_old_materials_and_recomputes_with_the_new_mode():
    from guidewire_rl.env.two_device import reward_formula as rf
    from guidewire_rl.train.replay import EpisodeReplay
    replay = EpisodeReplay(capacity=100, observation_dim=3, action_dim=2, horizon=2, batch_size=4, label_names=[])
    T = 6
    old = np.zeros((T, rf.ROUND_TWO_FIELDS), np.float32)
    old[:, rf.INDEX["progress_mm"]] = [5.0, -3.0, -2.0, 4.0, 3.0, 1.0]       # back 5, then past the old best by 3
    replay.add(np.zeros((T + 1, 3)), np.zeros((T, 2)), np.zeros(T), np.zeros(T, bool), materials=old)
    best = rf.Coefficients(progress_mode="best")
    replay.configure(reward_fn=lambda m: rf.compute(m, best), n_materials=len(rf.FIELDS),
                     materials_upgrade=partial(rf.upgrade_materials, action_slice=(0, 2)))
    m = replay.episodes[0]["materials"]
    assert m.shape == (T, len(rf.FIELDS))
    assert m[:, rf.INDEX["progress_new_mm"]] == pytest.approx([5.0, 0.0, 0.0, 0.0, 2.0, 1.0])
    assert replay.reward_fn(m) == pytest.approx(np.array([5.0, 0.0, 0.0, 0.0, 2.0, 1.0]) * 0.01 - 0.002)
    # a new episode recorded with the old field count is upgraded on the way in, too
    replay.add(np.zeros((T + 1, 3)), np.zeros((T, 2)), np.zeros(T), np.zeros(T, bool), materials=old)
    assert replay.episodes[1]["materials"].shape == (T, len(rf.FIELDS))


def test_action_change_is_the_smoothness_penalty_on_the_executed_action():
    """User 2026-11-13 ("那个抗抖动的你加一下"): 0.005 x the mean over the dimensions of the squared
    change of the executed action, as the observation's last_actions shows it; full-speed back and
    forth costs 0.02 a step, steady motion nothing; off (0) by default."""
    from guidewire_rl.env.two_device import reward_formula as rf
    m = np.zeros((3, len(rf.FIELDS)))
    m[:, rf.INDEX["action_change"]] = [4.0, 0.0, 0.25]
    smooth = rf.Coefficients(action_change=0.005)
    assert rf.compute(m, smooth) == pytest.approx(np.array([-0.02, 0.0, -0.00125]) - 0.002)
    assert rf.compute(m, rf.Coefficients()) == pytest.approx([-0.002] * 3)        # the default leaves it out
    parts = rf.components(m, smooth)
    assert parts["smooth"] == pytest.approx([-0.02, 0.0, -0.00125]) and "smooth" in rf.COMPONENTS
    np.testing.assert_allclose(sum(parts.values()), rf.compute(m, smooth))
    m[1, rf.INDEX["sim_error"]] = 1.0
    m[1, rf.INDEX["action_change"]] = 4.0
    assert rf.compute(m, smooth)[1] == 0.0                                         # an error step pays nothing
    assert rf.action_change_of([1, -1, 0.5, 0], [-1, 1, 0.5, 0]) == pytest.approx(2.0)
    # the formula part stores what the env hands it
    referee = SimpleNamespace(state=made_state(path_to_target_w=300.0, path_to_target_c=320.0, path_to_F_c=280.0))
    part = rf.TwoDeviceFormula(referee, action_change=0.005)
    part.reset()
    reward, mat = part.step({}, action_change=4.0)
    assert mat[rf.INDEX["action_change"]] == pytest.approx(4.0) and reward == pytest.approx(-0.022)
    assert part.action_term == "last_actions"
    assert rf.Coefficients.from_config({"type": "TwoDeviceFormula", "action_term": "last_actions",
                                        "action_change": 0.005}).action_change == 0.005


def test_action_change_of_old_episodes_comes_from_their_observations():
    from guidewire_rl.env.two_device import reward_formula as rf
    names = ["a[0]", "a[1]"] + [f"last_actions[{i}]" for i in range(12)] + ["tier1[0]"]
    assert rf.action_slice_from_names(names, 4) == (2, 6)                  # the newest action comes first
    assert rf.action_slice_from_names(["a[0]"], 4) is None
    with pytest.raises(ValueError):
        rf.action_slice_from_names(["last_actions[0]", "x[0]", "last_actions[1]"], 2)
    T = 4
    obs = np.zeros((T + 1, 15))
    newest = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-0.5, -1, -1, -1]])
    obs[:, 2:6] = newest
    obs[:, 6:10] = 9.0                                                     # older entries are not read
    old14 = np.zeros((T, rf.PROGRESS_NEW_FIELDS), np.float32)
    up = rf.upgrade_materials(old14, observations=obs, action_slice=(2, 6))
    assert up.shape == (T, len(rf.FIELDS))
    assert up[:, rf.INDEX["action_change"]] == pytest.approx([1.0, 4.0, 0.0, 0.0625])
    old12 = np.zeros((T, rf.ROUND_TWO_FIELDS), np.float32)
    assert rf.upgrade_materials(old12, observations=obs, action_slice=(2, 6)).shape == (T, len(rf.FIELDS))
    with pytest.raises(ValueError):
        rf.upgrade_materials(old14)                                        # no observations: refuse, don't guess
    with pytest.raises(ValueError):
        rf.upgrade_materials(old14, observations=obs[:-1], action_slice=(2, 6))


# -------------------------------------------------------------------------- action map ----
HIGH = np.array([40.0, np.pi, 25.0, np.pi / 2])


def test_limits_action_map_has_no_rules_but_the_limits():
    from guidewire_rl.env.two_device.actions import ShieldInfo, TwoDeviceLimits
    intervention = SimpleNamespace(fluoroscopy=SimpleNamespace(image_frequency=7.5),
                                   instrument_lengths_inserted=[100.0, 60.0],
                                   instrument_lengths_maximum=[450.0, 450.0],
                                   velocity_limits=HIGH.reshape(2, 2), last_action=None)
    shell = TwoDeviceLimits(intervention, phase_detector=SimpleNamespace(state=made_state(branch_c="bct", d=-30.0)))
    raw = shell.to_raw(np.array([1.0, -1.0, 1.0, 0.5], np.float32), -HIGH, HIGH)
    np.testing.assert_allclose(raw, [40.0, -np.pi, 25.0, np.pi / 4], rtol=1e-6)   # a bare fast catheter push goes through
    assert shell.last.stages == [] and shell.last.penalized == 0.0
    # the insertion limits: no pull below 0, no push past the length
    intervention.instrument_lengths_inserted = [0.3, 449.9]
    raw = shell.to_raw(np.array([-1.0, 0.0, 1.0, 0.0], np.float32), -HIGH, HIGH)
    np.testing.assert_allclose(raw, [0.0, 0.0, 0.0, 0.0])
    assert shell.last.stages == ["device_limit"]
    intervention.last_action = raw.reshape(2, 2)
    info = ShieldInfo(intervention, shell).info
    assert set(info["shield_causes"]) == {"mode", "speed", "device_limit"} and info["shield_causes"]["device_limit"] == 1
    assert info["shield_penalized"] == 0.0 and not info["nominal_available"]
    # a robot-mode layer: one device at a time (a stand-in rule)
    layered = TwoDeviceLimits(intervention, mode_map={"type": "test_two_device_round2.OneAtATime"})
    intervention.instrument_lengths_inserted = [100.0, 60.0]
    raw = layered.to_raw(np.array([0.5, 0.0, 0.8, 0.0], np.float32), -HIGH, HIGH)
    np.testing.assert_allclose(raw, [0.0, 0.0, 20.0, 0.0], atol=1e-6)
    assert layered.last.stages == ["mode"]
    with pytest.raises(ValueError):
        shell.to_raw(np.array([np.nan, 0, 0, 0], np.float32), -HIGH, HIGH)


class OneAtATime:
    """A stand-in robot-mode rule: only the device asked to move more moves."""

    def apply(self, normalized, state):
        out = np.asarray(normalized, float).copy()
        if abs(out[2]) >= abs(out[0]):
            out[0] = out[1] = 0.0
        else:
            out[2] = out[3] = 0.0
        return out


def test_guarded_shell_reports_bare_advance_only_to_a_referee_that_counts_it():
    from guidewire_rl.env.two_device.actions import TwoDeviceGuarded
    from guidewire_rl.env.two_device.phases import PhaseDetector
    referee = PhaseDetector(intervention=None)
    referee.state = made_state(branch_c="bct", d=-20.0, L_wire=180.0, L_cath=200.0)
    labels = [SimpleNamespace(name="distal_force", last_value=0.0),
              SimpleNamespace(name="catheter_tip_force", last_value=0.0)]
    shell = TwoDeviceGuarded(SimpleNamespace(fluoroscopy=SimpleNamespace(image_frequency=7.5)), referee, labels=labels)
    shell.to_raw(np.array([0.0, 0.0, 1.0, 0.0], np.float32), -HIGH, HIGH)
    assert "hard_rule" in shell.last.stages and "bare_advance" not in referee.pending_events


# ------------------------------------------------------------------------- observations ----
def fake_scene(tree, route):
    rng = np.random.default_rng(0)
    intervention = SimpleNamespace(vessel_tree=tree, fluoroscopy=ANGLE, _np_random=rng, simulation=None)
    referee = SimpleNamespace(route_chain=route.chain, route=route, state=made_state())
    return intervention, referee


def test_junction_map_all_mode_gives_the_six_every_step(tree, route):
    from guidewire_rl.env.two_device.observations import JunctionMap2D, TaskBox2D
    from guidewire_rl.env.two_device.route import to_tracking2d
    split = route.s_of_fork("v8")
    route.set_target(route.point_at(split + 40.0), catheter_goal_s=split + 5.0)
    intervention, referee = fake_scene(tree, route)
    box = TaskBox2D(intervention, region="tree")
    term = JunctionMap2D(intervention, referee, n_junctions=6, mode="all", offset_mm=3.0, grid_mm=2.0, task_box=box)
    term.reset()
    assert term.obs.shape == (24,) and term.listed_names == ["lsa", "lcca", "bct", "rcca", "v7", "v8"]
    first = term.obs.copy()
    term.step()                                       # nothing moves it within an episode
    np.testing.assert_array_equal(term.obs, first)
    assert np.all(np.abs(first) <= 1.0)
    # each entry: the mouth within 3 mm (+ the 2 mm grid), the direction a unit vector
    low, high = box.get()
    for k, (_name, mouth, _direction, _s) in enumerate(route.all_junctions[:6]):
        position = (first[4 * k: 4 * k + 2] + 1.0) / 2.0 * (high - low) + low
        assert np.all(np.abs(position - to_tracking2d(mouth, ANGLE).reshape(2)) <= 3.0 + 1.0 + 1e-6)
        assert np.linalg.norm(first[4 * k + 2: 4 * k + 4]) == pytest.approx(1.0, abs=1e-5)
    # a different episode draw moves the offsets, never the order
    intervention._np_random = np.random.default_rng(7)
    term.reset()
    assert not np.array_equal(term.obs, first) and term.listed_names[0] == "lsa"
    # more slots than junctions: the target fills in
    wide = JunctionMap2D(intervention, referee, n_junctions=12, mode="all", task_box=box)
    wide.reset()
    assert wide.obs.shape == (48,)
    target2d = box.normalize(to_tracking2d(route.target, ANGLE).reshape(2))
    np.testing.assert_allclose(wide.obs[36:38], target2d, atol=1e-5)
    np.testing.assert_allclose(wide.obs[44:46], target2d, atol=1e-5)
    with pytest.raises(ValueError):
        JunctionMap2D(intervention, referee, mode="some")


def test_junction_map_keeps_junctions_past_the_target_with_no_direction(tree, route):
    """A target in the common carotid (before its side branch at 64 mm): the six junctions keep
    the same slots and positions as for the ICA target; the side branch's and the ICA/ECA split's
    directions are (0, 0), the four before the target unit vectors as before."""
    from guidewire_rl.env.two_device.observations import JunctionMap2D, TaskBox2D
    split = route.s_of_fork("v8")
    obs = {}
    for name, s_target in (("ica", split + 40.0), ("cca", route.s_rcca + 40.0)):
        route.set_target(route.point_at(s_target), catheter_goal_s=split + 5.0)
        intervention, referee = fake_scene(tree, route)
        intervention._np_random = np.random.default_rng(3)
        term = JunctionMap2D(intervention, referee, n_junctions=6, mode="all",
                             task_box=TaskBox2D(intervention, region="tree"))
        term.reset()
        assert term.listed_names == ["lsa", "lcca", "bct", "rcca", "v7", "v8"]
        obs[name] = term.obs.reshape(6, 4).copy()
    np.testing.assert_array_equal(obs["cca"][:, :2], obs["ica"][:, :2])        # positions: the vessel's
    np.testing.assert_array_equal(obs["cca"][:4], obs["ica"][:4])              # before the target: unchanged
    np.testing.assert_array_equal(obs["cca"][4:, 2:], 0.0)                     # past it: no direction
    np.testing.assert_allclose(np.linalg.norm(obs["ica"][:, 2:], axis=1), 1.0, atol=1e-5)
    s = [s for _, _, _, s in route.all_junctions[:6]]
    assert s[3] < route.s_rcca + 40.0 < s[4]


def test_landmarks_come_from_the_route_and_count_in_order(route):
    """The start curriculum's landmark grid (user 2026-11-10 14:56, 22:24): nine milestones on this
    route, generated from its structure; the grid is 1 + the milestones holding from the first."""
    from guidewire_rl.env.two_device.phases import CATH, WIRE, PhaseDetector
    split = route.s_of_fork("v8")
    route.set_target(route.point_at(split + 40.0), catheter_goal_s=split + 5.0)
    referee = PhaseDetector(intervention=None)
    referee.route = route
    marks = referee.landmarks = referee.make_landmarks()
    s_O, s_rcca, s_v7 = route.s_O, route.s_rcca, route.s_of_fork("v7")
    expected = [(CATH, s_O), (CATH, s_O + 8.0), (WIRE, s_rcca + 8.0), (WIRE, s_v7), (WIRE, split + 8.0),
                (WIRE, None), (CATH, s_rcca + 8.0), (CATH, s_rcca + 60.0), (CATH, split + 5.0)]
    assert len(marks) == 9
    for (_what, device, s), (device_e, s_e) in zip(marks, expected):
        assert device == device_e and (s is None if s_e is None else s == pytest.approx(s_e))
    g = route.graph
    at = lambda s: route.locate(route.point_at(s))
    aorta = lambda x: route.locate(g.point("aorta", x))
    T = route.target
    far = T + 100.0
    # the procedure in order: each step adds one milestone
    walk = [(aorta(s_O - 30.0), aorta(s_O - 40.0), far),        # 1: nothing yet
            (aorta(s_O - 30.0), aorta(s_O + 20.0), far),        # 2: catheter past the trunk along the aorta
            (aorta(s_O - 30.0), at(s_O + 12.0), far),           # 3: catheter seated in the trunk
            (at(s_rcca + 10.0), at(s_O + 12.0), far),           # 4: wire into the carotid
            (at(s_v7 + 3.0), at(s_O + 12.0), far),              # 5: wire past the side branch
            (at(split + 10.0), at(s_O + 12.0), far),            # 6: wire into the ICA
            (at(split + 40.0), at(s_O + 12.0), T),              # 7: wire at the target
            (at(split + 40.0), at(s_rcca + 10.0), T),           # 8: catheter into the carotid
            (at(split + 40.0), at(s_rcca + 62.0), T),           # 9: catheter at the first tier's depth
            (at(split + 40.0), at(split + 6.0), T)]             # 10: catheter at F
    grids = [1 + referee.landmarks_held(w, c, p) for w, c, p in walk]
    assert grids == list(range(1, 11))
    # the catheter swung out up the aorta counts as past the trunk's mouth, never as seated in it
    assert 1 + referee.landmarks_held(aorta(s_O - 30.0), aorta(s_O + 40.0), far) == 2
    # in order: the wire up the carotid does not count while the catheter is not seated
    assert 1 + referee.landmarks_held(at(s_rcca + 10.0), aorta(s_O - 40.0), far) == 1


def test_tier_flag_and_the_catheters_goal_in_the_image(tree, route):
    from guidewire_rl.env.two_device.observations import TaskBox2D, TaskTarget2D, TierFlag
    from guidewire_rl.env.two_device.route import to_tracking2d
    split = route.s_of_fork("v8")
    route.set_target(route.point_at(split + 40.0), catheter_goal_s=split + 5.0)
    intervention, referee = fake_scene(tree, route)
    flag = TierFlag(referee)
    flag.reset()
    assert flag.obs.tolist() == [0.0] and flag.space.shape == (1,)
    referee.state = made_state(tier1_done=True)
    flag.step()
    assert flag.obs.tolist() == [1.0]
    goal = TaskTarget2D(intervention, point="F", phase_detector=referee, task_box=TaskBox2D(intervention))
    goal.reset()
    np.testing.assert_allclose(goal.obs, to_tracking2d(route.F, ANGLE).reshape(2), atol=1e-5)
    with pytest.raises(ValueError):
        TaskTarget2D(intervention, point="F")
    with pytest.raises(ValueError):
        TaskTarget2D(intervention, point="middle")


# ------------------------------------------------------------------------------- config ----
def test_round_two_config_is_the_checklist():
    from guidewire_rl.env import registry
    config = load_config(ROUND2)
    env = config["env"]
    terms = [(t["name"], t["type"]) for t in env["observation"]]
    assert [n for n, _ in terms] == ["wire_tip", "wire_shape", "cath_tip", "cath_shape", "target", "cath_target",
                                     "junctions", "inserted", "d", "rotation", "last_actions", "tier1", "history"]
    assert "suggestion" not in [n for n, _ in terms] and "phase" not in [n for n, _ in terms]
    junctions = next(t for t in env["observation"] if t["name"] == "junctions")
    assert junctions["mode"] == "all" and junctions["n_junctions"] == 6
    assert env["reward"] == ["ZeroReward"]
    formula = env["reward_formula"]
    assert formula["type"] == "TwoDeviceFormula" and formula["progress_ruler"] == "nearest"
    assert (formula["progress"], formula["follow"], formula["step_penalty"], formula["wrong_vessel"]) == (0.01, 0.01, 0.002, 0.1)
    assert (formula["force_cap"], formula["cath_knee_N"], formula["cath_width_N"], formula["wire_knee_N"],
            formula["wire_width_N"]) == (0.1, 2.0, 3.0, 1.0 / 2, 1.0)
    assert formula["arch_ruler"] is False and formula["lead_soft_min_mm"] is None
    assert formula["action_change"] == 0.005                     # smoothness (user 2026-11-13)
    assert env["action"]["type"] == "TwoDeviceLimits" and "force_limits_N" not in env["action"]
    assert env["truncation"] == [{"type": "MaxSteps", "max_steps": 800}, "SimError", "WallBreach", "TipForceBlowup",
                                 {"type": "Stalled", "window_steps": 200, "min_mm": 3.0, "always": False}]
    assert config["evaluation"]["end_when_stalled"] is False     # off until switched on for an evaluation
    referee = next(m for m in env["monitors"] if m["name"] == "phase_detector")
    assert referee["split_branch"] == "v8" and referee["catheter_goal_past_split_mm"] == 5.0
    assert referee["thresholds"] == {"tier1_cath_depth": 60.0, "tier2_cath_past_split": 5.0, "wrong_in": 8.0,
                                     "wrong_out": 0.0, "seated": 8.0, "unseated": 0.0, "target_mm": 5.0}
    assert any(m["type"] == "TaskBox2D" and m["region"] == "tree" for m in env["monitors"])
    target = env["scene"]["target"]
    assert (target["past_fork"], target["depth_min_mm"], target["depth_max_mm"]) == ("v8", 35.0, 45.0)
    training = config["training"]
    assert "cutoff_mixture" not in training and "stratify_by" not in (config.get("replay") or {})
    assert training["discount_min"] == training["discount_max"] == 0.99375
    assert training["start_curriculum"]["bins"] == 10 and training["start_curriculum"]["baseline_fraction"] == 0.3
    assert training["demos"] == {"dir": "runs/demos_round2c", "fraction": 0.25}   # user 2026-11-13
    assert training["self_demos"] == {"enabled": True, "capacity": 500}
    assert [h["label"] for h in config["aux_heads"]] == ["distal_force", "catheter_tip_force",
                                                          "wire_tip_clearance", "catheter_tip_clearance"]
    assert all(h["loss"] == "huber" and h["weight"] == 1.0 for h in config["aux_heads"])
    assert config["evaluation"]["interval"] == 25000 and config["evaluation"]["episodes"] == 50
    for kind, specs in (("monitor", env["monitors"]), ("observation", env["observation"]),
                        ("reward", env["reward"] + [env["reward_formula"]]), ("terminal", env["terminal"]),
                        ("truncation", env["truncation"]), ("info", env["info"]), ("action", [env["action"]])):
        for spec in specs:
            registry.resolve(kind, registry.as_spec(spec)["type"])


@pytest.mark.sofa
@pytest.mark.upstream          # the round-two scene needs the new env's stEVE (the Angled wire)
def test_env_charges_the_change_of_the_executed_action_it_shows():
    """The env's action_change material is the change of the newest executed action between two
    consecutive observations (what the policy sees), and the live reward pays 0.005 x it."""
    from guidewire_rl.env import build_env
    from guidewire_rl.env.two_device import reward_formula as rf
    cfg = load_config(ROUND2)
    env = build_env(cfg["env"])
    try:
        assert env.action_slice == rf.action_slice_from_names(env.observation_names, 4)
        lo, hi = env.action_slice
        obs, _ = env.reset(seed=1000)
        assert obs[lo:hi] == pytest.approx(np.zeros(4))                     # nothing executed yet
        plain = rf.Coefficients.from_config({**cfg["env"]["reward_formula"], "action_change": 0.0})
        steps = [np.array([0.5, 0.2, 0.4, -0.3], np.float32), np.array([0.5, 0.2, 0.4, -0.3], np.float32),
                 np.array([-0.5, -0.2, -0.4, 0.3], np.float32)]
        for k, action in enumerate(steps):
            previous = obs
            obs, reward, _, _, info = env.step(action)
            m = info["reward_materials"]
            executed = obs[lo:hi]
            assert executed == pytest.approx(info["executed_action"], abs=1e-5)  # the observation = the executed command
            expected = float(np.mean((executed - previous[lo:hi]) ** 2))
            assert m[rf.INDEX["action_change"]] == pytest.approx(expected, abs=1e-6)
            assert reward == pytest.approx(float(rf.compute(m, plain)) - 0.005 * expected, abs=1e-6)
        assert info["reward_materials"][rf.INDEX["action_change"]] > 0.1          # the turnaround is charged
    finally:
        env.close()
