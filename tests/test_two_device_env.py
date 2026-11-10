"""The two-device task's MDP parts (guidewire_rl/env/two_device/, configs/two_devices_round1.yaml).

Without SOFA: the route's ruler on the rebuilt phantom, the referee on made-up states, every
reward term (potentials continuous at their bounds, the hook term locked once P3 is reached, the
arch ruler of P0 meeting the plain one at the hook point), the safety shell's stages (same-length
guard: only the crossing device is sped up, the other never moved), the observation helpers and
the config. With SOFA (``sofa``): the env
builds with the plan's 110 observations, the info keys the trainer reads, ``LastAction`` reads the
executed command, the cutoff reaches the truncation, the per-instrument tip force equals G13's
distal force for the leading device, and a scripted push gets the referee from P0 to P1.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import apply_override, load_config

CONFIG = WORKSPACE_ROOT / "configs" / "two_devices_round1.yaml"
T = 1.0 / 7.5


# ---------------------------------------------------------------------------- route ----
@pytest.fixture(scope="module")
def route():
    from guidewire_rl.env.two_device.route import Route
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    return Route(BiroPhantomArch(version="rebuilt_cut"))


def test_route_landmarks(route):
    g = route.graph
    assert route.chain == ("aorta", "bct", "rcca", "rcca_rest")
    assert route.s_O == pytest.approx(177.9, abs=0.2)           # trunk's junction on the aorta
    assert g.mouth_arc[g.index["bct"]] == pytest.approx(19.7, abs=0.2)   # its mouth on the wall
    assert g.mouth_arc[g.index["rcca"]] == 0.0                  # carotid opens at the trunk's end
    assert route.s_rcca == pytest.approx(route.s_O + 59.7 - 19.7, abs=0.3)
    assert [name for name, _, _ in route.junctions] == ["bct", "rcca"]
    # the trunk's mouth is where the log measured the opening (z 95.1)
    assert route.junctions[0][1][2] == pytest.approx(95.1, abs=0.3)
    # first fork of the carotid (v7) 64 mm past its opening
    v7 = g.point("v7", 0.0)
    assert route.depth(v7, "rcca") == pytest.approx(63.8, abs=0.5)


def test_route_ruler_is_continuous_along_the_route(route):
    route.set_target(route.point_at(route.s_rcca + 45.0))
    s = np.arange(5.0, route.s_rcca + 80.0, 1.0)
    points = [route.point_at(x) for x in s]
    along = np.array([route.s(p) for p in points])
    np.testing.assert_allclose(along, s, atol=1e-6)
    depth = np.array([route.depth(p, "bct") for p in points])
    assert np.max(np.abs(np.diff(depth))) <= 1.0 + 1e-6          # no jump at the trunk's mouth
    k = int(np.searchsorted(s, route.s_O + 10.0))
    assert depth[k] == pytest.approx(s[k] - route.s_O, abs=1e-6)    # mm past the mouth
    to_target = np.array([route.path_to_target(p) for p in points])
    k = int(np.argmin(to_target))
    assert s[k] == pytest.approx(route.s_rcca + 45.0, abs=1.0)
    np.testing.assert_allclose(np.diff(to_target[:k]), -1.0, atol=1e-4)   # float32 junctions
    to_F = np.array([route.path_to_F(p) for p in points])
    assert np.all(to_F[s >= route.s_rcca + 15.0] == 0.0)        # past F: done, going deeper is fine


def test_route_side_vessels(route):
    g = route.graph
    # ascending aorta past O continues the aorta's arc length and counts away from the target
    p = g.point("aorta", route.s_O + 40.0)
    assert route.aorta_position(p) == pytest.approx(40.0, abs=1e-6)
    assert route.s(p) == pytest.approx(route.s_O + 40.0, abs=1e-6)
    assert route.depth(p, "bct") < 0
    for vessel in ("lsa", "lcca"):
        q = g.point(vessel, g.mouth_arc[g.index[vessel]] + 8.0)
        assert route.branch_of(q) == vessel
        assert route.depth(q, vessel) == pytest.approx(8.0, abs=1e-6)
        assert route.depth(q, "bct") < 0
    q = g.point("rsa", 8.0)
    assert route.depth(q, "rsa") == pytest.approx(8.0, abs=1e-6)
    assert route.depth(q, "bct") > 0 and route.depth(q, "rcca") < 0


def test_route_target_window_and_split(route):
    from guidewire_rl.env.two_device.targets import RouteWindowTarget
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    tree = BiroPhantomArch(version="rebuilt_cut")
    fluoro = SimpleNamespace(image_rot_zx=[61.0, -21.0], image_center=[0, 0, 0], field_of_view=None)
    train = RouteWindowTarget(tree, fluoro, split="train")
    evaluate = RouteWindowTarget(tree, fluoro, split="eval")
    assert train.candidate_depths == [35.0 + 2 * i for i in range(11)]
    assert evaluate.candidate_depths == [36.0 + 2 * i for i in range(10)]
    from guidewire_rl.env.two_device.route import Route
    r = Route.of(tree)
    for target in (train, evaluate):
        for point, depth in zip(target.candidates(), target.candidate_depths):
            assert r.depth(point, "rcca") == pytest.approx(depth, abs=1e-4)
    train.reset(seed=4)
    from eve.util.coordtransform import tracking3d_to_vessel_cs
    back = tracking3d_to_vessel_cs(train.coordinates3d, fluoro.image_rot_zx, fluoro.image_center)
    np.testing.assert_allclose(back, train.coordinates_vessel, atol=1e-3)
    assert train.depth_mm in train.candidate_depths


# ------------------------------------------------------------------------- referee ----
def made_state(**fields):
    from guidewire_rl.env.two_device.phases import PhaseState
    base = dict(d=20.0, L_wire=100.0, L_cath=80.0, depth_bct_c=-50.0, depth_rcca_c=-90.0,
                depth_rcca_w=-70.0, hook_up=False, aorta_pos_c=-60.0, branch_w="aorta",
                branch_c="aorta", clearance_w=2.0, clearance_c=2.0, dist_to_target_w=100.0,
                depth_wrong={"lcca": -30.0, "lsa": -30.0, "rsa": -60.0})
    base.update(fields)
    return PhaseState(**base)


@pytest.fixture
def referee():
    from guidewire_rl.env.two_device.phases import PhaseDetector
    detector = PhaseDetector(intervention=None)
    detector.state = made_state()
    return detector


def run(detector, states):
    return [detector.advance(s) for s in states]


def test_referee_needs_two_steps_and_goes_forward(referee):
    out = run(referee, [made_state(aorta_pos_c=31.0)])
    assert out[-1].phase == 0                                    # one step is not enough
    out = run(referee, [made_state(aorta_pos_c=29.0), made_state(aorta_pos_c=31.0)])
    assert [s.phase for s in out] == [0, 0]                      # broken run starts over
    out = run(referee, [made_state(aorta_pos_c=32.0)])
    assert out[-1].phase == 1 and out[-1].steps_in_phase == 0 and out[-1].max_phase == 1
    hooked = dict(aorta_pos_c=35.0, d=-36.0, hook_up=True)
    out = run(referee, [made_state(**hooked), made_state(**hooked)])
    assert out[-1].phase == 2
    out = run(referee, [made_state(aorta_pos_c=0.0, d=-40.0, depth_bct_c=9.0)] * 2)
    assert out[-1].phase == 3
    out = run(referee, [made_state(aorta_pos_c=0.0, d=12.0, depth_bct_c=20.0, depth_rcca_w=16.0)] * 2)
    assert out[-1].phase == 4 and out[-1].max_phase == 4


def test_referee_falls_back(referee):
    referee.state = made_state()
    referee.state.phase = 2
    # hook straightened while the catheter is still in the aorta -> P1
    out = run(referee, [made_state(aorta_pos_c=10.0, d=-5.0)] * 2)
    assert out[-1].phase == 1
    # far below O -> P0 (checked before the straightened hook)
    referee.state.phase = 2
    out = run(referee, [made_state(aorta_pos_c=-61.0, d=-5.0)] * 2)
    assert out[-1].phase == 0
    # engaged in the trunk wins over a straight hook
    referee.state.phase = 2
    out = run(referee, [made_state(aorta_pos_c=0.0, d=-5.0, depth_bct_c=8.5)] * 2)
    assert out[-1].phase == 3
    # P4 -> P2 when the catheter leaves the opening, counted as dislodged once
    referee.state.phase = 4
    out = run(referee, [made_state(aorta_pos_c=0.0, d=30.0, depth_bct_c=3.0, depth_rcca_w=40.0)] * 2)
    assert out[-1].phase == 2 and out[-1].events["dislodged"] == 1 and out[0].events["dislodged"] == 0
    # hysteresis: between 4 and 8 mm into the trunk nothing changes in P2 or P3
    referee.state.phase = 3
    out = run(referee, [made_state(aorta_pos_c=0.0, d=-20.0, depth_bct_c=6.0)] * 3)
    assert out[-1].phase == 3


def test_referee_recovers_after_the_catheter_slips_out_with_the_wire_out(referee):
    """WP-B's seed 8: in P4 the catheter slips briefly out of the opening (< 4 mm), P2; the wire is
    already out (d > 0, up the carotid), so that is no straightened hook: P2 holds, the catheter
    sits 24 mm in again -> P3, the wire is still 15+ mm in the carotid -> P4."""
    referee.state.phase = 4
    up = dict(aorta_pos_c=0.0, d=30.0, depth_rcca_w=40.0, depth_bct_w=100.0)
    out = run(referee, [made_state(depth_bct_c=3.0, **up)] * 2)
    assert out[-1].phase == 2 and out[-1].events["dislodged"] == 1
    out = run(referee, [made_state(depth_bct_c=x, **up) for x in (3.0, 5.0, 6.0, 7.0)])
    assert [s.phase for s in out] == [2, 2, 2, 2]            # not P1 (used to drop there for good)
    out = run(referee, [made_state(depth_bct_c=24.0, **up)] * 2)
    assert out[-1].phase == 3
    out = run(referee, [made_state(depth_bct_c=24.0, **up)] * 2)
    assert out[-1].phase == 4 and out[-1].max_phase == 4
    # still P1 when the wire inside the catheter straightens the hook in the aorta ...
    referee.state.phase = 2
    out = run(referee, [made_state(aorta_pos_c=10.0, d=-5.0, depth_bct_w=-20.0)] * 2)
    assert out[-1].phase == 1
    # ... but not when that wire tip is already past the trunk's mouth (the catheter is in deep)
    referee.state.phase = 2
    out = run(referee, [made_state(aorta_pos_c=0.0, d=-5.0, depth_bct_c=6.0, depth_bct_w=1.0)] * 3)
    assert out[-1].phase == 2


def test_referee_events_count_once(referee):
    seq = [made_state(depth_wrong={"lcca": 9.0, "lsa": -30.0, "rsa": -60.0})] * 3 + \
          [made_state(depth_wrong={"lcca": 3.0, "lsa": -30.0, "rsa": -60.0}),
           made_state(depth_wrong={"lcca": 9.0, "lsa": -30.0, "rsa": -60.0})]
    out = run(referee, seq)
    assert [s.events["wrong_vessel"] for s in out] == [1, 0, 0, 0, 1]
    assert out[-1].event_totals["wrong_vessel"] == 2
    out = run(referee, [made_state(d=101.0)] * 2 + [made_state(d=80.0), made_state(d=105.0)])
    assert [s.events["wire_lead_arch"] for s in out] == [1, 0, 0, 1]
    out = run(referee, [made_state(depth_rcca_w=65.0)] * 2)
    assert [s.events["wire_past_fork"] for s in out] == [1, 0]
    referee.report("bare_advance")
    referee.report("bare_advance")
    assert referee.pending_events["bare_advance"] == 2


def test_referee_success_and_failures(referee):
    referee.state.phase = 4
    platform = dict(aorta_pos_c=0.0, d=30.0, depth_bct_c=60.0, depth_rcca_c=20.0,
                    depth_rcca_w=50.0, dist_to_target_w=3.0)
    out = run(referee, [made_state(**platform)] * 7)
    assert not out[-1].success and out[-1].platform_steps == 7
    out = run(referee, [made_state(**platform)])
    assert out[-1].success
    bad = dict(platform, d=50.0)                                # wire too far ahead
    assert not run(referee, [made_state(**bad)])[-1].success
    out = run(referee, [made_state(clearance_w=-7.0)])
    assert out[-1].hard_failure and out[-1].failure_reason == "outside_wire"
    out = run(referee, [made_state(clearance_c=-5.0)])        # within the tubes' 6 mm margin
    assert not out[-1].hard_failure
    out = run(referee, [made_state(d=0.2, L_wire=50.2, L_cath=50.0)])
    assert out[-1].failure_reason == "same_length"
    out = run(referee, [made_state(d=0.0, L_wire=0.0, L_cath=0.0)])   # the start is not a failure
    assert not out[-1].hard_failure


# ------------------------------------------------------------------------- rewards ----
class FakeReferee:
    def __init__(self, state):
        self.state = state


def rewards_of(term, states):
    referee = FakeReferee(states[0])
    term.phase_detector = referee
    term.reset()
    out = []
    for state in states[1:]:
        referee.state = state
        term.step()
        out.append(term.reward)
    return np.array(out)


def test_hook_potential_is_continuous_at_its_bounds():
    from guidewire_rl.env.two_device.rewards import HookPrepPotential
    term = HookPrepPotential(FakeReferee(made_state()), factor=0.006)
    d = np.linspace(10.0, -50.0, 121)                            # crosses 0 and -35
    r = rewards_of(term, [made_state(d=x, phase=1, previous_phase=1) for x in d])
    potential = np.clip(-d, 0.0, 35.0)
    np.testing.assert_allclose(r, 0.006 * np.diff(potential), atol=1e-12)
    assert np.all(np.abs(r) <= 0.006 * 0.5 + 1e-12)            # no step bigger than the d step
    assert r.sum() == pytest.approx(0.006 * 35.0)


def test_hook_potential_locks_at_P3_and_never_claws_back():
    from guidewire_rl.env.two_device.rewards import HookPrepPotential
    term = HookPrepPotential(FakeReferee(made_state()), factor=0.006)
    states = [made_state(d=0.0, phase=1, previous_phase=1)]
    states += [made_state(d=-x, phase=2, previous_phase=2) for x in (10.0, 20.0, 35.0)]
    states += [made_state(d=-35.0, phase=3, previous_phase=2)]                     # P2 -> P3
    states += [made_state(d=x, phase=3, previous_phase=3) for x in (-20.0, 0.0, 15.0)]   # wire out
    states += [made_state(d=15.0, phase=2, previous_phase=3),                       # fall back
               made_state(d=-30.0, phase=2, previous_phase=2)]
    r = rewards_of(term, states)
    assert r[:3].sum() == pytest.approx(0.006 * 35.0)
    assert np.all(r[3:] == 0.0)
    assert term.locked
    term.reset()
    assert not term.locked


def test_progress_potentials():
    from guidewire_rl.env.two_device.rewards import (CatheterToF, LeadingTipProgress,
                                                     PhasePotential)
    # the lead changes hands: min(path) is continuous, the sum telescopes
    states = [made_state(path_to_target_w=w, path_to_target_c=c, phase=1)
              for w, c in ((100.0, 110.0), (98.0, 99.0), (97.0, 92.0), (96.0, 90.0))]
    r = rewards_of(LeadingTipProgress(FakeReferee(states[0]), 0.01), states)
    np.testing.assert_allclose(r, [0.02, 0.06, 0.02], atol=1e-12)
    r = rewards_of(CatheterToF(FakeReferee(states[0]), 0.003),
                   [made_state(path_to_F_c=x, phase=1) for x in (50.0, 40.0, 0.0, 0.0)])
    np.testing.assert_allclose(r, [0.03, 0.12, 0.0], atol=1e-12)
    r = rewards_of(PhasePotential(FakeReferee(states[0]), 0.2),
                   [made_state(phase=p) for p in (0, 1, 2, 1, 2)])
    np.testing.assert_allclose(r, [0.2, 0.2, -0.2, 0.2], atol=1e-12)


@pytest.fixture
def arch_referee(route):
    """A referee wired to the real route (no simulator): the hook point H = O + 30 mm."""
    from guidewire_rl.env.two_device.phases import PhaseDetector
    detector = PhaseDetector(intervention=None)
    route.set_target(route.point_at(route.s_rcca + 45.0))
    detector.route = route
    detector.hook_point = route.aorta_location(route.s_O + 30.0)
    detector.hook_to_target = route.path_to_target(None, detector.hook_point)
    detector.hook_to_F = route.path_to_F(None, detector.hook_point)
    return detector


def arch_state(detector, s_wire, s_cath, phase):
    """A made-up state with both tips on the aorta's centerline at arc s (vessel geometry real)."""
    route = detector.route
    lw, lc = (route.locate(route.graph.point("aorta", x)) for x in (s_wire, s_cath))
    return made_state(phase=phase, path_to_target_w=route.path_to_target(None, lw),
                      path_to_target_c=route.path_to_target(None, lc), path_to_F_c=route.path_to_F(None, lc),
                      to_hook_w=detector._to_hook(lw), to_hook_c=detector._to_hook(lc),
                      hook_to_target=detector.hook_to_target, hook_to_F=detector.hook_to_F)


def test_arch_ruler_meets_the_plain_one_at_the_hook_point(arch_referee):
    route, H = arch_referee.route, arch_referee.route.s_O + 30.0
    at_hook = arch_state(arch_referee, H, H, phase=0).progress_paths()
    plain = arch_state(arch_referee, H, H, phase=1).progress_paths()
    np.testing.assert_allclose(at_hook, plain, atol=1e-6)
    # before H: what is left to H plus the fixed rest; past H in P0: flat
    before = arch_state(arch_referee, H - 25.0, H - 40.0, phase=0).progress_paths()
    np.testing.assert_allclose(before, [25.0 + arch_referee.hook_to_target,
                                        40.0 + arch_referee.hook_to_target,
                                        40.0 + arch_referee.hook_to_F], atol=1e-6)
    past = arch_state(arch_referee, H + 30.0, H + 10.0, phase=0).progress_paths()
    np.testing.assert_allclose(past, at_hook, atol=1e-6)
    # a wrong vessel is not progress in P0: deeper into the left subclavian is further from H
    g = route.graph
    lsa = [route.locate(g.point("lsa", g.mouth_arc[g.index["lsa"]] + x)) for x in (5.0, 15.0)]
    assert arch_referee._to_hook(lsa[1]) > arch_referee._to_hook(lsa[0])


def test_arch_ruler_rewards_reaching_the_hook_point_and_does_not_jump_into_P1(arch_referee):
    from guidewire_rl.env.two_device.rewards import CatheterToF, LeadingTipProgress
    H = arch_referee.route.s_O + 30.0
    cath = np.arange(H - 20.0, H + 6.0, 2.0)            # catheter pushed past O to H and on
    states = [arch_state(arch_referee, x + 15.0, x, phase=0 if x < H else 1) for x in cath]
    r_cath = rewards_of(CatheterToF(FakeReferee(states[0]), 0.003), states)
    r_lead = rewards_of(LeadingTipProgress(FakeReferee(states[0]), 0.01), states)
    switch = int(np.argmax(cath >= H)) - 1              # reward index of the step into P1
    np.testing.assert_allclose(r_cath[:switch + 1], 0.003 * 2.0, atol=1e-6)   # 2 mm closer to H each step
    assert np.all(r_cath[switch + 1:] < 0)              # P1: deeper than H is away from F
    wire_left = np.maximum(H - (cath + 15.0), 0.0)        # wire's way left to H
    np.testing.assert_allclose(r_lead[:switch + 1], 0.01 * -np.diff(wire_left)[:switch + 1], atol=1e-6)
    assert abs(r_lead[switch]) <= 0.01 * 2.0 + 1e-6      # no jump at P0 -> P1


def test_simulation_error_steps_pay_nothing_and_move_no_potential():
    from guidewire_rl.env.two_device.rewards import LeadingTipProgress
    states = [made_state(path_to_target_w=100.0, path_to_target_c=120.0, phase=1),
              made_state(path_to_target_w=90.0, path_to_target_c=120.0, phase=1),
              made_state(path_to_target_w=260.0, path_to_target_c=260.0, sim_error=True, phase=1),
              made_state(path_to_target_w=85.0, path_to_target_c=120.0, phase=1)]
    r = rewards_of(LeadingTipProgress(FakeReferee(states[0]), 0.01), states)
    np.testing.assert_allclose(r, [0.1, 0.0, 0.05], atol=1e-12)


def test_penalties_and_outcome():
    from guidewire_rl.env.two_device.rewards import (EventPenalty, OutcomeReward, ShieldPenalty,
                                                     StepPenalty, WindowPenalty)
    window = WindowPenalty(FakeReferee(made_state()), factor=0.001)
    capped = WindowPenalty(FakeReferee(made_state()), factor=0.001, cap_mm=30.0)
    assert capped.outside_mm(made_state(phase=0, d=200.0)) == 30.0   # 120 mm beyond P0's window, capped
    r = rewards_of(capped, [made_state(), made_state(phase=0, d=200.0), made_state(phase=0, d=95.0)])
    np.testing.assert_allclose(r, [-0.03, -0.015], atol=1e-12)
    assert window.outside_mm(made_state(phase=0, d=5.0)) == 5.0
    assert window.outside_mm(made_state(phase=0, d=50.0)) == 0.0
    assert window.outside_mm(made_state(phase=2, d=-20.0)) == 10.0
    assert window.outside_mm(made_state(phase=4, d=70.0)) == 10.0
    # P2 with the wire out and past the trunk's mouth (slipped back from P3/P4): P3's window
    assert window.outside_mm(made_state(phase=2, d=30.0, depth_bct_w=50.0)) == 0.0
    assert window.outside_mm(made_state(phase=2, d=90.0, depth_bct_w=50.0)) == 10.0
    assert window.outside_mm(made_state(phase=2, d=-110.0, depth_bct_w=-5.0)) == 0.0
    # ... but the plain P2 window while the wire tip is still behind the mouth or inside the catheter
    assert window.outside_mm(made_state(phase=2, d=30.0, depth_bct_w=-5.0)) == 60.0
    assert window.outside_mm(made_state(phase=2, d=-20.0, depth_bct_w=5.0)) == 10.0
    events = {"wrong_vessel": 2, "dislodged": 1, "bare_advance": 0, "wire_lead_arch": 0,
              "wire_past_fork": 0}
    r = rewards_of(EventPenalty(FakeReferee(made_state())), [made_state(), made_state(events=events)])
    assert r[0] == pytest.approx(-0.2)
    r = rewards_of(OutcomeReward(FakeReferee(made_state())),
                   [made_state(), made_state(success=True), made_state(hard_failure=True), made_state()])
    np.testing.assert_allclose(r, [1.0, -1.0, 0.0])
    shell = SimpleNamespace(last=SimpleNamespace(penalized=0.4))
    term = ShieldPenalty(shell, factor=0.05)
    term.step()
    assert term.reward == pytest.approx(-0.02)
    step = StepPenalty()
    step.step()
    assert step.reward == -0.002


# --------------------------------------------------------------------------- shell ----
class ShellReferee:
    def __init__(self, **fields):
        self.state = made_state(**fields)
        self.reported = []

    def report(self, event, count=1):
        self.reported.append(event)


def make_shell(nominal=None, forces=(0.0, 0.0), shell=None, **state):
    from guidewire_rl.env.two_device.actions import TwoDeviceGuarded
    referee = ShellReferee(**state)
    labels = [SimpleNamespace(name="distal_force", last_value=forces[0]),
              SimpleNamespace(name="catheter_tip_force", last_value=forces[1])]
    advisor = None if nominal is None else SimpleNamespace(action=np.asarray(nominal, float))
    intervention = SimpleNamespace(fluoroscopy=SimpleNamespace(image_frequency=7.5))
    return TwoDeviceGuarded(intervention, referee, labels=labels, nominal_advisor=advisor,
                            **(shell or {})), referee


HIGH = np.array([40.0, np.pi, 25.0, np.pi / 2])


def shell_step(shell, action, high=HIGH):
    raw = shell.to_raw(np.asarray(action, np.float32), -high, high)
    return raw.astype(np.float64), shell.last


def test_shell_speed_limit_and_residual_box():
    shell, _ = make_shell(d=20.0, L_wire=100.0, L_cath=80.0)
    big = np.array([50.0, 4.0, 50.0, 4.0])                        # instruments allow more
    raw, record = shell_step(shell, [1.0, 1.0, 1.0, 1.0], high=big)
    np.testing.assert_allclose(raw, [40.0, np.pi, 25.0, np.pi / 2], rtol=1e-6)
    assert record.stages == ["speed"] and record.penalized > 0
    nominal = [0.2, 0.0, -0.5, 0.5]
    shell, _ = make_shell(nominal=nominal, phase=1, d=-40.0, L_wire=100.0, L_cath=140.0)
    raw, record = shell_step(shell, [1.0, -1.0, 1.0, 0.0])
    assert "residual" not in record.stages                          # off by default (planner's job)
    on = {"residual_phases": [1]}
    shell, _ = make_shell(nominal=nominal, shell=on, phase=1, d=-40.0, L_wire=100.0, L_cath=140.0)
    raw, record = shell_step(shell, [1.0, -1.0, 1.0, 0.0])
    np.testing.assert_allclose(raw / HIGH, [0.7, -0.5, 0.0, 0.0], atol=1e-6)
    assert "residual" in record.stages and record.penalized == 0.0   # the box is not charged
    shell, _ = make_shell(nominal=nominal, shell=on, phase=2, d=-40.0, L_wire=100.0, L_cath=140.0)
    raw, record = shell_step(shell, [1.0, -1.0, 1.0, 0.0])
    assert "residual" not in record.stages                          # only in the listed phases


def test_shell_branch_push_caps():
    # in a branch the pushes are capped: catheter 0.4 of its limit, wire 0.5 of its limit
    caps = {"branch_push_max": 0.4, "wire_branch_push_max": 0.5}
    shell, _ = make_shell(shell=caps, branch_c="bct", branch_w="rcca", d=30.0, L_wire=260.0, L_cath=230.0)
    raw, record = shell_step(shell, [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(raw, [0.5 * 40.0, 0.0, 0.4 * 25.0, 0.0], rtol=1e-6)
    assert "speed" in record.stages and record.penalized > 0
    shell, _ = make_shell(shell=caps, branch_c="aorta", branch_w="aorta", d=30.0, L_wire=100.0, L_cath=70.0)
    raw, record = shell_step(shell, [1.0, 0.0, 1.0, 0.0])            # not in a branch: no cap
    np.testing.assert_allclose(raw, [40.0, 0.0, 25.0, 0.0], rtol=1e-6)
    shell, _ = make_shell(branch_c="bct", branch_w="rcca", d=30.0, L_wire=260.0, L_cath=230.0)
    raw, record = shell_step(shell, [1.0, 0.0, 1.0, 0.0])            # caps off by default
    np.testing.assert_allclose(raw, [40.0, 0.0, 25.0, 0.0], rtol=1e-6)


def test_shell_hard_rule_stops_fast_bare_catheter_advance():
    # catheter in the trunk, 20 mm ahead of the wire, pushed fast: cut to 0.15 (3.75 mm/s)
    shell, referee = make_shell(branch_c="bct", d=-20.0, L_wire=180.0, L_cath=200.0)
    raw, record = shell_step(shell, [0.0, 0.0, 0.8, 0.3])
    assert raw[2] == pytest.approx(0.15 * HIGH[2]) and raw[3] == pytest.approx(0.3 * HIGH[3])
    assert record.stages == ["hard_rule"] and referee.reported == ["bare_advance"]
    assert record.penalized == pytest.approx(0.65)
    # slow feeding over the wire passes (the lab script's 3 mm/s = 0.12), uncounted
    shell, referee = make_shell(branch_c="bct", d=-20.0, L_wire=180.0, L_cath=200.0)
    raw, record = shell_step(shell, [0.0, 0.0, 0.12, 0.0])
    assert raw[2] == pytest.approx(0.12 * HIGH[2]) and record.stages == [] and referee.reported == []
    # the wire is within 10 mm behind the tip: not bare, any push passes
    shell, referee = make_shell(branch_c="bct", d=-8.0, L_wire=192.0, L_cath=200.0)
    raw, record = shell_step(shell, [0.0, 0.0, 0.8, 0.0])
    assert raw[2] == pytest.approx(0.8 * HIGH[2]) and referee.reported == []
    # pulling back is fine; in the aorta anything goes
    shell, referee = make_shell(branch_c="bct", d=-20.0, L_wire=180.0, L_cath=200.0)
    raw, _ = shell_step(shell, [0.0, 0.0, -0.5, 0.0])
    assert raw[2] == pytest.approx(-0.5 * HIGH[2])
    shell, referee = make_shell(branch_c="aorta", d=-20.0, L_wire=180.0, L_cath=200.0)
    raw, _ = shell_step(shell, [0.0, 0.0, 0.8, 0.0])
    assert raw[2] == pytest.approx(0.8 * HIGH[2]) and referee.reported == []


def test_shell_force_limit():
    limits = {"force_limits_N": [1.0, 1.0], "force_limit_hook_factor": 2.0}
    shell, _ = make_shell(forces=(1.5, 0.0), shell=limits, d=30.0, L_wire=130.0, L_cath=100.0)
    raw, record = shell_step(shell, [0.5, 0.0, 0.5, 0.0])
    assert raw[0] == 0.0 and raw[2] == pytest.approx(0.5 * HIGH[2]) and "force" in record.stages
    shell, _ = make_shell(forces=(1.5, 0.0), shell=limits, d=-30.0, L_wire=100.0, L_cath=130.0)
    raw, _ = shell_step(shell, [0.5, 0.0, -0.5, 0.0])               # wire inside: its limit is off
    assert raw[0] == pytest.approx(0.5 * HIGH[0])
    shell, _ = make_shell(forces=(0.0, 1.5), shell=limits, phase=1, d=-40.0, L_wire=100.0, L_cath=140.0)
    raw, _ = shell_step(shell, [0.0, 0.0, 0.5, 0.0])                # P1: catheter limit x2
    assert raw[2] == pytest.approx(0.5 * HIGH[2])
    shell, _ = make_shell(forces=(0.0, 2.5), shell=limits, phase=1, d=-40.0, L_wire=100.0, L_cath=140.0)
    raw, _ = shell_step(shell, [0.0, 0.0, 0.5, 0.0])
    assert raw[2] == 0.0
    # separate limits per device, as the config sets them (wire 0.64 N, catheter 2.42 N)
    shell, _ = make_shell(forces=(0.7, 2.0), shell={"force_limits_N": [0.64, 2.42]},
                          phase=2, d=30.0, L_wire=130.0, L_cath=100.0)
    raw, _ = shell_step(shell, [0.5, 0.0, 0.5, 0.0])
    assert raw[0] == 0.0 and raw[2] == pytest.approx(0.5 * HIGH[2])


def ends_at(raw, L_wire, L_cath):
    return (L_wire + raw[0] * T) - (L_cath + raw[2] * T)


def test_shell_same_length_guard():
    # catheter catching up with the wire: stopped at the band's edge (2.5 mm), charged
    shell, _ = make_shell(d=5.0, L_wire=105.0, L_cath=100.0)
    raw, record = shell_step(shell, [0.0, 0.0, 1.0, 0.0])
    assert ends_at(raw, 105.0, 100.0) == pytest.approx(2.5, abs=1e-6)
    assert raw[0] == 0.0                                            # the wire is not moved
    assert record.stages == ["protector_clamp"] and record.penalized > 0
    # the wire pulled back through the catheter tip at 30 mm/s: sped up (to 38.25) to land 2.5 mm
    # on the other side, free; the catheter is not moved
    shell, _ = make_shell(d=2.6, L_wire=102.6, L_cath=100.0)
    raw, record = shell_step(shell, [-0.75, 0.0, 0.0, 0.0])
    assert ends_at(raw, 102.6, 100.0) == pytest.approx(-2.5, abs=1e-6)
    assert raw[2] == 0.0 and -40.0 <= raw[0] < -30.0
    assert record.stages == ["protector_cross"] and record.penalized == 0.0
    # already at full speed and too far to cross in one step: stopped at the near edge, charged,
    # the catheter still not moved (it used to be pushed to help)
    shell, _ = make_shell(d=3.0, L_wire=103.0, L_cath=100.0)
    raw, record = shell_step(shell, [-1.0, 0.0, 0.0, 0.0])
    assert ends_at(raw, 103.0, 100.0) == pytest.approx(2.5, abs=1e-6)
    assert raw[2] == 0.0 and record.stages == ["protector_clamp"]
    # wire out of a seated catheter (catheter in the trunk, wire just behind its tip): the wire
    # cannot cross alone this step, so it stops at -2.5; the catheter is never pulled back
    shell, _ = make_shell(branch_c="bct", d=-3.0, L_wire=197.0, L_cath=200.0)
    raw, record = shell_step(shell, [1.0, 0.0, 0.0, 0.0])
    assert raw[2] == 0.0 and ends_at(raw, 197.0, 200.0) == pytest.approx(-2.5, abs=1e-6)
    shell, _ = make_shell(branch_c="bct", d=-2.5, L_wire=197.5, L_cath=200.0)
    raw, record = shell_step(shell, [1.0, 0.0, 0.0, 0.0])            # from the edge it crosses alone
    assert raw[2] == 0.0 and record.stages == [] and ends_at(raw, 197.5, 200.0) > 2.5
    # the catheter alone cannot jump the 5 mm band (3.3 mm per step): clamped, the wire not moved
    shell, _ = make_shell(d=2.6, L_wire=102.6, L_cath=100.0)
    raw, record = shell_step(shell, [0.0, 0.0, 1.0, 0.0])
    assert raw[0] == 0.0 and ends_at(raw, 102.6, 100.0) == pytest.approx(2.5, abs=1e-6)
    assert record.stages == ["protector_clamp"]
    # both moving the crossing way: either may be sped up, each in its own direction
    shell, _ = make_shell(d=2.6, L_wire=102.6, L_cath=100.0)
    raw, record = shell_step(shell, [-0.5, 0.0, 0.4, 0.0])           # -20 and +10: d would end at -1.4
    assert raw[0] <= -20.0 and raw[2] >= 10.0 and record.stages == ["protector_cross"]
    assert ends_at(raw, 102.6, 100.0) == pytest.approx(-2.5, abs=1e-6)
    # a device moving away from the band is left alone while the other is slowed
    shell, _ = make_shell(d=4.0, L_wire=104.0, L_cath=100.0)
    raw, record = shell_step(shell, [-1.0, 0.0, -0.2, 0.0])          # wire -40, catheter pulling -5
    assert raw[2] == pytest.approx(-0.2 * HIGH[2]) and ends_at(raw, 104.0, 100.0) == pytest.approx(2.5, abs=1e-6)
    # a clean jump across the band is left alone
    shell, _ = make_shell(d=2.6, L_wire=102.6, L_cath=100.0)
    raw, record = shell_step(shell, [-1.0, 0.0, 0.0, 0.0])
    assert record.stages == [] and ends_at(raw, 102.6, 100.0) < -2.5
    # at the start (both at 0) a step only has to leave the band
    shell, _ = make_shell(d=0.0, L_wire=0.0, L_cath=0.0)
    raw, record = shell_step(shell, [0.3, 0.0, 0.0, 0.0])
    assert record.stages == [] and 0 < ends_at(raw, 0.0, 0.0) < 2.5
    shell, _ = make_shell(d=1.6, L_wire=1.6, L_cath=0.0)
    raw, record = shell_step(shell, [-0.1, 0.0, 0.0, 0.0])          # back towards 0: held
    assert ends_at(raw, 1.6, 0.0) == pytest.approx(1.6, abs=1e-6)
    # stEVE drops a pull that would end below 0 mm; the shell predicts with that and shows it
    shell, _ = make_shell(d=4.0, L_wire=4.5, L_cath=0.5)
    raw, record = shell_step(shell, [-0.9, 0.0, 0.0, 0.0])           # 4.8 mm of 4.5
    assert raw[0] == 0.0 and record.stages == ["device_limit"] and record.penalized == 0.0
    # and the guard never asks for one: crossing would need 38.25 mm/s, the wire's 4.5 mm allow
    # only 33.75, so it stops at the edge instead
    shell, _ = make_shell(d=2.6, L_wire=4.5, L_cath=1.9)
    raw, record = shell_step(shell, [-0.75, 0.0, 0.0, 0.0])
    assert raw[0] * T + 4.5 > 0.0 and record.stages == ["protector_clamp"]
    assert ends_at(raw, 4.5, 1.9) == pytest.approx(2.5, abs=1e-6)


class OpeningIntervention:
    """Just what ``WireFirst`` uses: inserted lengths follow the pushes exactly."""

    def __init__(self, error_after=None):
        self.velocity_limits = HIGH.reshape(2, 2)
        self.fluoroscopy = SimpleNamespace(image_frequency=7.5)
        self.simulation = SimpleNamespace(simulation_error=False)
        self.inserted = np.array([5.0, 3.0])
        self.last_action = np.ones((2, 2))
        self.calls, self.error_after = [], error_after

    @property
    def instrument_lengths_inserted(self):
        return self.inserted.tolist()

    def reset_instruments(self):
        self.calls.append("reset")
        self.inserted = np.zeros(2)

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(2, 2)
        self.calls.append(action.copy())
        self.last_action = action
        self.inserted = self.inserted + action[:, 0] * T
        if self.error_after is not None and len(self.calls) - 1 >= self.error_after:
            self.simulation.simulation_error = True


def test_wire_first_opening_pushes_the_wire_alone_to_40_mm():
    from guidewire_rl.env import registry
    from guidewire_rl.env.two_device.start import WireFirst
    assert registry.resolve("start", "WireFirst") is WireFirst
    assert registry.resolve("start", "InsertionPoint").__module__.startswith("eve.start")
    intervention = OpeningIntervention()
    WireFirst(intervention).reset()
    steps = intervention.calls[1:]
    assert intervention.calls[0] == "reset" and len(steps) == 8        # 7 full steps of 5.33 mm + 2.67
    assert all(np.all(s[1] == 0.0) and s[0, 1] == 0.0 for s in steps)  # catheter still, no turning
    assert all(s[0, 0] == pytest.approx(40.0) for s in steps[:-1]) and 0.0 < steps[-1][0, 0] < 40.0
    np.testing.assert_allclose(intervention.inserted, [40.0, 0.0], atol=1e-9)
    assert np.all(intervention.last_action == 0.0)                      # no action shown as taken
    # a NaN in the opening ends it (stEVE has thrown the devices back; the env reports the error)
    intervention = OpeningIntervention(error_after=3)
    WireFirst(intervention).reset()
    assert len(intervention.calls) - 1 == 3


def test_shell_info_keys():
    from guidewire_rl.env.two_device.actions import STAGES, ShieldInfo
    shell, _ = make_shell(d=20.0, L_wire=100.0, L_cath=80.0)
    shell_step(shell, [0.5, -0.5, 0.25, 0.0])
    intervention = SimpleNamespace(velocity_limits=HIGH.reshape(2, 2),
                                   last_action=np.array([[10.0, 0.0], [5.0, 0.0]]))
    info = ShieldInfo(intervention, shell).info
    assert set(info) >= {"policy_action", "executed_action", "nominal_action", "shield_delta",
                         "nominal_available", "shield_penalized", "shield_stages", "shield_causes"}
    np.testing.assert_allclose(info["executed_action"], [0.25, 0.0, 0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(info["shield_delta"], info["executed_action"] - info["policy_action"])
    assert not info["nominal_available"] and np.all(info["nominal_action"] == 0)
    assert set(info["shield_causes"]) == set(STAGES) and sum(info["shield_causes"].values()) == 0
    # causes are told apart: here the hard rule and the same-length clamp in one step
    shell, _ = make_shell(branch_c="bct", d=-12.0, L_wire=188.0, L_cath=200.0)
    shell_step(shell, [1.0, 0.0, 0.8, 0.0])
    causes = ShieldInfo(intervention, shell).info["shield_causes"]
    assert causes["hard_rule"] == 1 and causes["force"] == 0 and causes["device_limit"] == 0
    # the suggestion reported is the advisor's current one (for the next step)
    advisor = SimpleNamespace(action=np.array([0.1, 0.2, 0.3, 0.4]))
    info = ShieldInfo(intervention, shell, nominal_advisor=advisor).info
    assert info["nominal_available"]
    np.testing.assert_allclose(info["nominal_action"], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)


class FakeController:
    """Stands in for the script (``NominalAdvisor`` builds it from its dotted path)."""

    def __init__(self, phase_detector, gain: float = 0.5):
        self.phase_detector, self.gain = phase_detector, gain

    def nominal_for_phase(self, state):
        return None if state.phase != 1 else [self.gain, 0.0, -2.0, 0.0]


def test_script_suggestion_reaches_the_box_and_the_observation():
    from guidewire_rl.env.two_device.actions import NominalAdvisor, TwoDeviceGuarded
    from guidewire_rl.env.two_device.observations import ScriptSuggestion
    referee = ShellReferee(phase=1, d=-40.0, L_wire=100.0, L_cath=140.0)
    advisor = NominalAdvisor(None, referee, controller={"type": "test_two_device_env.FakeController",
                                                       "gain": 0.4})
    advisor.reset()
    np.testing.assert_allclose(advisor.action, [0.4, 0.0, -1.0, 0.0])    # clipped to [-1, 1]
    suggestion = ScriptSuggestion(advisor)
    suggestion.step()
    np.testing.assert_allclose(suggestion.obs, [0.4, 0.0, -1.0, 0.0])
    shell = TwoDeviceGuarded(SimpleNamespace(fluoroscopy=SimpleNamespace(image_frequency=7.5)),
                             referee, labels=[SimpleNamespace(name="distal_force", last_value=0.0),
                                              SimpleNamespace(name="catheter_tip_force", last_value=0.0)],
                             nominal_advisor=advisor, residual_phases=[1])
    raw, record = shell_step(shell, [-1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(raw / HIGH, [-0.1, 0.5, -0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(record.nominal, [0.4, 0.0, -1.0, 0.0])
    referee.state.phase = 2
    advisor.step()
    assert advisor.action is None
    suggestion.step()
    assert np.all(suggestion.obs == 0.0)
    no_script = ScriptSuggestion()
    no_script.step()
    assert no_script.obs.shape == (4,) and np.all(no_script.obs == 0.0)


# --------------------------------------------------------------------- observations ----
def test_frame_history_shows_earlier_frames_as_seen():
    from guidewire_rl.env.two_device.observations import FrameHistory
    source = SimpleNamespace(obs=np.array([0.0, 0.0], np.float32),
                             space=SimpleNamespace(low=-np.ones(2), high=np.ones(2)))
    history = FrameHistory({"tip": source}, terms=["tip"], frames=2)
    history.reset()
    np.testing.assert_array_equal(history.obs, [0, 0, 0, 0])
    for t in (1, 2, 3):
        source.obs = np.array([t, -t], np.float32)
        history.step()
    np.testing.assert_array_equal(history.obs, [2, -2, 1, -1])      # t-1, then t-2
    with pytest.raises(KeyError):
        FrameHistory({"tip": source}, terms=["cath_tip"])


def test_observation_helpers():
    from guidewire_rl.env.two_device.observations import (InsertionDiff, PhaseOneHot,
                                                          sample_behind_2d)
    line = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    np.testing.assert_allclose(sample_behind_2d(line, 4, 4.0), [[4, 0], [8, 0], [10, 2], [10, 6]])
    np.testing.assert_allclose(sample_behind_2d(line, 2, 30.0), [[10, 10], [10, 10]])
    intervention = SimpleNamespace(instrument_lengths_inserted=[250.0, 100.0])
    term = InsertionDiff(intervention)
    term.step()
    assert term.obs[0] == 1.0                                       # 150 mm, clipped
    intervention.instrument_lengths_inserted = [100.0, 140.0]
    term.step()
    assert term.obs[0] == pytest.approx(-0.4)
    one_hot = PhaseOneHot(FakeReferee(made_state(phase=3)))
    one_hot.step()
    np.testing.assert_array_equal(one_hot.obs, [0, 0, 0, 1, 0])


# -------------------------------------------------------------------------- config ----
EXPECTED_TERMS = [("wire_tip", 2), ("wire_shape", 8), ("cath_tip", 2), ("cath_shape", 12),
                  ("target", 2), ("junctions", 8), ("inserted", 2), ("d", 1), ("rotation", 4),
                  ("last_actions", 12), ("suggestion", 4), ("phase", 5), ("history", 48)]


def test_config_follows_the_plan():
    config = load_config(CONFIG)
    env = config["env"]
    scene = env["scene"]
    assert [t["name"] for t in env["observation"]] == [n for n, _ in EXPECTED_TERMS]
    assert scene["vessel_tree"] == {"type": "BiroPhantomArch", "version": "rebuilt_cut"}
    wire, catheter = scene["instruments"]
    assert wire["type"] == "Angled" and "mass_density" not in wire    # default mass
    assert (wire["young_modulus"], wire["flex_young_modulus"], wire["tip_angle_deg"]) == (24900.0, 325.0, 72.0)
    assert catheter["type"] == "HookCatheter"
    assert scene["fluoroscopy"]["image_rot_zx"] == [61.0, -21.0]
    assert "dt_simulation" not in scene["simulation"] and "alarm_distance" not in scene["simulation"]
    assert scene["simulation"]["friction"] == 0.1
    assert scene["target"] == {"type": "RouteWindowTarget", "threshold": 5.0, "split": "train"}
    assert env["truncation"] == ["PhaseCutoff", {"type": "MaxSteps", "max_steps": 600}, "SimError"]
    assert "pathfinder" not in env
    assert [r["type"] for r in env["reward"]] == [
        "LeadingTipProgress", "CatheterToF", "HookPrepPotential", "PhasePotential", "WindowPenalty",
        "EventPenalty", "ShieldPenalty", "StepPenalty", "OutcomeReward"]
    assert [l["name"] for l in env["labels"]] == [
        "distal_force", "catheter_tip_force", "wire_tip_clearance", "catheter_tip_clearance"]
    action = env["action"]
    assert action["type"] == "TwoDeviceGuarded" and action["protector_mm"] == 2.5
    assert action["residual_phases"] == [] and action["force_limits_N"] == [1.0, 2.42]
    assert action["branch_push_max"] == 0.5 and "wire_branch_push_max" not in action
    evaluation = copy.deepcopy(config)
    apply_override(evaluation, "env.scene.target.split=eval")
    assert evaluation["env"]["scene"]["target"]["split"] == "eval"


@pytest.mark.upstream
def test_every_part_resolves():
    from guidewire_rl.env import registry
    names = {"monitor": ["PhaseDetector", "NominalAdvisor", "WallDistance"], "action": ["TwoDeviceGuarded"],
             "target": ["RouteWindowTarget"], "instrument": ["HookCatheter"],
             "observation": ["InstrumentTip2D", "InstrumentShape2D", "TaskTarget2D", "JunctionMap2D",
                             "PhaseOneHot", "InsertionDiff", "InsertionLengthsNormalized",
                             "ScriptSuggestion", "FrameHistory"],
             "observation_wrapper": ["TipNoise2D", "Normalize2DTask"],
             "reward": ["LeadingTipProgress", "CatheterToF", "HookPrepPotential", "PhasePotential",
                        "WindowPenalty", "EventPenalty", "ShieldPenalty", "StepPenalty", "OutcomeReward"],
             "terminal": ["PlatformReached", "HardFailure"], "truncation": ["PhaseCutoff", "WallBreach"],
             "info": ["PhaseInfo", "ShieldInfo", "WallInfo"],
             "label": ["InstrumentTipForce", "CatheterTipForce", "TipClearance"]}
    for kind, types in names.items():
        for name in types:
            cls = registry.resolve(kind, name)
            assert cls.__name__ == name and cls.__module__.startswith("guidewire_rl.env.")


@pytest.mark.upstream
def test_hook_catheter_is_the_labs():
    from eve.intervention.instrument import MeshSection, StraightSection
    from guidewire_rl.env.instruments.hook import HookCatheter
    catheter = HookCatheter()
    assert type(catheter.base_section) is StraightSection and isinstance(catheter.tip_section, MeshSection)
    assert catheter.length == 450.0
    assert catheter.tip_section.length == pytest.approx(20 + 8 * np.radians(130) + 8, abs=0.05)
    assert catheter.base_section.length + catheter.tip_section.length == pytest.approx(450.0)
    assert catheter.hook_length == pytest.approx(26.15, abs=0.01)
    for section in (catheter.base_section, catheter.tip_section):
        assert (section.diameter_outer, section.diameter_inner, section.young_modulus,
                section.mass_density) == (2.0, 1.0, 4e3, 5e-6)
    assert catheter.tip_section.collis_edges_per_mm == 1.0
    assert catheter.base_section.collis_edges_per_mm == 0.1


# ---------------------------------------------------------------------------- SOFA ----
@pytest.fixture(scope="module")
def sofa_env():
    from guidewire_rl.env import build_env
    env = build_env(load_config(CONFIG)["env"])
    yield env
    env.close()


@pytest.mark.sofa
@pytest.mark.upstream
def test_env_builds_with_the_plans_observation(sofa_env):
    env = sofa_env
    assert env.observation_layout == EXPECTED_TERMS
    assert env.observation_space.shape == (110,)
    assert env.label_names == ["distal_force", "catheter_tip_force", "wire_tip_clearance",
                               "catheter_tip_clearance"]
    observation, info = env.reset(seed=2, options={"cutoff": 3})
    assert info["cutoff"] == 3 and info["phase"] == 0
    env.set_cutoff(5)
    observation, info = env.reset(seed=2)
    assert info["cutoff"] == 5
    assert np.all(np.abs(observation) <= 1.0)
    keys = {"phase", "max_phase", "events", "is_success", "policy_action", "executed_action",
            "nominal_action", "shield_delta", "d", "cutoff", "sim_error", "shield_causes"}
    actions = [[0.8, 0.2, 0.0, 0.0]] * 4 + [[0.6, -0.3, 0.6, 0.4]] * 4
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(np.asarray(action, np.float32))
        assert keys <= set(info)
        assert np.all(np.isfinite(observation)) and np.isfinite(reward)
        # LastAction reads what was executed (after the shell), newest first
        start = sum(size for name, size in EXPECTED_TERMS[:9])
        np.testing.assert_allclose(observation[start:start + 4], info["executed_action"], atol=1e-5)
        assert not terminated and not truncated
    wire, catheter = env.intervention.instrument_lengths_inserted
    assert 0 < catheter < wire


@pytest.mark.sofa
@pytest.mark.upstream
def test_tip_force_of_the_leading_device_is_g13s_distal_force():
    """With the catheter ahead, its tip force is the combined beam's distal force (DistalForce);
    the wire's is measured further back."""
    from guidewire_rl.env import build_env
    config = load_config(CONFIG)
    config["env"]["labels"].append({"name": "lead_distal_force", "type": "DistalForce",
                                    "statistic": "max", "distal_length_mm": 3.0})
    env = build_env(config["env"])
    try:
        env.reset(seed=5)
        touched = 0
        for k in range(45):
            action = [0.0, 0.0, 1.0, 0.0] if k else [0.0, 0.0, 1.0, 0.0]
            _, _, terminated, truncated, info = env.step(np.asarray(action, np.float32))
            labels = info["labels"]
            assert labels["catheter_tip_force"] == pytest.approx(labels["lead_distal_force"], abs=1e-9)
            touched += labels["catheter_tip_force"] > 0
            if terminated or truncated:
                break
        assert touched > 0
    finally:
        env.close()


@pytest.mark.sofa
@pytest.mark.upstream
def test_scripted_push_reaches_the_hook_phase(sofa_env):
    """Both pushed over the arch, the wire 15+ mm ahead: once the catheter is 30 mm past O the
    referee says P1; nothing fails on the way."""
    env = sofa_env
    referee = env.monitors["phase_detector"]
    env.reset(seed=3)
    phases = []
    for _ in range(130):
        d = referee.state.d
        action = np.array([1.0, 0.0, 1.0 if d > 15 else 0.0, 0.0], np.float32)
        _, _, terminated, truncated, info = env.step(action)
        phases.append(info["phase"])
        assert not terminated and not truncated
        assert info["shield_causes"]["speed"] == 0          # the config's limits are the shell's
        if info["phase"] == 1:
            break
    assert phases[-1] == 1 and set(phases) == {0, 1}
    assert referee.state.aorta_pos_c >= 30.0
