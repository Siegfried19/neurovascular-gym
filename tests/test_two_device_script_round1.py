"""The first training round's two-device script (guidewire_rl/env/two_device/script_round1.py).

This file is tests/test_two_device_script.py as of commit e908bee (2026-10-31), pointed at the kept
copy of that version of the script, plus one check that the copy is still that version word for
word. Without SOFA: the geometry it reads with, the guard that keeps the two tips out of the
same-length band, every stage change on hand-made readouts, and the P1 suggestion. With SOFA
(``sofa``): one whole episode on ``configs/two_devices_round1.yaml`` (heavy default masses, what this
version was tuned on) from the insertion point to the referee's success, passing the five phases in
order.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.env.two_device import script_round1 as S

SPEC = WORKSPACE_ROOT / "guidewire_rl" / "env" / "vessels" / "biro_phantom" / "rebuilt_cut.json"
CONFIG = WORKSPACE_ROOT / "configs" / "two_devices_round1.yaml"


# ------------------------------------------------------------------- a vessel tree without stEVE ----
class _Branch:
    def __init__(self, name, coordinates, radii):
        self.name = name
        self.coordinates = np.asarray(coordinates, float)
        self.radii = np.asarray(radii, float)


class _Insertion:
    def __init__(self, position):
        self.position = np.asarray(position, float)


class FakeTree:
    """The rebuilt phantom's centerlines with the bits of stEVE's vessel tree the script reads."""

    def __init__(self):
        spec = json.loads(SPEC.read_text())
        self.branches = [_Branch(k, v["coordinates"], v["radii"]) for k, v in spec["branches"].items()]
        self._by = {b.name: b for b in self.branches}
        self.insertion = _Insertion(spec["insertion"]["position"])

    def keys(self):
        return tuple(self._by)

    def __getitem__(self, name):
        return self._by[name]


@pytest.fixture(scope="module")
def anatomy():
    return S.Anatomy(FakeTree())


def controller(anatomy, **params):
    ctl = S.StagedController(S.ScriptParams(**params))
    ctl.anatomy = anatomy
    ctl.dt, ctl.shell_dt = 0.132, 1.0 / 7.5
    ctl.limits = np.array([40.0, np.pi, 25.0, np.pi / 2])
    return ctl


def readout(anatomy, **kw):
    line = np.zeros((60, 3))
    r = S.Readout(wire=kw.pop("wire", line), cath=kw.pop("cath", line),
                  L=np.asarray(kw.pop("L", [100.0, 100.0]), float), rot=np.zeros(2),
                  target=kw.pop("target", anatomy.rcca_point(45.0)))
    r.target_depth = kw.pop("target_depth", 45.0)
    for key, value in kw.items():
        assert hasattr(r, key), key
        setattr(r, key, value)
    if "d" not in kw:
        r.d = float(r.L[0] - r.L[1])
    return r


def step(ctl, r):
    """One controller step on a readout, guard included (as ``act`` does it)."""
    ctl.steps += 1
    v_w, w_w, v_c, w_c = ctl._policy(r)
    v_w, v_c = ctl._cross(r, v_w, v_c)
    return v_w, w_w, v_c, w_c


# -------------------------------------------------------------------------------- geometry ----
def test_project_and_point_at_on_a_bent_line():
    line = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], float)
    s, d = S.project(line, np.array([10.0, 4.0, 3.0]))
    assert s == pytest.approx(14.0) and d == pytest.approx(3.0)
    assert np.allclose(S.point_at(line, 12.0), [10, 2, 0])
    assert np.allclose(S.point_at(line, 99.0), [10, 10, 0])        # clamped


def test_split_counts_the_shorter_instrument_back_from_the_leading_tip():
    beam = np.c_[np.linspace(100, 0, 101), np.zeros(101), np.zeros(101)]   # tip first, 100 mm
    wire, cath = S.split(beam, [92.0, 80.0])          # polyline 8 mm shorter than the wire's length
    assert np.allclose(wire[0], [100, 0, 0])          # the leader's tip is the beam's tip
    assert np.allclose(cath[0], [88, 0, 0])           # 12 mm back from it, not 80 mm from the base


def test_bend_angle_sign_and_degenerate_cases():
    shaft, base = np.array([0.0, 0, -15]), np.array([0.0, 0, 0])
    tip = np.array([5.0, 0, 3])                       # bent towards +x
    assert S.bend_angle(tip, base, shaft, np.array([5.0, 0, 0])) == pytest.approx(0.0, abs=1e-9)
    assert S.bend_angle(tip, base, shaft, np.array([0.0, 5, 0])) == pytest.approx(np.pi / 2)
    assert S.bend_angle(np.array([0.0, 0, 3]), base, shaft, np.array([1.0, 0, 0])) == 0.0   # straight


def test_anatomy_of_the_rebuilt_phantom(anatomy):
    m = anatomy.bct_mouth
    assert m.a0 == pytest.approx(19.7, abs=0.5) and m.s_parent == pytest.approx(183.9, abs=1.0)
    assert anatomy.bct_beyond_mouth == pytest.approx(40.0, abs=0.5)
    assert anatomy.rcca_task_len == pytest.approx(37.7, abs=0.2)
    p = anatomy.rcca_point(45.0)                       # in rcca_rest, 7.3 mm past rcca's end
    assert anatomy.vessel_of(p) == "rcca" and anatomy.depth_rcca(p)[0] == pytest.approx(45.0, abs=0.1)
    assert anatomy.vessel_of(anatomy.aorta[60]) == "aorta"
    assert anatomy.vessel_of(m.centre + 8.0 * m.axis) == "bct"


# ------------------------------------------------------------------ same-length band guard ----
def test_guard_never_ends_a_step_in_the_band_and_holds_the_catheter_for_a_crossing(anatomy):
    ctl = controller(anatomy)
    T, band = ctl.shell_dt, ctl.p.cross_band_mm
    for d0 in np.linspace(-12, 12, 49):
        if abs(d0) < band + ctl.p.cross_edge_margin_mm:
            continue
        for v_w in (-40.0, -20.0, -4.0, 0.0, 4.0, 20.0, 40.0):
            for v_c in (-8.0, 0.0, 3.0, 8.0):
                r = readout(anatomy, L=[100.0 + d0, 100.0])
                w, c = ctl._cross(r, v_w, v_c)
                d1 = d0 + (w - c) * T
                assert abs(d1) >= band, (d0, v_w, v_c, w, c, d1)
                assert abs(w) <= ctl.p.v_wire_max + 1e-9
                if (w, c) != (v_w, v_c) and np.sign(d1) != np.sign(d0):
                    assert c == 0.0                    # a crossing it makes holds the catheter still


def test_guard_crosses_in_one_step_from_the_band_edge(anatomy):
    ctl = controller(anatomy)
    r = readout(anatomy, L=[100.0 - 2.65, 100.0])    # wire just inside the catheter, wants out
    w, c = ctl._cross(r, 30.0, -3.0)
    assert c == 0.0 and (r.d + (w - c) * ctl.shell_dt) >= ctl.p.cross_band_mm


# ------------------------------------------------------------------------ stage changes ----
def test_approach_hands_over_past_the_mouth(anatomy):
    ctl = controller(anatomy)
    s_o = anatomy.bct_mouth.s_parent
    r = readout(anatomy, L=[200.0, 160.0], s_cath=s_o + 20.0, vessel_cath="aorta")
    v_w, _, v_c, _ = step(ctl, r)
    assert ctl.stage == "approach" and v_c == ctl.p.v_approach and v_w > 0
    r = readout(anatomy, L=[310.0, 270.0], s_cath=s_o + 34.0, vessel_cath="aorta")
    v_w, _, v_c, _ = step(ctl, r)
    assert (ctl.stage, ctl.substage) == ("place", "retract") and v_w == -ctl.p.v_retract


def test_approach_takes_the_wire_back_out_of_a_side_vessel(anatomy):
    ctl = controller(anatomy)
    m = anatomy.mouths["lcca"]
    tip = m.centre + 10.0 * m.axis
    wire = np.repeat(tip[None], 60, axis=0)
    r = readout(anatomy, L=[150.0, 110.0], wire=wire, vessel_wire="lcca", s_cath=120.0)
    v_w, _, v_c, _ = step(ctl, r)
    assert ctl.substage == "redirect" and v_w < 0 and v_c == 0.0


def test_place_turn_pull_seat(anatomy):
    ctl = controller(anatomy)
    ctl._go("place", "retract")
    p, s_o = ctl.p, anatomy.bct_mouth.s_parent
    step(ctl, readout(anatomy, L=[230.0, 268.0]))     # d = -38: wire is behind the hook
    assert (ctl.stage, ctl.substage) == ("place", "")
    # from beyond: first the overshoot point, then back to mouth + offset, hook down
    r = readout(anatomy, L=[230.0, 267.0], s_base=s_o + p.offsets_mm[0] + p.over_mm, phi_up=np.pi)
    step(ctl, r)
    assert ctl._over
    r = readout(anatomy, L=[230.0, 267.0], s_base=s_o + p.offsets_mm[0], phi_up=np.pi, phi_down=0.0)
    step(ctl, r)
    assert ctl.stage == "turn"
    for _ in range(p.turn_hold):
        step(ctl, readout(anatomy, L=[230.0, 267.0], phi_up=0.05))
    assert ctl.stage == "pull"
    _, _, v_c, _ = step(ctl, readout(anatomy, L=[230.0, 267.0], vessel_cath="aorta", depth_bct_cath=-3.0))
    assert v_c == -p.v_pull
    step(ctl, readout(anatomy, L=[230.0, 266.0], vessel_cath="bct", depth_bct_cath=3.0))
    assert ctl.stage == "seat"
    for _ in range(p.seat_hold):
        step(ctl, readout(anatomy, L=[230.0, 265.0], vessel_cath="bct", depth_bct_cath=p.seat_mm + 1))
    assert (ctl.stage, ctl.substage) == ("wire", "out")


def test_a_turn_that_times_out_places_the_hook_again_then_pulls_where_it_settled(anatomy):
    ctl = controller(anatomy)
    p = ctl.p
    settled = np.radians(-50.0)                       # wound-up shaft: the hook stays 50 deg off
    ctl._go("turn", "", 260.0)
    for _ in range(p.turn_timeout + 2):
        step(ctl, readout(anatomy, L=[223.0, 260.0], phi_up=settled))
    assert (ctl.stage, ctl.tries) == ("place", 1)     # a new try: hook down, next place
    assert ctl.offset == p.offsets_mm[1] and ctl.aim == pytest.approx(np.radians(p.aims_deg[1]))
    # no tries left: pull with the aim where it settled (clipped)
    ctl.tries = len(p.offsets_mm) - 1
    ctl._go("turn", "", 260.0)
    for _ in range(p.turn_timeout + 2):
        step(ctl, readout(anatomy, L=[223.0, 260.0], phi_up=settled))
    assert ctl.stage == "pull"
    assert ctl.aim == pytest.approx(np.radians(-p.adapt_aim_deg))
    _, _, _, w_c = step(ctl, readout(anatomy, L=[223.0, 260.0], phi_up=settled, vessel_cath="aorta"))
    assert w_c == pytest.approx(0.0, abs=0.5)          # no longer forced back towards the first aim


def test_a_pull_that_finds_nothing_or_another_vessel_starts_the_next_try(anatomy):
    ctl = controller(anatomy)
    ctl._go("pull", "", 250.0)
    step(ctl, readout(anatomy, L=[212.0, 250.0 - ctl.p.pull_max_mm - 1], vessel_cath="aorta"))
    assert (ctl.stage, ctl.tries) == ("place", 1) and ("place:lower", ctl.steps) in ctl.log
    assert ctl.offset == ctl.p.offsets_mm[1]          # the next place along the aorta
    ctl = controller(anatomy)
    ctl._go("pull", "", 250.0)
    m = anatomy.mouths["lcca"]
    cath = np.repeat((m.centre + 6.0 * m.axis)[None], 60, axis=0)
    step(ctl, readout(anatomy, L=[212.0, 249.0], cath=cath, vessel_cath="lcca"))
    assert (ctl.stage, ctl.substage) == ("pull", "free")


def test_wire_out_climb_fork_up_follow(anatomy):
    ctl = controller(anatomy)
    p = ctl.p
    ctl._go("wire", "out", 230.0)
    v_w, _, v_c, _ = step(ctl, readout(anatomy, L=[200.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0))
    assert v_w > 0 and v_c == 0.0
    lead = p.out_lead_mm
    step(ctl, readout(anatomy, L=[230.0 + lead, 230.0], vessel_cath="bct", depth_bct_cath=20.0))
    assert ctl.substage == "climb"
    v_w, _, v_c, _ = step(ctl, readout(anatomy, L=[230.0 + lead, 230.0], vessel_cath="bct", depth_bct_cath=20.0))
    assert v_c == p.v_climb and v_w == pytest.approx(v_c)            # both up, d kept
    step(ctl, readout(anatomy, L=[238.0 + lead, 238.0], vessel_cath="bct", depth_bct_cath=p.climb_depth_mm + 1))
    assert ctl.substage == "fork"
    _, w_w, _, _ = step(ctl, readout(anatomy, L=[270.0, 240.0], vessel_cath="bct", depth_bct_cath=26.0,
                                     wire_branch="bct", depth_bct_wire=36.0))
    assert w_w != 0.0                                                # turning while it probes the fork
    step(ctl, readout(anatomy, L=[274.0, 240.0], vessel_cath="bct", depth_bct_cath=26.0,
                      wire_branch="rsa", wire_branch_depth=p.rsa_back_mm + 1, depth_bct_wire=40.0))
    assert ctl.substage == "backoff"
    for _ in range(p.backoff_steps + 1):
        step(ctl, readout(anatomy, L=[266.0, 240.0], vessel_cath="bct", depth_bct_cath=26.0,
                          wire_branch="bct", depth_bct_wire=34.0))
    assert ctl.substage == "fork"
    step(ctl, readout(anatomy, L=[280.0, 240.0], vessel_cath="bct", depth_bct_cath=26.0,
                      wire_branch="rcca", wire_branch_depth=p.in_rcca_mm + 1))
    assert ctl.substage == "up"
    step(ctl, readout(anatomy, L=[320.0, 240.0], vessel_cath="bct", depth_bct_cath=26.0,
                      wire_branch="rcca", wire_branch_depth=45.0 + p.wire_extra_mm + 0.5))
    assert ctl.stage == "follow"


def test_catheter_lost_while_the_wire_works_sends_the_wire_back_behind_the_hook(anatomy):
    ctl = controller(anatomy)
    ctl._go("wire", "fork", 240.0)
    v_w, _, _, _ = step(ctl, readout(anatomy, L=[270.0, 240.0], vessel_cath="aorta", depth_bct_cath=-5.0,
                                     wire_branch="bct", depth_bct_wire=20.0))
    assert (ctl.stage, ctl.substage, ctl.tries) == ("place", "rewire", 1) and v_w < 0


def test_follow_brakes_on_the_slip_and_trims_to_success(anatomy):
    ctl = controller(anatomy)
    p = ctl.p
    ctl._go("follow", "", 260.0)
    base = dict(L=[330.0, 280.0], target_depth=45.0, wire_branch="rcca", wire_branch_depth=40.0)
    _, _, v_c, _ = step(ctl, readout(anatomy, vessel_cath="bct", depth_bct_cath=20.0, **base))
    assert v_c == p.v_follow
    for depth in (25.0, 30.0, 33.0, 34.5, 35.5):                      # moving as pushed
        _, _, v_c, _ = step(ctl, readout(anatomy, vessel_cath="bct", depth_bct_cath=depth, **base))
    assert ctl.substage == "" and v_c == p.v_follow_slow              # slow near the trunk's end
    _, _, v_c, _ = step(ctl, readout(anatomy, vessel_cath="rcca", depth_bct_cath=40.0, depth_rcca_cath=12.0,
                                     **base))
    assert ctl.substage == "brake"
    for _ in range(p.brake_min_steps + 1):
        _, _, v_c, _ = step(ctl, readout(anatomy, vessel_cath="rcca", depth_rcca_cath=20.0, **base))
    assert ctl.substage == "trim"
    target = anatomy.rcca_point(45.0)
    wire = np.repeat(target[None], 60, axis=0)
    ok = dict(L=[330.0, 305.0], target=target, target_depth=45.0, wire=wire, wire_branch="rcca",
              wire_branch_depth=45.0, vessel_cath="rcca", depth_rcca_cath=20.0)
    for _ in range(p.success_hold):
        out = step(ctl, readout(anatomy, **ok))
    assert ctl.stage == "done" and ctl.success and out == (0.0, 0.0, 0.0, 0.0)


def test_nominal_for_phase_is_zero_outside_p1_and_does_not_advance(anatomy):
    ctl = controller(anatomy)

    class State:
        phase = 3

    assert np.all(ctl.nominal_for_phase(State()) == 0.0)
    assert np.all(ctl.nominal_for_phase(None, 0) == 0.0)
    assert ctl.stage == "approach" and ctl.steps == 0


class _FakeSim:
    def __init__(self, dofs, inserted):
        self.dof_positions = dofs
        self.inserted_lengths = inserted
        self.rotations = [0.0, 0.0]
        self.dt_simulation = 0.006


class _FakeIntervention:
    """Enough of stEVE's intervention for ``read``: both instruments laid along the aorta."""

    def __init__(self, anatomy, L_wire, L_cath):
        tree = FakeTree()
        self.vessel_tree = tree
        s = np.arange(0.0, max(L_wire, L_cath) + 1.0, 1.0)[::-1]          # tip first
        dofs = np.stack([S.point_at(anatomy.aorta, x) for x in s])
        self.simulation = _FakeSim(dofs, [L_wire, L_cath])

        class Fl:
            image_frequency = 7.5

        class Tg:
            coordinates_vessel = anatomy.rcca_point(45.0)

        self.fluoroscopy, self.target = Fl(), Tg()
        self.velocity_limits = np.array([[40.0, np.pi], [25.0, np.pi / 2]])


def test_nominal_for_phase_in_p1_takes_the_wire_back_behind_the_hook(anatomy):
    intervention = _FakeIntervention(anatomy, L_wire=230.0, L_cath=220.0)    # wire 10 mm ahead
    ctl = S.StagedController(intervention=intervention)
    ctl.reset()                                           # NominalAdvisor's way: no env argument

    class State:
        phase = 1

    nominal = ctl.nominal_for_phase(State())
    assert nominal.shape == (4,) and nominal[0] < 0.0      # wire back
    assert ctl.stage == "approach" and ctl.steps == 0


def test_act_reads_a_fake_scene_and_starts_with_the_wire(anatomy):
    intervention = _FakeIntervention(anatomy, L_wire=0.0, L_cath=0.0)
    ctl = S.StagedController()
    ctl.reset(intervention)
    action = ctl.act(intervention)
    assert action.shape == (4,) and action.dtype == np.float32
    assert action[0] == pytest.approx(1.0) and action[2] == 0.0    # wire first, at its limit


# --------------------------------------------------------------------- a whole episode ----
@pytest.mark.sofa
@pytest.mark.upstream
@pytest.mark.xfail(reason="out of date since round two (2026-11-10): the referee's success is the two tiers now; "
                          "measured 2026-11-13: the script still walks all its stages to 'done' (phase 4) on "
                          "both seeds, the referee no longer calls it a success. Left for the user to decide: "
                          "assert the stages only, or drop this old version", strict=False)
def test_whole_episode_from_the_insertion_point_to_success():
    """configs/two_devices_round1.yaml from the insertion point: the referee's success, its five
    phases in order, the script's seven stages in order. Runs are not repeatable bit for bit
    (multithreaded collision detection), so a second seed is tried before failing."""
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env

    env = build_env(load_config(CONFIG)["env"])
    try:
        for seed in (5, 2):
            ctl = S.StagedController()
            obs, info = env.reset(seed=seed)
            ctl.reset(env)
            phases = [int(info["phase"])]
            for _ in range(450):
                obs, reward, terminated, truncated, info = env.step(ctl.act(env))
                assert np.all(np.isfinite(obs)) and np.isfinite(reward)
                phases.append(int(info["phase"]))
                if terminated or truncated:
                    break
            if info["is_success"]:
                break
        assert info["is_success"] and terminated and not info.get("hard_failure")
        firsts = [phases.index(k) for k in range(5)]
        assert firsts == sorted(firsts)
        stages = [name.split(":")[0] for name, _ in ctl.log]
        order = [stages.index(s) for s in S.STAGES[:-1]]
        assert order == sorted(order)
    finally:
        env.close()


# ------------------------------------------------------------------- the kept copy is the original ----
ROUND1_SHA256 = "6eea8afdc8dbdc6b8bbb6ed9f85e9d39ecbce642d6998ff6d007a16c85361b25"   # sha256 of guidewire_rl/env/two_device/script.py at commit e908bee
ROUND1_HEADER_LINES = 4                                                          # the copy's note on top


def test_the_copy_is_the_first_rounds_script_word_for_word():
    path = WORKSPACE_ROOT / "guidewire_rl" / "env" / "two_device" / "script_round1.py"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert all(line.startswith("#") for line in lines[:ROUND1_HEADER_LINES])
    body = "".join(lines[ROUND1_HEADER_LINES:]).encode("utf-8")
    assert hashlib.sha256(body).hexdigest() == ROUND1_SHA256
