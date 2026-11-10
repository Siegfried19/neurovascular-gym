"""The two-device script (guidewire_rl/env/two_device/script.py).

Without SOFA: the geometry it reads with (the carotid's split and the aiming directions
included), the guard that keeps the two tips out of the same-length band, every stage change on
hand-made readouts, and the P1 suggestion. With SOFA (``sofa``): one whole episode on
``configs/two_devices_ica.yaml`` from the insertion point to the referee's success in the internal
carotid, passing the five phases in order (about 10 minutes at 1 s per step).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.env.two_device import script as S

SPEC = WORKSPACE_ROOT / "guidewire_rl" / "env" / "vessels" / "biro_phantom" / "rebuilt_cut.json"
CONFIG = WORKSPACE_ROOT / "configs" / "two_devices_ica.yaml"


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


def test_anatomy_knows_the_carotid_split_and_where_to_aim(anatomy):
    """The ICA/ECA split (``v8``) 90.9 mm up the carotid chain, the downward branch (``v7``) at
    63.8 mm; points labelled by branch; the aiming directions: away from the subclavian at the
    trunk's end (lateral 6.7 mm off), the ICA's side at the split (the ECA leaves the other way)."""
    an = anatomy
    assert an.split_mm == pytest.approx(90.9, abs=0.3) and an.down_mm == pytest.approx(63.8, abs=0.3)
    assert an.carotid_of(an.rcca_point(50.0))[0] == "cca"
    label, side, chain = an.carotid_of(an.rcca_point(an.split_mm + 5.0))
    assert label == "ica" and side == pytest.approx(5.0, abs=0.1) and chain == pytest.approx(an.split_mm + 5.0, abs=0.1)
    assert an.carotid_of(S.point_at(an.eca, 10.0))[:2] == ("eca", pytest.approx(10.0, abs=0.1))
    assert an.carotid_of(S.point_at(an.down, 8.0))[:2] == ("down", pytest.approx(8.0, abs=0.1))
    assert an.carotid_of(an.aorta[60])[0] == "aorta"
    axis_end = S.unit(an.bct[-1] - S.point_at(an.bct, an._cb[-1] - 8.0, an._cb))
    rsa_side = S.lateral(S.point_at(an.rsa, 10.0, an._cs), an.bct[-1], axis_end)
    assert np.dot(an.lat_trunk, rsa_side) < -0.9                 # away from the subclavian
    o = an.rcca_point(an.split_mm)
    axis_split = S.unit(o - an.rcca_point(an.split_mm - 8.0))
    eca_side = S.lateral(S.point_at(an.eca, 10.0), o, axis_split)
    assert np.dot(an.lat_ica, eca_side) < -0.9                   # the ICA's side, opposite the ECA's


def test_aim_angle_sign_follows_a_right_handed_turn_about_the_shaft():
    """A J bend towards +x on a shaft pointing +z needs +90 deg to point at +y (the wire's
    positive turn command turns the bend that way); no bend -> None."""
    wire = np.array([[0.0, 0.0, 0.0]] + [[0.0, 0.0, -k] for k in range(1, 40)])
    wire[:6, 0] = [2.0, 1.6, 1.2, 0.8, 0.4, 0.0]
    assert np.degrees(S.aim_angle(wire, np.array([0.0, 1.0, 0.0]))) == pytest.approx(90.0, abs=1.0)
    assert np.degrees(S.aim_angle(wire, np.array([0.0, -1.0, 0.0]))) == pytest.approx(-90.0, abs=1.0)
    straight = np.array([[0.0, 0.0, -k] for k in range(40)])
    assert S.aim_angle(straight, np.array([1.0, 0.0, 0.0])) is None


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
        step(ctl, readout(anatomy, L=[230.0, 267.0], phi_up=np.radians(p.aims_deg[0]) + 0.05))
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
    settled = np.radians(-70.0)                       # wound-up shaft: the hook stays 40 deg off the aim
    ctl._go("turn", "", 260.0)
    for _ in range(p.turn_timeout + 2):
        step(ctl, readout(anatomy, L=[223.0, 260.0], phi_up=settled))
    assert (ctl.stage, ctl.tries) == ("place", 1)     # a new try: hook down, next place
    assert ctl.offset == p.offsets_mm[1] and ctl.aim == pytest.approx(np.radians(p.aims_deg[1]))
    # no tries left: pull with the aim where it settled (clipped), here on the other side of the aim
    ctl.tries = len(p.offsets_mm) - 1
    ctl._go("turn", "", 260.0)
    settled = np.radians(60.0)
    for _ in range(p.turn_timeout + 2):
        step(ctl, readout(anatomy, L=[223.0, 260.0], phi_up=settled))
    assert ctl.stage == "pull"
    assert ctl.aim == pytest.approx(np.radians(p.adapt_aim_deg))
    _, _, _, w_c = step(ctl, readout(anatomy, L=[223.0, 260.0], phi_up=settled, vessel_cath="aorta"))
    assert abs(w_c) < abs(ctl._turn_rate(S.wrap(settled - np.radians(p.aims_deg[0]))))   # pulled towards where it settled, not the first aim


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


def test_wire_out_aim_burst_rsa_retreat_up(anatomy):
    """Wire out to 16 mm ahead (catheter still), the aim stage (skipped on a bend it cannot read),
    25 mm/s bursts, one step back on a stall, the subclavian -> retreat into the trunk, the carotid
    -> up at 10 mm/s with the catheter only re-seated."""
    ctl = controller(anatomy)
    p = ctl.p
    ctl._go("wire", "out", 230.0)
    v_w, _, v_c, _ = step(ctl, readout(anatomy, L=[200.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0))
    assert v_w > 0 and v_c == 0.0
    step(ctl, readout(anatomy, L=[230.0 + p.out_lead_mm, 230.0], vessel_cath="bct", depth_bct_cath=20.0))
    assert ctl.substage == "fork" and ("aim:skip", ctl.steps) in ctl.log      # zeros: no bend to aim
    trunk = dict(vessel_cath="bct", depth_bct_cath=20.0, wire_branch="bct", depth_bct_wire=30.0)
    v_w, w_w, v_c, _ = step(ctl, readout(anatomy, L=[247.0, 230.0], **trunk))
    assert v_w == p.v_fork and v_c == 0.0
    for k in range(p.fork_stall_steps):                                      # the tip gains nothing
        v_w, _, _, _ = step(ctl, readout(anatomy, L=[248.0 + k, 230.0], **trunk))
    assert v_w == -p.v_backoff and ("fork:stall", ctl.steps) in ctl.log and ctl.substage == "backoff"
    v_w, _, _, _ = step(ctl, readout(anatomy, L=[248.0, 230.0], **trunk))
    assert ctl.substage == "fork" and v_w == p.v_fork                          # one step back, then again
    v_w, _, _, _ = step(ctl, readout(anatomy, L=[252.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0,
                                     wire_branch="rsa", wire_branch_depth=p.rsa_back_mm + 1))
    assert ctl.substage == "rsaback" and v_w < 0
    step(ctl, readout(anatomy, L=[246.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0,
                      wire_branch="bct", depth_bct_wire=p.rsa_retreat_to_mm - 2.0))
    assert ctl.substage == "fork"                                             # back in the trunk: again
    ctl._go("wire", "rsaback", 230.0)                                          # a retreat that swings into the carotid
    step(ctl, readout(anatomy, L=[256.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0,
                      wire_branch="rcca", wire_branch_depth=p.in_rcca_mm + 10.0))
    assert ctl.substage == "up"
    ctl._go("wire", "fork", 230.0)
    step(ctl, readout(anatomy, L=[270.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0,
                      wire_branch="rcca", wire_branch_depth=p.in_rcca_mm + 1))
    assert ctl.substage == "up"
    v_w, _, v_c, _ = step(ctl, readout(anatomy, L=[280.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0,
                                       wire_branch="rcca", wire_branch_depth=20.0))
    assert v_w == p.v_rcca and v_c == 0.0
    v_w, _, v_c, _ = step(ctl, readout(anatomy, L=[280.0, 230.0], vessel_cath="bct", depth_bct_cath=5.0,
                                       wire_branch="rcca", wire_branch_depth=22.0))
    assert v_c == -p.v_reseat                                                  # catheter slipping: slow pull


def test_a_burst_that_pulls_the_catheter_back_backs_off_and_a_deep_loss_recovers(anatomy):
    ctl = controller(anatomy)
    p = ctl.p
    ctl._go("wire", "fork", 230.0)
    trunk = dict(vessel_cath="bct", wire_branch="bct")
    step(ctl, readout(anatomy, L=[247.0, 230.0], depth_bct_cath=20.0, depth_bct_wire=28.0, **trunk))
    v_w, _, _, _ = step(ctl, readout(anatomy, L=[250.0, 230.0], depth_bct_cath=20.0 - p.fork_cath_drop_mm - 1,
                                     depth_bct_wire=31.0, **trunk))
    assert v_w == -p.v_backoff and ("fork:cath_drop", ctl.steps) in ctl.log
    # a dip to -8 mm during the bursts is not a loss (fork_lost_mm); -13 mm is; up the carotid -7 is
    ctl = controller(anatomy)
    ctl._go("wire", "fork", 230.0)
    step(ctl, readout(anatomy, L=[260.0, 230.0], vessel_cath="aorta", depth_bct_cath=p.lost_mm - 2.0, wire_branch="bct", depth_bct_wire=36.0))
    assert ctl.stage == "wire"
    v_w, _, _, _ = step(ctl, readout(anatomy, L=[260.0, 230.0], vessel_cath="aorta", depth_bct_cath=p.fork_lost_mm - 1.0,
                                     wire_branch="bct", depth_bct_wire=36.0))
    assert (ctl.stage, ctl.substage, ctl.tries) == ("place", "rewire", 1) and v_w < 0
    ctl = controller(anatomy)
    ctl._go("wire", "up", 230.0)
    step(ctl, readout(anatomy, L=[280.0, 230.0], vessel_cath="aorta", depth_bct_cath=p.lost_mm - 1.0,
                      wire_branch="rcca", wire_branch_depth=20.0))
    assert (ctl.stage, ctl.substage, ctl.tries) == ("place", "rewire", 1)


def test_the_script_gives_up_after_two_catheter_losses_and_then_sends_nothing(anatomy):
    ctl = controller(anatomy)
    p = ctl.p
    lost = dict(L=[260.0, 230.0], vessel_cath="aorta", depth_bct_cath=p.lost_mm - 1.0, wire_branch="bct",
                depth_bct_wire=36.0)
    ctl._go("wire", "up", 230.0)
    step(ctl, readout(anatomy, **lost))                                   # first loss: re-seat
    assert (ctl.stage, ctl.substage, ctl.rewires, ctl.failed) == ("place", "rewire", 1, False)
    ctl._go("wire", "up", 230.0)
    assert step(ctl, readout(anatomy, **lost)) == (0.0, 0.0, 0.0, 0.0)   # second loss: give up
    assert ctl.failed and ctl.failure_reason == "rewires:2"
    assert ctl.log.count(("failed:rewires:2", ctl.steps)) == 1
    assert step(ctl, readout(anatomy, L=[100.0, 60.0])) == (0.0, 0.0, 0.0, 0.0)   # and stays silent
    assert sum(str(name).startswith("failed:") for name, _ in ctl.log) == 1


def test_the_script_gives_up_when_a_stage_runs_past_its_budget(anatomy):
    ctl = controller(anatomy, stage_budgets={"wire": 6}, fork_max_tries=1000)
    ctl._go("wire", "fork", 230.0)
    stuck = dict(L=[247.0, 230.0], vessel_cath="bct", wire_branch="bct", depth_bct_cath=20.0, depth_bct_wire=36.0)
    for _ in range(5):
        step(ctl, readout(anatomy, **stuck))
    assert not ctl.failed and ctl.stage == "wire"
    assert step(ctl, readout(anatomy, **stuck)) == (0.0, 0.0, 0.0, 0.0)
    assert ctl.failed and ctl.failure_reason == "budget:wire"
    # the count starts over when a stage is entered; a stage without a budget never gives up
    ctl = controller(anatomy, stage_budgets={"wire": 6}, fork_max_tries=1000)
    ctl.steps = 100
    ctl._go("wire", "fork", 230.0)
    for _ in range(5):
        step(ctl, readout(anatomy, **stuck))
    assert not ctl.failed
    ctl = controller(anatomy, stage_budgets={}, fork_max_tries=1000)
    ctl._go("wire", "fork", 230.0)
    for _ in range(300):
        step(ctl, readout(anatomy, **stuck))
    assert not ctl.failed and ctl.stage == "wire"


def test_the_script_gives_up_after_too_many_hook_placements(anatomy):
    ctl = controller(anatomy, give_up_tries=2)
    r = readout(anatomy, L=[230.0, 268.0])                                # d = -38: behind the hook
    ctl._next_try(r)
    assert ctl.tries == 1 and not ctl.failed
    ctl._next_try(r)
    assert step(ctl, r) == (0.0, 0.0, 0.0, 0.0) and ctl.failure_reason == "tries:2"
    # the defaults: the 2026-11-07 acceptance's successes never came near them
    p = S.ScriptParams()
    assert p.give_up_rewires == 2 and p.give_up_tries == len(p.offsets_mm) + 1 == 6
    assert p.stage_budgets == {"wire": 240, "ica": 300, "follow": 450}


def ica_readout(anatomy, L, chain, label=None, side=None, **kw):
    """A readout with the wire tip ``chain`` mm up the carotid chain of the ICA task (target at
    split + 30), the catheter seated 20 mm in the trunk unless given."""
    an = anatomy
    target_depth = an.split_mm + 30.0
    if label is None:
        label = "cca" if chain <= an.split_mm else "ica"
    if side is None:
        side = chain if label == "cca" else chain - an.split_mm
    base = dict(vessel_cath="bct", depth_bct_cath=20.0, wire_branch="rcca", wire_branch_depth=chain,
                target=an.rcca_point(target_depth), target_depth=target_depth, wire_carotid=label,
                wire_carotid_depth=side, wire_chain=chain)
    base.update(kw)
    return readout(anatomy, L=L, **base)


def test_up_hands_over_to_the_ica_stage_which_aims_backs_off_from_the_eca_and_anchors(anatomy):
    ctl = controller(anatomy)
    p, an = ctl.p, anatomy
    ctl._go("wire", "up", 230.0)
    step(ctl, ica_readout(an, [300.0, 230.0], p.up_goal_mm + 1.0))
    assert ctl.stage == "ica"
    v_w, w_w, v_c, _ = step(ctl, ica_readout(an, [310.0, 230.0], 60.0))
    assert v_w == p.v_ica and v_c == 0.0                     # between the forks: plain push
    # into the ECA: back below the split, then on
    v_w, _, _, _ = step(ctl, ica_readout(an, [345.0, 230.0], an.split_mm + 7.0, label="eca", side=p.eca_back_mm + 1))
    assert ctl.substage == "backoff" and v_w == -p.v_ica_backoff and ("ica:eca", ctl.steps) in ctl.log
    step(ctl, ica_readout(an, [330.0, 230.0], an.split_mm - p.eca_back_to_mm - 1.0))
    assert ctl.substage == ""
    # anchored past the split: the catheter follows
    step(ctl, ica_readout(an, [370.0, 230.0], an.split_mm + p.ica_anchor_mm + 1.0))
    assert ctl.stage == "follow"


def test_ica_stage_aims_at_the_ica_side_below_the_split_and_pushes_only_when_aimed(anatomy):
    ctl = controller(anatomy)
    p, an = ctl.p, anatomy
    ctl._go("ica", "", 230.0)
    chain = an.split_mm - 10.0
    # a wire polyline up the carotid with its J bend pointing to the ECA's side
    o = an.rcca_point(chain)
    shaft = S.unit(o - an.rcca_point(chain - 8.0))
    wrong = -an.lat_ica
    wire = np.array([o + wrong * max(0.0, 2.0 - 0.4 * k) - shaft * k for k in range(60)])
    v_w, w_w, _, _ = step(ctl, ica_readout(an, [340.0, 230.0], chain, wire=wire))
    assert v_w == 0.0 and abs(w_w) > 1.0                     # turn, no push, while it points the wrong way
    good = np.array([o + an.lat_ica * max(0.0, 2.0 - 0.4 * k) - shaft * k for k in range(60)])
    v_w, w_w, _, _ = step(ctl, ica_readout(an, [340.0, 230.0], chain, wire=good))
    assert v_w == p.aim_v and abs(w_w) < 1.0


def test_follow_pushes_the_catheter_leaves_the_wire_alone_pulses_on_a_stall_and_trims(anatomy):
    ctl = controller(anatomy)
    p, an = ctl.p, anatomy
    anchor = an.split_mm + 31.0
    ctl._go("follow", "", 230.0)
    r = ica_readout(an, [371.0, 230.0], anchor)
    v_w, _, v_c, _ = step(ctl, r)
    assert v_c == p.v_follow and v_w == 0.0                  # the wire's base is left alone
    goal_c, cath_min = ctl._cath_goal(r)
    assert cath_min == pytest.approx(an.split_mm + p.cath_ica_mm) and goal_c == pytest.approx(an.split_mm + 30.0 - p.cath_goal_offset_mm)
    v_w, _, _, _ = step(ctl, ica_readout(an, [371.0, 232.0], anchor - 4.0, depth_bct_cath=22.0))
    assert v_w == 0.0                                         # even when its tip slipped 4 mm: no feed
    for k in range(p.follow_stall_steps + 1):                 # catheter tip not moving: pulses
        _, _, v_c, _ = step(ctl, ica_readout(an, [371.0, 240.0 + k], anchor, depth_bct_cath=22.0))
    assert any(name == "follow:pulse" for name, _ in ctl.log) and ctl._f_mode == "pulse"
    pulses = [step(ctl, ica_readout(an, [371.0, 250.0 + k, ], anchor, depth_bct_cath=22.0))[2]
              for k in range(p.pulse_push_steps + p.pulse_pause_steps)]
    assert 0.0 in pulses and p.v_follow in pulses
    # a slip into the carotid (20 mm in one step): brake
    _, _, v_c, _ = step(ctl, ica_readout(an, [371.0, 263.0], anchor, vessel_cath="rcca", cath_carotid="cca",
                                         cath_carotid_depth=5.0, cath_chain=5.0, depth_rcca_cath=5.0))
    assert ctl.substage == "brake" and v_c == -p.v_brake
    # the wire slips out of the ICA: back to the ica stage
    step(ctl, ica_readout(an, [360.0, 263.0], an.split_mm + 2.0, vessel_cath="rcca", cath_carotid="cca",
                          cath_carotid_depth=5.0, cath_chain=5.0, depth_rcca_cath=5.0))
    assert ctl.stage == "ica" and ("follow:reanchor", ctl.steps) in ctl.log


def test_trim_meets_the_ica_success_test_and_holds(anatomy):
    ctl = controller(anatomy)
    p, an = ctl.p, anatomy
    target_depth = an.split_mm + 30.0
    target = an.rcca_point(target_depth)
    ctl._go("follow", "", 300.0)
    near = ica_readout(an, [371.0, 350.0], target_depth + 1.0, vessel_cath="rcca", cath_carotid="ica",
                       cath_carotid_depth=7.0, cath_chain=an.split_mm + 7.0, depth_rcca_cath=an.split_mm + 7.0)
    step(ctl, near)
    assert ctl.stage == "trim"
    wire = np.repeat(target[None], 60, axis=0)
    ok = dict(wire=wire, vessel_cath="rcca", cath_carotid="ica", cath_carotid_depth=8.0,
              cath_chain=an.split_mm + 8.0, depth_rcca_cath=an.split_mm + 8.0)
    for _ in range(p.success_hold):
        out = step(ctl, ica_readout(an, [371.0, 349.0], target_depth, **ok))
    assert ctl.stage == "done" and ctl.success and out == (0.0, 0.0, 0.0, 0.0)


def test_carotid_target_keeps_the_old_end_point(anatomy):
    """A target in the common carotid (the round-one configs): up to target - 5, follow, catheter
    >= 15 mm into the carotid."""
    ctl = controller(anatomy)
    p = ctl.p
    ctl._go("wire", "up", 230.0)
    r = readout(anatomy, L=[300.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0, wire_branch="rcca",
                wire_branch_depth=45.0 + p.wire_extra_mm + 0.5, wire_carotid="cca", wire_chain=45.0 + p.wire_extra_mm + 0.5)
    step(ctl, r)
    assert ctl.stage == "follow" and not ctl._ica_task(r)
    assert ctl._cath_goal(r)[1] == p.cath_min_mm


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
def test_whole_episode_from_the_insertion_point_to_success():
    """configs/two_devices_ica.yaml from the insertion point: the referee's success (both devices
    in the internal carotid), its five phases in order, the script's stages in order. Runs are not
    repeatable bit for bit (multithreaded collision detection), so a second seed is tried before
    failing. About 10 minutes per seed at 1 s per step."""
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env

    env = build_env(load_config(CONFIG)["env"])
    try:
        for seed in (1005, 1006):
            ctl = S.StagedController()
            obs, info = env.reset(seed=seed)
            ctl.reset(env)
            phases = [int(info["phase"])]
            for _ in range(800):
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
