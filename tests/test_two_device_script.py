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


# ------------------------------------- a procedure a policy without memory can copy (11-14) ----
def test_the_procedure_switches_default_to_the_procedure_as_it_was():
    p = S.ScriptParams()
    assert (p.first_place, p.turn_dir, p.turn_far_deg, p.stall_needs_push, p.push_done_ratio, p.give_up) \
        == (True, 0, 60.0, False, 0.5, True)


def test_first_place_off_turns_the_hook_up_where_the_retraction_left_it_on_the_first_try_only(anatomy):
    behind = readout(anatomy, L=[230.0, 268.0])                          # d = -38: wire behind the hook
    ctl = controller(anatomy)                                            # default: place first
    ctl._go("place", "retract")
    step(ctl, behind)
    assert (ctl.stage, ctl.substage) == ("place", "")
    # off, first try: straight to the turn, with the turn's action (catheter still, hook turned)
    ctl = controller(anatomy, first_place=False)
    p = ctl.p
    ctl._go("place", "retract")
    _, _, v_c, w_c = step(ctl, behind)
    assert (ctl.stage, ctl.substage, ctl.tries) == ("turn", "", 0)
    assert [name for name, _ in ctl.log] == ["approach", "place:retract", "turn"]
    assert v_c == 0.0 and w_c == pytest.approx(ctl._turn_rate(S.wrap(behind.phi_up - ctl.aim)))
    for _ in range(p.turn_hold):                                         # hook up and held: pull
        step(ctl, readout(anatomy, L=[230.0, 268.0], phi_up=ctl.aim + 0.05))
    assert ctl.stage == "pull"
    # a pull that finds nothing: the next try places as before (hook down, then the place stage)
    step(ctl, readout(anatomy, L=[230.0, 268.0 - p.pull_max_mm - 1], vessel_cath="aorta"))
    assert (ctl.stage, ctl.substage, ctl.tries) == ("place", "", 1) and ("place:lower", ctl.steps) in ctl.log
    # a retraction on a later try places too
    ctl = controller(anatomy, first_place=False)
    ctl.tries = 1
    ctl._go("place", "retract")
    step(ctl, behind)
    assert (ctl.stage, ctl.substage) == ("place", "")


def test_far_turn_is_full_rate_one_way_round_beyond_turn_far_deg_and_none_otherwise():
    w = S.ScriptParams().w_cath_max
    ctl = S.StagedController(turn_dir=-1)
    for deg in (61.0, 90.0, 179.0, -61.0, -90.0, -179.0):
        assert ctl._far_turn(np.radians(deg)) == -w, deg
    for deg in (0.0, 30.0, -30.0, 59.0, -59.0):
        assert ctl._far_turn(np.radians(deg)) is None, deg
    assert S.StagedController(turn_dir=1)._far_turn(np.radians(-90.0)) == w
    assert S.StagedController(turn_dir=-3)._far_turn(np.radians(90.0)) == -w          # its sign only
    assert S.StagedController(turn_dir=-1, turn_far_deg=30.0)._far_turn(np.radians(40.0)) == -w
    ctl = S.StagedController()                                                         # turn_dir 0
    assert all(ctl._far_turn(np.radians(deg)) is None for deg in (-179.0, -90.0, 90.0, 179.0))


def _hook_turn_rates(anatomy, deg, **params):
    """The catheter turn rate each hook stage sends with the hook ``deg`` degrees off that stage's
    aim: place (hook down, ``place_aim_deg``), place:lower (``phi_down``), turn and pull (first aim)."""
    e = np.radians(deg)
    L = [223.0, 260.0]
    rates = {}
    ctl = controller(anatomy, **params)
    ctl._go("place", "", 260.0)
    rates["place"] = step(ctl, readout(anatomy, L=L, phi_up=S.wrap(np.radians(ctl.p.place_aim_deg) + e)))[3]
    assert (ctl.stage, ctl.substage) == ("place", "")
    ctl = controller(anatomy, **params)
    ctl._go("place", "lower", 260.0)
    rates["lower"] = step(ctl, readout(anatomy, L=L, phi_down=e))[3]
    assert (ctl.stage, ctl.substage) == ("place", "lower")
    for stage in ("turn", "pull"):
        ctl = controller(anatomy, **params)
        ctl._go(stage, "", 260.0)
        rates[stage] = step(ctl, readout(anatomy, L=L, phi_up=S.wrap(ctl.aim + e), vessel_cath="aorta"))[3]
        assert ctl.stage == stage
    return rates


def test_turn_dir_turns_the_hook_one_way_round_while_far_off_its_aim_in_place_lower_turn_and_pull(anatomy):
    p = S.ScriptParams()
    for turn_dir in (0, -1, 1):
        for deg in (150.0, 90.0, -90.0, -150.0, 40.0, -40.0):
            rates = _hook_turn_rates(anatomy, deg, turn_dir=turn_dir)
            if turn_dir and abs(deg) > p.turn_far_deg:
                want = turn_dir * p.w_cath_max                   # whatever the side of the aim
            else:                                                # the old rate: the short way round
                want = float(np.clip(p.k_rot * np.radians(deg), -p.w_cath_max, p.w_cath_max))
            assert rates == pytest.approx(dict.fromkeys(rates, want)), (turn_dir, deg, rates)


class _Shell:
    """An env whose shell executed ``last_action`` last step ([[wire push, wire turn], [catheter push,
    catheter turn]], mm/s and rad/s, as stEVE keeps it); no attribute when ``None``."""

    simulation = None

    def __init__(self, last_action):
        if last_action is not None:
            self.last_action = np.asarray(last_action, float)


def test_note_shell_tells_whether_the_push_asked_last_step_was_executed(anatomy):
    asked = [20.0, 0.0, 8.0, 0.0]

    def note(done, last=asked, steps=1, **params):
        ctl = controller(anatomy, **params)
        ctl.steps, ctl.last_physical = steps, np.asarray(last, float)
        ctl._note_shell(_Shell(done))
        return ctl._wire_pushed, ctl._cath_pushed

    assert note([[20.0, 0.0], [8.0, 0.0]]) == (True, True)
    assert note([[10.0, 0.0], [4.0, 0.0]]) == (True, True)               # half of the ask is done
    assert note([[9.9, 0.0], [8.0, 0.0]]) == (False, True)
    assert note([[20.0, 0.0], [3.9, 0.0]]) == (True, False)
    assert note([[-5.0, 0.0], [-5.0, 0.0]]) == (False, False)            # pulled instead
    assert note([[7.9, 0.0], [3.1, 0.0]], push_done_ratio=0.4) == (False, False)
    assert note([[8.0, 0.0], [3.2, 0.0]], push_done_ratio=0.4) == (True, True)
    # no push to check: not asked (<= 1 mm/s, a pull), the first step, no record, one device only
    assert note([[0.0, 0.0], [0.0, 0.0]], last=[1.0, 0.0, 1.0, 0.0]) == (True, True)
    assert note([[0.0, 0.0], [0.0, 0.0]], last=[-15.0, 0.0, -3.0, 0.0]) == (True, True)
    assert note([[0.0, 0.0], [0.0, 0.0]], steps=0) == (True, True)
    assert note(None) == (True, True)
    assert note([0.0, 0.0]) == (False, True)
    # a step that was executed sets the flags back
    ctl = controller(anatomy)
    ctl.steps, ctl.last_physical = 1, np.asarray(asked)
    ctl._note_shell(_Shell([[0.0, 0.0], [0.0, 0.0]]))
    assert not ctl._wire_pushed and not ctl._cath_pushed
    ctl._note_shell(_Shell([[20.0, 0.0], [8.0, 0.0]]))
    assert ctl._wire_pushed and ctl._cath_pushed
    # the blocked-wire count: off, as before (less than a quarter of the ask, in a row); on, never
    ctl = controller(anatomy)
    ctl.steps, ctl.last_physical = 1, np.asarray(asked)
    for done, blocked in ((4.9, 1), (4.9, 2), (5.0, 0), (0.0, 1)):
        ctl._note_shell(_Shell([[done, 0.0], [8.0, 0.0]]))
        assert ctl._wire_blocked == blocked and not ctl._wire_pushed, done
    ctl = controller(anatomy, stall_needs_push=True)
    ctl.steps, ctl.last_physical = 1, np.asarray(asked)
    for _ in range(2 * ctl.p.blocked_steps):
        ctl._note_shell(_Shell([[0.0, 0.0], [8.0, 0.0]]))
        assert ctl._wire_blocked == 0 and not ctl._wire_pushed


def test_act_checks_the_shells_last_action_against_what_it_asked_the_step_before(anatomy):
    for on in (False, True):
        intervention = _FakeIntervention(anatomy, L_wire=0.0, L_cath=0.0)
        intervention.last_action = np.zeros((2, 2))                     # the shell executed nothing
        ctl = S.StagedController(stall_needs_push=on)
        ctl.reset(intervention)
        ctl.act(intervention)                                            # first step: nothing asked before
        assert ctl._wire_pushed and ctl._cath_pushed and ctl.last_physical[0] == pytest.approx(40.0)
        ctl.act(intervention)                                            # asked 40 mm/s of wire, got 0
        assert not ctl._wire_pushed and ctl._cath_pushed                 # (no catheter push asked)
        assert ctl._wire_blocked == (0 if on else 1)
        intervention.last_action = np.array([[ctl.last_physical[0], 0.0], [0.0, 0.0]])
        ctl.act(intervention)
        assert ctl._wire_pushed and ctl._wire_blocked == 0


def _fork(anatomy, progress, wire_pushed, cath_pushed=True, **params):
    """Fork bursts on steps with the wire tip ``progress`` mm into the trunk, each step's push (the
    one ``_note_shell`` checked) executed or not; the controller and the wire speeds it sent."""
    ctl = controller(anatomy, **params)
    ctl._go("wire", "fork", 230.0)
    v = []
    for x, ok in zip(progress, wire_pushed):
        ctl._wire_pushed, ctl._cath_pushed = ok, cath_pushed
        v.append(step(ctl, readout(anatomy, L=[247.0, 230.0], vessel_cath="bct", depth_bct_cath=20.0,
                                   wire_branch="bct", depth_bct_wire=x))[0])
    return ctl, v


def test_stall_needs_push_fork_bursts_count_only_the_pushes_executed(anatomy):
    p = S.ScriptParams()
    forks = lambda ctl: [name for name, _ in ctl.log if name.startswith("fork:")]
    # the tip does not move while the push is not made: on, no stall and nothing in the history ...
    ctl, v = _fork(anatomy, [30.0] * 10, [False] * 10, stall_needs_push=True)
    assert v == [p.v_fork] * 10 and ctl.substage == "fork" and ctl._fork_hist == [] and forks(ctl) == []
    # ... off, a stall as before
    ctl, v = _fork(anatomy, [30.0] * 3, [False] * 3)
    assert v[-1] == -p.v_backoff and forks(ctl) == ["fork:stall"] and ctl.steps == 3
    # on: executed pushes still stall, the ones not made in between are skipped
    ctl, v = _fork(anatomy, [30.0] * 5, [True, False, True, False, True], stall_needs_push=True)
    assert v[:4] == [p.v_fork] * 4 and v[4] == -p.v_backoff and forks(ctl) == ["fork:stall"]
    # the catheter's push does not matter here
    ctl, v = _fork(anatomy, [30.0] * 3, [True] * 3, cath_pushed=False, stall_needs_push=True)
    assert v[-1] == -p.v_backoff and forks(ctl) == ["fork:stall"]
    # a dip while the push was not made: on, no buckle, and what the tip moved over that step is left
    # out of the history (the last entry is 32 + the 6 mm the executed push gained); off, a buckle
    dip, made = [30.0, 32.0, 28.0, 34.0], [True, True, False, True]
    ctl, v = _fork(anatomy, dip, made, stall_needs_push=True)
    assert v == [p.v_fork] * 4 and ctl._fork_hist == [30.0, 32.0, 38.0] and forks(ctl) == []
    ctl, v = _fork(anatomy, dip, made)
    assert v[2] == -p.v_backoff and ("fork:buckle", 3) in ctl.log
    # the policy pulled 10 mm back over two steps it did not push, then pushed and gained 1 mm: that
    # retreat is not a buckle of the executed pushes (0.4 + 1 mm gained, no stall either)
    ctl, v = _fork(anatomy, [30.0, 30.4, 20.0, 20.0, 21.0], [True, True, False, False, True], stall_needs_push=True)
    assert v == [p.v_fork] * 5 and forks(ctl) == [] and ctl._fork_hist == pytest.approx([30.0, 30.4, 31.4])


def test_stall_needs_push_ica_stall_counts_only_the_pushes_executed(anatomy):
    p, an = S.ScriptParams(), anatomy
    n = p.ica_stall_steps + 1

    def run(steps, wire_pushed, cath_pushed=True, **params):
        ctl = controller(anatomy, **params)
        ctl._go("ica", "", 230.0)
        v = []
        for _ in range(steps):                                 # below both aiming windows: plain pushes
            ctl._wire_pushed, ctl._cath_pushed = wire_pushed, cath_pushed
            v.append(step(ctl, ica_readout(an, [330.0, 230.0], 40.0))[0])
        return ctl, v

    ctl, v = run(n, False)                                     # off: a stall as before
    assert v[:-1] == [p.v_ica] * (n - 1) and v[-1] == -p.v_ica_backoff and ("ica:stall", n) in ctl.log
    ctl, v = run(3 * n, False, stall_needs_push=True)          # on: pushes not made never stall
    assert v == [p.v_ica] * (3 * n) and ctl._ica_hist == [] and (ctl.stage, ctl.substage) == ("ica", "")
    ctl, v = run(n, True, cath_pushed=False, stall_needs_push=True)     # executed pushes still do
    assert v[-1] == -p.v_ica_backoff and ("ica:stall", n) in ctl.log


def test_stall_needs_push_follow_stall_counts_only_the_catheter_pushes_executed(anatomy):
    p, an = S.ScriptParams(), anatomy
    anchor = an.split_mm + 31.0
    n = p.follow_stall_steps + 1

    def run(steps, cath_pushed, wire_pushed=True, **params):
        ctl = controller(anatomy, **params)
        ctl._go("follow", "", 230.0)
        for _ in range(steps):                                 # the catheter tip does not move
            ctl._cath_pushed, ctl._wire_pushed = cath_pushed, wire_pushed
            step(ctl, ica_readout(an, [371.0, 240.0], anchor, depth_bct_cath=22.0))
        return ctl

    ctl = run(n, False)                                        # off: pulses as before
    assert ("follow:pulse", n) in ctl.log and ctl._f_mode == "pulse"
    ctl = run(3 * n, False, stall_needs_push=True)             # on: pushes not made never stall
    assert ctl._f_mode == "steady" and ctl._f_hist == [] and not any(nm == "follow:pulse" for nm, _ in ctl.log)
    ctl = run(n, True, wire_pushed=False, stall_needs_push=True)        # executed pushes still do
    assert ("follow:pulse", n) in ctl.log and ctl._f_mode == "pulse"


def test_give_up_off_never_ends_the_script(anatomy):
    reasons = ((dict(rewires=2), "rewires:2"), (dict(tries=6), "tries:6"),
               (dict(stage="wire", steps=240), "budget:wire"))
    for state, why in reasons:
        for give_up in (True, False):
            ctl = controller(anatomy, give_up=give_up)
            for key, value in state.items():
                setattr(ctl, key, value)
            assert ctl._give_up_reason() == (why if give_up else ""), (state, give_up)
    # the scenes of the give-up tests above: the script goes on
    p = S.ScriptParams()
    ctl = controller(anatomy, give_up=False)
    lost = dict(L=[260.0, 230.0], vessel_cath="aorta", depth_bct_cath=p.lost_mm - 1.0, wire_branch="bct",
                depth_bct_wire=36.0)
    for k in (1, 2, 3):
        ctl._go("wire", "up", 230.0)
        assert step(ctl, readout(anatomy, **lost)) != (0.0, 0.0, 0.0, 0.0)
        assert (ctl.stage, ctl.substage, ctl.rewires, ctl.failed) == ("place", "rewire", k, False)
    ctl = controller(anatomy, give_up=False, give_up_tries=2)
    r = readout(anatomy, L=[230.0, 268.0])
    ctl._next_try(r)
    ctl._next_try(r)
    assert step(ctl, r) != (0.0, 0.0, 0.0, 0.0) and not ctl.failed and ctl.failure_reason == ""
    ctl = controller(anatomy, give_up=False, stage_budgets={"wire": 6}, fork_max_tries=1000)
    ctl._go("wire", "fork", 230.0)
    stuck = dict(L=[247.0, 230.0], vessel_cath="bct", wire_branch="bct", depth_bct_cath=20.0, depth_bct_wire=36.0)
    for _ in range(30):
        step(ctl, readout(anatomy, **stuck))
    assert not ctl.failed and ctl.stage == "wire"
    assert not any(str(name).startswith("failed:") for name, _ in ctl.log)


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
        # round two's finish (the catheter 5 mm past the split, no lead condition) can arrive while
        # the script is still following, so 'trim' may never come; the stages that did come are in order
        present = [s for s in S.STAGES[:-1] if s in stages]
        assert present[:8] == list(S.STAGES[:8])
        order = [stages.index(s) for s in present]
        assert order == sorted(order)
    finally:
        env.close()
