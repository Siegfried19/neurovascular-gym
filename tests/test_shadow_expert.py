"""The hand-written procedure in shadow (``env.shadow_expert``, guidewire_rl/env/two_device/shadow.py;
DAgger-style labels, 2026-11-13).

Without SOFA: the config switch builds nothing when off and the named controller when on; a controller
that has given up, finished or raises gives no label (NaN) and says why. Synced (``sync: true``), a
script behind the referee's phase is moved forward to the stage the phase implies (never back), and a
label chosen in a stage behind the phase is dropped.

With SOFA (``sofa``, ``upstream``), on the round-two env (two_devices_ica.yaml): for the same 30 random
executed actions from the same seed, an episode with the controller in shadow is bit for bit (the
observations, rewards, force labels) an episode without it (it only reads the simulation); its labels
are finite and normalized, its stage a stage name. Driven by a second copy of the procedure (as a
scripted prefix or a demonstration recording is), the shadow's label on every step is the executed
action itself.

Repeated episodes in one process are not all bit for bit the same, shadow or not (measured
2026-11-13, six episodes of the same seed and actions in one process, with and without the shadow:
two classes, episodes 1, 4, 6 and 2, 3, 5, each identical within, shadowed and plain episodes in
both; between the classes the beam's nodes differ by 1e-11 mm at the first step and 3e-7 mm after 30,
the observations by 6e-8, the rewards by 4e-10, the labels by 3e-7): SOFA's own round-off depends on
the process's history. So a shadowed episode must match a plain one to within that round-off
(``TOLERANCE``, about 100 times it), far below anything a change of the simulation would leave.
"""

from __future__ import annotations

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.env.two_device.shadow import ShadowExpert

ROUND2 = WORKSPACE_ROOT / "configs" / "two_devices_ica.yaml"
STEPS, SEED = 30, 1000


class Stub:
    """A controller with the procedure's interface; ``mode`` decides what ``act`` does."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode, self.stage, self.substage, self.failed, self.failure_reason = mode, "approach", "", False, ""
        self.log = [("approach", 0)]
        self.resets = 0

    def reset(self, env=None) -> None:
        self.resets += 1

    def act(self, env=None) -> np.ndarray:
        if self.mode == "raise":
            raise RuntimeError("no polyline")
        if self.mode == "give_up":
            self.failed, self.failure_reason = True, "budget:wire"
            self.log.append(("failed:budget:wire", 1))
        if self.mode == "done":
            self.stage = "done"
            self.log.append(("done", 1))
        return np.array([0.5, -1.0, 0.25, 0.0], np.float32)


def test_switch_and_labels_without_sofa():
    assert ShadowExpert.from_config(None) is None and ShadowExpert.from_config(False) is None
    shadow = ShadowExpert.from_config({"type": "StagedController", "turn_hold": 4})
    from guidewire_rl.env.two_device.script import StagedController
    assert isinstance(shadow.controller, StagedController) and shadow.controller.p.turn_hold == 4
    with pytest.raises(KeyError):
        ShadowExpert.from_config({"type": "NoSuchController"})
    out = ShadowExpert(Stub("ok")).label(None, 4)
    assert out["expert_action"].tolist() == [0.5, -1.0, 0.25, 0.0] and out["expert_off"] == ""
    assert out["expert_stage"] == "approach" and "expert_events" not in out
    for mode, off in (("give_up", "gave_up:budget:wire"), ("done", "done"), ("raise", "error:")):
        stub = Stub(mode)
        out = ShadowExpert(stub).label(None, 4)
        assert np.isnan(out["expert_action"]).all() and out["expert_action"].shape == (4,)
        assert out["expert_off"].startswith(off)
    stub = Stub("give_up")
    shadow = ShadowExpert(stub)
    assert shadow.label(None, 4)["expert_events"] == ["failed:budget:wire"]
    stub.mode = "ok"                                   # once given up: no label any more, act not asked
    assert np.isnan(shadow.label(None, 4)["expert_action"]).all() and len(stub.log) == 2


class StagedStub(Stub):
    """A controller that keeps its stage unless moved (``_go``, as ``StagedController``'s); ``fall_back``:
    its ``act`` falls back to that stage (the script's own recovery rules)."""

    def __init__(self, stage="approach", substage="", ica=True, fall_back=None) -> None:
        super().__init__("ok")
        from types import SimpleNamespace
        self.stage, self.substage, self.steps, self.ica, self.fall_back = stage, substage, 0, ica, fall_back
        self.p = SimpleNamespace(engage_mm=2.0)
        self.entered = []

    def _go(self, stage, substage="", L_cath=0.0):
        self.stage, self.substage = stage, substage
        self.entered.append((stage, substage, L_cath))
        self.log.append((stage if not substage else f"{stage}:{substage}", self.steps))

    def _ica_task(self, r):
        return self.ica

    def act(self, env=None):
        self.steps += 1
        if self.fall_back:
            self.stage, self.substage = self.fall_back
        return super().act(env)


class RefereeEnv:
    """Just the referee's phase (``monitors["phase_detector"].state.phase``)."""

    def __init__(self, phase: int) -> None:
        from types import SimpleNamespace
        self.monitors = {"phase_detector": SimpleNamespace(state=SimpleNamespace(phase=phase))}


def readout(cath_in_bct=False, depth_bct_cath=-np.inf, d=-40.0):
    from types import SimpleNamespace
    return lambda env: SimpleNamespace(cath_in_bct=cath_in_bct, depth_bct_cath=depth_bct_cath, d=d,
                                       L=np.array([300.0, 250.0]))


@pytest.mark.parametrize("phase, stage, geometry, expected", [
    (0, "approach", {}, ("approach", "")),                                    # nothing to catch up
    (1, "approach", {}, ("place", "retract")),
    (2, "approach", {}, ("turn", "")),
    (2, "place", {"cath_in_bct": True, "depth_bct_cath": 5.0}, ("seat", "")),
    (3, "turn", {"cath_in_bct": True, "depth_bct_cath": 12.0, "d": -30.0}, ("seat", "")),
    (3, "seat", {"cath_in_bct": True, "depth_bct_cath": 12.0, "d": 4.0}, ("wire", "out")),
    (3, "place", {"cath_in_bct": False, "d": -30.0}, ("wire", "out")),       # the script does not see it seated
    (4, "place", {}, ("ica", "")),
    (4, "wire", {}, ("ica", "")),
    (2, "follow", {}, ("follow", "")),                                        # ahead: left alone
    (3, "wire", {"d": 20.0}, ("wire", "")),                                   # at the stage already
])
def test_sync_moves_a_lagging_script_forward_only(phase, stage, geometry, expected):
    stub = StagedStub(stage)
    shadow = ShadowExpert(stub, sync=True, readout=readout(**geometry))
    out = shadow.label(RefereeEnv(phase), 4)
    assert (stub.stage, stub.substage) == expected
    assert out["expert_off"] == "" and np.isfinite(out["expert_action"]).all()
    moved = expected != (stage, "")
    assert any(e.startswith("sync:") for e in out.get("expert_events", [])) == moved
    if moved:
        assert stub.entered[-1][2] == 250.0                                   # entered at the catheter's length


def test_sync_off_by_default_and_the_safety_net():
    stub = StagedStub("approach")
    out = ShadowExpert(stub, readout=readout()).label(RefereeEnv(4), 4)      # sync off: nothing moves
    assert stub.stage == "approach" and out["expert_off"] == ""
    carotid = StagedStub("place", ica=False)
    ShadowExpert(carotid, sync=True, readout=readout()).label(RefereeEnv(4), 4)
    assert (carotid.stage, carotid.substage) == ("wire", "up")              # a carotid target climbs first
    # the script falls back by its own rules behind what the phase allows: no label
    stub = StagedStub("seat", fall_back=("place", "rewire"))
    out = ShadowExpert(stub, sync=True, readout=readout(cath_in_bct=True, depth_bct_cath=10.0)).label(RefereeEnv(3), 4)
    assert np.isnan(out["expert_action"]).all() and out["expert_off"] == "behind:place@P3"
    stub = StagedStub("wire", fall_back=("wire", "aim"))                     # still at the phase's least stage
    out = ShadowExpert(stub, sync=True, readout=readout(d=20.0)).label(RefereeEnv(4), 4)
    assert out["expert_off"] == "" and stub.substage == "aim"
    assert ShadowExpert.from_config({"type": "StagedController", "sync": True}).sync


def _run(env, actions):
    obs, _ = env.reset(seed=SEED)
    rows = [(obs, 0.0, {})]
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append((obs, reward, info))
        assert not (terminated or truncated or info.get("simulation_error") or info.get("wall_breached"))
    return rows


TOLERANCE = {"obs": 1e-5, "reward": 1e-7, "labels": 1e-4}     # SOFA's repeat round-off x ~100 (docstring)


def _gaps(a, b):
    """Largest differences between two runs: observations, rewards, labels (NaN = NaN)."""
    def labels(info):
        values = info.get("labels") or {}
        return np.array([values[k] for k in sorted(values)], dtype=float)
    gaps = {"obs": 0.0, "reward": 0.0, "labels": 0.0}
    for (obs_a, rew_a, info_a), (obs_b, rew_b, info_b) in zip(a, b):
        la, lb = labels(info_a), labels(info_b)
        assert np.array_equal(np.isnan(la), np.isnan(lb))
        gaps["obs"] = max(gaps["obs"], float(np.max(np.abs(obs_a - obs_b))))
        gaps["reward"] = max(gaps["reward"], abs(rew_a - rew_b))
        gaps["labels"] = max(gaps["labels"], float(np.nanmax(np.abs(la - lb), initial=0.0)))
    return gaps


@pytest.mark.sofa
@pytest.mark.upstream          # the round-two scene needs the new env's stEVE
def test_the_shadow_only_reads_the_simulation_and_labels_every_step():
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env
    from guidewire_rl.env.two_device.script import STAGES, StagedController

    env = build_env(load_config(ROUND2)["env"])
    try:
        assert env.shadow_expert is None                                  # off in the config
        rng = np.random.default_rng(0)
        actions = [rng.uniform(-1, 1, 4).astype(np.float32) for _ in range(STEPS)]
        actions[:10] = [np.array([1.0, 0.0, 1.0, 0.0], np.float32)] * 10   # get both devices in first
        plain = _run(env, actions)
        assert all("expert_action" not in info for _, _, info in plain)
        env.shadow_expert = ShadowExpert(StagedController())
        for _ in range(2):
            shadowed = _run(env, actions)
            gaps = _gaps(plain, shadowed)
            assert all(gaps[key] <= TOLERANCE[key] for key in TOLERANCE), gaps
            for _, _, info in shadowed[1:]:
                label = info["expert_action"]
                assert label.dtype == np.float32 and label.shape == (4,)
                assert np.all(np.isfinite(label)) and np.all(np.abs(label) <= 1.0), info["expert_off"]
                assert info["expert_stage"] in STAGES and info["expert_off"] == ""
        # driven by the procedure itself (a scripted prefix, a recording), the label is the action taken
        driver = StagedController()
        env.reset(seed=SEED)
        driver.reset(env)
        for _ in range(STEPS):
            action = driver.act(env)
            _, _, _, _, info = env.step(action)
            np.testing.assert_array_equal(info["expert_action"], action)
            assert info["expert_stage"] == driver.stage
    finally:
        env.close()


@pytest.mark.sofa
@pytest.mark.upstream
@pytest.mark.slow              # ~250 SOFA steps with two controllers: minutes (--run-slow)
def test_a_synced_shadow_kept_behind_catches_up_with_the_driving_script():
    """Synced labels from the state (lead 2026-11-13): the script drives seed 1000 from the insertion
    point through the hook, the seat and into the wire stage; the env's shadow, synced, is put back to
    ``approach`` every 40 steps. After each reset it reaches the driver's stage within 3 steps, every
    step is labelled, and once at the driver's stage its labels are the driver's actions (each dimension
    within 0.2 on >= 90 % of those steps, per stage). Measured 2026-11-13 over 340 steps
    (runs/round2_twocrutch_v2/shadow_sync_agreement.json): catch-up 0-2 steps; same-stage agreement
    100 % in every stage but place (95 %); inside P1 the phase cannot tell place from turn, so a shadow
    reset there replays its placement while the driver turns (turn: 10 of 31 steps at the same stage)."""
    from collections import defaultdict
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env
    from guidewire_rl.env.two_device.script import StagedController

    refresh, steps = 40, 250
    env = build_env(load_config(ROUND2)["env"])
    try:
        env.shadow_expert = ShadowExpert(StagedController(), sync=True)
        driver = StagedController()
        env.reset(seed=SEED)
        driver.reset(env)
        rows = []
        for k in range(steps):
            if k > 0 and k % refresh == 0:
                env.shadow_expert.controller.reset(env)                # back to approach
            action = driver.act(env)
            _, _, te, tr, info = env.step(action)
            rows.append((k, driver.stage, info["expert_stage"], np.asarray(action), info["expert_action"],
                         info["expert_off"]))
            assert not (te or tr or info.get("simulation_error") or info.get("wall_breached"))
    finally:
        env.close()
    assert {"seat", "wire"} <= {r[1] for r in rows}                    # got as far as the wire stage
    assert all(r[5] == "" for r in rows)                                # every step labelled
    for k in range(refresh, steps, refresh):
        caught = next(j - k for j in range(k, min(k + refresh, steps)) if rows[j][1] == rows[j][2])
        assert caught <= 3, (k, rows[k][1])
    same = defaultdict(list)
    for _, stage, shadow, action, label, _ in rows:
        if stage == shadow:
            same[stage].append(np.all(np.abs(label - action) <= 0.2))
    for stage, agree in same.items():
        if len(agree) >= 5:
            assert np.mean(agree) >= 0.9, (stage, np.mean(agree))
