"""Episodes that start from a saved state (guidewire_rl/env/two_device/snapshot.py; work log 2026-11-09).

With SOFA (``sofa``), on the real-mass two-device config: a state captured after 60 scripted steps,
restored into a fresh scene, lays the same number of nodes, puts the tips back within 0.05 mm of the
reference after the same one substep, and the next 8 recorded actions give tips within 0.1 mm and the same labels (to 0.05 N) as the reference run;
the referee's phase and step count come back with it. ``FromSnapshot`` as ``env.start`` restores a
library snapshot at reset (inserted lengths, referee phase, finite first observation, ``last`` says
which) and falls back to the ordinary opening with ``fraction: 0``. A snapshot from another physics
step or device set is refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from two_device_scenes import real_mass_env

pytestmark = [pytest.mark.upstream, pytest.mark.sofa]

PREFIX, TAIL = 60, 8


def scripted_prefix(env, seed: int, steps: int):
    from guidewire_rl.env.two_device.script import StagedController
    ctl = StagedController()
    env.reset(seed=seed)
    ctl.reset(env)
    for _ in range(steps):
        env.step(ctl.act(env))
    return ctl


def tips(env):
    sim = env.simulation
    return np.asarray(sim.dof_positions, dtype=float)[:3, :3].copy()      # the leading tip and two nodes


def test_restore_continues_the_reference_episode(tmp_path):
    from guidewire_rl.env.two_device import snapshot

    with real_mass_env() as env:
        ctl = scripted_prefix(env, 1000, PREFIX)
        snap = snapshot.capture(env, seed=1000, step=PREFIX, stage=ctl.stage)
        snapshot.save(tmp_path / "s.npz", snap)
        saved_tips, saved_phase = tips(env), env.monitors["phase_detector"].state.phase
        # the restore takes one zero-action substep to lay the nodes: give the reference the same one
        env.simulation.step(np.zeros((2, 2)), env.simulation.dt_simulation)
        after_substep = tips(env)
        actions, reference = [], []
        for _ in range(TAIL):
            a = ctl.act(env)
            actions.append(np.array(a, np.float32))
            _, _, _, _, info = env.step(a)
            reference.append((tips(env), dict(info["labels"]), int(info["phase"])))

    loaded = snapshot.load(tmp_path / "s.npz")
    assert loaded["meta"] == snap["meta"] and loaded["meta"]["step"] == PREFIX
    with real_mass_env() as env:
        env.reset(seed=1000)
        report = snapshot.restore_simulation(env.simulation, loaded)
        snapshot.apply_referee_state(env.monitors["phase_detector"], loaded["referee"])
        assert report["nodes"] == len(loaded["position"]) and report["tip_shift_mm"] < 0.3    # one substep of motion
        np.testing.assert_allclose(tips(env), after_substep, atol=0.05)
        np.testing.assert_allclose(env.simulation.inserted_lengths, loaded["xtip"], atol=1e-6)
        referee = env.monitors["phase_detector"].state
        assert referee.phase == saved_phase and referee.step == loaded["referee"]["state"]["step"]
        for k, a in enumerate(actions):
            _, _, terminated, truncated, info = env.step(a)
            ref_tips, ref_labels, ref_phase = reference[k]
            np.testing.assert_allclose(tips(env), ref_tips, atol=0.1)
            for name in ("distal_force", "catheter_tip_force"):
                assert abs(info["labels"][name] - ref_labels[name]) < 0.05
            assert info["phase"] == ref_phase and not info["simulation_error"]
            assert not (terminated or truncated)


def test_from_snapshot_start_restores_a_library_snapshot_and_falls_back(tmp_path):
    from guidewire_rl.env.two_device import snapshot

    with real_mass_env() as env:
        ctl = scripted_prefix(env, 1001, PREFIX)
        snap = snapshot.capture(env, seed=1001, step=PREFIX, stage=ctl.stage)
        snapshot.save(tmp_path / "seed1001_step60.npz", snap)

    with real_mass_env(start={"type": "FromSnapshot", "library": str(tmp_path), "fraction": 1.0}) as env:
        observation, info = env.reset(seed=5)
        start = env._env.start
        assert start.last is not None and start.last["step"] == PREFIX and start.last["nodes"] == len(snap["position"])
        np.testing.assert_allclose(env.simulation.inserted_lengths, snap["xtip"], atol=1e-6)
        assert env.monitors["phase_detector"].state.phase == snap["meta"]["phase"]
        assert info["phase"] == snap["meta"]["phase"] and np.isfinite(observation).all()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
        assert not info["simulation_error"] and np.isfinite(info["labels"]["distal_force"])

    with real_mass_env(start={"type": "FromSnapshot", "library": str(tmp_path), "fraction": 0.0}) as env:
        env.reset(seed=5)
        assert env._env.start.last is None
        np.testing.assert_allclose(env.simulation.inserted_lengths, [40.0, 0.0], atol=1e-6)   # WireFirst


def test_snapshot_from_another_scene_is_refused(tmp_path):
    from guidewire_rl.env.two_device import snapshot

    with real_mass_env() as env:
        scripted_prefix(env, 1000, 5)
        snap = snapshot.capture(env, seed=1000, step=5, stage="approach")
        env.reset(seed=1000)
        other = dict(snap, dt=snap["dt"] * 2)
        with pytest.raises(RuntimeError, match="dt"):
            snapshot.restore_simulation(env.simulation, other)
        other = dict(snap, instruments=[["guidewire", 400.0], snap["instruments"][1]])
        with pytest.raises(RuntimeError, match="instruments"):
            snapshot.restore_simulation(env.simulation, other)
    with pytest.raises(FileNotFoundError):
        from guidewire_rl.env.two_device.snapshot import FromSnapshot
        FromSnapshot(None, library=str(tmp_path / "empty"), fraction=1.0)
