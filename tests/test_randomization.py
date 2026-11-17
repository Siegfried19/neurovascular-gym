"""Per-episode scene parameters (guidewire_rl/env/randomization.py) and the uniform training target."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from guidewire_rl.config import load_config
from guidewire_rl.env.randomization import EpisodeRandomizer, InterventionWithRandomizer, ScenarioInfo

FRICTION = "configs/two_devices_ica_randomized.yaml"


def fake_simulation(friction=0.1, rebuild=1):
    return SimpleNamespace(friction=friction, rebuild_every_resets=rebuild)


def test_off_draws_nothing_and_reports_the_configured_friction():
    sim = fake_simulation(0.1)
    randomizer = EpisodeRandomizer(sim)
    randomizer.reset(seed=3)
    assert sim.friction == 0.1 and randomizer.drawn == {"friction": 0.1}
    assert ScenarioInfo(sim).info == {"scenario": {"friction": 0.1}}


def test_friction_is_drawn_per_episode_in_range_and_by_seed():
    sim = fake_simulation(0.1)
    randomizer = EpisodeRandomizer(sim, friction={"range": [0.03, 0.3], "log": True, "nominal_fraction": 0.3})
    values = []
    for seed in range(400):
        randomizer.reset(seed=seed)
        assert sim.friction == randomizer.drawn["friction"]
        values.append(sim.friction)
    values = np.array(values)
    nominal = values == 0.1
    assert 0.2 < nominal.mean() < 0.4                               # about 30 % keep the configured value
    drawn = values[~nominal]
    assert drawn.min() >= 0.03 and drawn.max() <= 0.3
    assert 0.4 < (drawn < np.sqrt(0.03 * 0.3)).mean() < 0.6         # log-uniform: half below the geometric mean
    randomizer.reset(seed=7)
    first = sim.friction
    randomizer.reset()                                              # no seed: the stream goes on
    randomizer.reset(seed=7)
    assert sim.friction == first
    assert ScenarioInfo(sim, randomizer).info == {"scenario": {"friction": first}}
    # the nominal value is the configured one, not the last draw
    assert randomizer.nominal_friction == 0.1


def test_the_env_pool_passes_the_scenario_on():
    """The simulator processes send a fixed list of info keys to the trainer; without this one the
    training episodes' records had no scenario (round3_randomized_v1's first minutes, 2026-11-16)."""
    from guidewire_rl.train.env_pool import INFO_KEYS
    assert "scenario" in INFO_KEYS


def test_bad_friction_specs_are_refused():
    with pytest.raises(ValueError):
        EpisodeRandomizer(fake_simulation(), friction={"range": [0.3, 0.03]})
    with pytest.raises(ValueError):
        EpisodeRandomizer(fake_simulation(), friction={"range": [0.03, 0.3], "nominal_fraction": 1.5})
    with pytest.raises(KeyError):
        EpisodeRandomizer(fake_simulation(), friction={"range": [0.03, 0.3], "lgo": True})
    with pytest.raises(ValueError):                                 # the coefficient is only read at a rebuild
        EpisodeRandomizer(fake_simulation(rebuild=0), friction={"range": [0.03, 0.3]})


def test_the_draw_comes_before_the_scene_is_built():
    sim = fake_simulation(0.1)
    seen = []

    class Intervention:
        marker = 5

        def reset(self, episode_number=0, seed=None, options=None):
            seen.append(sim.friction)

    wrapped = InterventionWithRandomizer(
        Intervention(), EpisodeRandomizer(sim, friction={"range": [0.2, 0.2]}))
    wrapped.reset(0, 1, None)
    assert seen == [0.2] and wrapped.marker == 5
    wrapped.requested_snapshot = "x"                                # writes reach the intervention
    assert wrapped.intervention.requested_snapshot == "x"


class SteppedSimulation:
    """A simulation that only records what it was told to execute."""

    def __init__(self):
        self.friction = 0.1
        self.rebuild_every_resets = 1
        self.executed = []

    def step(self, action, duration):
        self.executed.append(np.array(action, dtype=np.float64))


EXECUTION = {"delay_steps": [0.0, 1.0], "gain": [0.9, 1.1], "jitter": [0.0, 0.1],
             "deadband_push_mm_s": [0.0, 0.3], "deadband_rot_rad_s": [0.0, 0.03]}


def test_execution_off_or_not_live_executes_the_command_itself():
    command = np.array([[10.0, 1.0], [-5.0, 0.5]])
    sim = SteppedSimulation()
    randomizer = EpisodeRandomizer(sim, execution=EXECUTION)
    randomizer.reset(seed=1)
    sim.step(command, 0.1)                                          # e.g. the start's own pushes
    np.testing.assert_array_equal(sim.executed[-1], command)
    exact = EpisodeRandomizer(SteppedSimulation(), execution={**EXECUTION, "nominal_fraction": 1.0})
    exact.reset(seed=1)
    exact.live = True
    exact.simulation.step(command, 0.1)
    np.testing.assert_array_equal(exact.simulation.executed[-1], command)
    assert exact.drawn["delay_steps"] == 0.0 and exact.drawn["gain_2"] == 1.0 and exact.drawn["jitter"] == 0.0


def test_delay_blends_with_the_previous_command():
    sim = SteppedSimulation()
    randomizer = EpisodeRandomizer(sim, execution={"delay_steps": [0.25, 0.25]})
    randomizer.reset(seed=0)
    randomizer.live = True
    a, b = np.array([[8.0, 1.0], [4.0, -1.0]]), np.array([[-8.0, 0.0], [0.0, 1.0]])
    sim.step(a, 0.1)
    np.testing.assert_allclose(sim.executed[-1], 0.75 * a)          # at rest before the first command
    sim.step(b, 0.1)
    np.testing.assert_allclose(sim.executed[-1], 0.75 * b + 0.25 * a)
    randomizer.reset(seed=0)                                        # a new episode forgets the last command
    randomizer.live = True
    sim.step(b, 0.1)
    np.testing.assert_allclose(sim.executed[-1], 0.75 * b)
    # 3.75 steps (0.5 s at 7.5 Hz): step t executes 0.25 x command[t-3] + 0.75 x command[t-4]
    slow = EpisodeRandomizer(SteppedSimulation(), execution={"delay_steps": [3.75, 3.75]})
    slow.reset(seed=0)
    slow.live = True
    commands = [np.full((2, 2), float(k + 1)) for k in range(7)]
    for command in commands:
        slow.simulation.step(command, 0.1)
    expected = [0.0, 0.0, 0.0, 0.25 * 1, 0.25 * 2 + 0.75 * 1, 0.25 * 3 + 0.75 * 2, 0.25 * 4 + 0.75 * 3]
    np.testing.assert_allclose([e[0, 0] for e in slow.simulation.executed], expected)
    assert slow.drawn["delay_steps"] == 3.75


def test_gain_jitter_and_dead_band_stay_in_their_ranges_and_follow_the_seed():
    sim = SteppedSimulation()
    randomizer = EpisodeRandomizer(sim, execution={"gain": [0.9, 1.1], "jitter": [0.1, 0.1]})
    command = np.array([[10.0, 1.0], [-5.0, 0.5]])
    runs = []
    for seed in (3, 4, 3):
        randomizer.reset(seed=seed)
        randomizer.live = True
        sim.executed.clear()
        for _ in range(50):
            sim.step(command, 0.1)
        ratio = np.array(sim.executed) / command
        gains = np.array([randomizer.drawn[f"gain_{k}"] for k in range(4)]).reshape(2, 2)
        assert np.all((gains >= 0.9) & (gains <= 1.1))
        assert np.all(np.abs(ratio / gains - 1.0) <= 0.1 + 1e-12) and np.std(ratio, axis=0).min() > 0.01
        runs.append(np.array(sim.executed))
    np.testing.assert_array_equal(runs[0], runs[2])
    assert not np.array_equal(runs[0], runs[1])
    band = EpisodeRandomizer(SteppedSimulation(), execution={"deadband_push_mm_s": [0.5, 0.5],
                                                             "deadband_rot_rad_s": [0.05, 0.05]})
    band.reset(seed=0)
    band.live = True
    band.simulation.step(np.array([[0.4, 0.04], [-0.6, -0.06]]), 0.1)
    np.testing.assert_allclose(band.simulation.executed[-1], [[0.0, 0.0], [-0.6, -0.06]])


def test_a_changed_push_still_respects_the_insertion_limits():
    sim = SteppedSimulation()
    intervention = SimpleNamespace(instrument_lengths_inserted=[0.5, 449.5], instrument_lengths_maximum=[450.0, 450.0])
    randomizer = EpisodeRandomizer(sim, execution={"delay_steps": [1.0, 1.0]}, intervention=intervention)
    randomizer.reset(seed=0)
    randomizer.live = True
    sim.step(np.array([[-10.0, 0.0], [10.0, 0.0]]), 0.1)            # would end at -0.5 and 450.5 mm
    np.testing.assert_allclose(sim.executed[-1], [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(sim.executed[-1], [[0.0, 0.0], [0.0, 0.0]])   # one step late: nothing yet
    sim.step(np.array([[0.0, 1.0], [0.0, 1.0]]), 0.1)               # last step's pushes arrive: both dropped
    np.testing.assert_allclose(sim.executed[-1], [[0.0, 0.0], [0.0, 0.0]])


def test_bad_execution_specs_are_refused():
    for spec in ({"delay_steps": [0.0, 20.0]}, {"gain": [1.1, 0.9]}, {"gain": [0.0, 1.0]}, {"jitter": [0.0, 1.0]},
                 {"deadband_push_mm_s": [-0.1, 0.1]}, {"nominal_fraction": 2.0}):
        with pytest.raises(ValueError):
            EpisodeRandomizer(SteppedSimulation(), execution=spec)
    with pytest.raises(KeyError):
        EpisodeRandomizer(SteppedSimulation(), execution={"delay": [0.0, 1.0]})


@pytest.mark.sofa
@pytest.mark.upstream
def test_the_policy_sees_the_command_while_the_drive_does_something_else():
    """Gain 0.5 on every channel: the devices advance half as far as with an exact drive, while the
    observation's last action and info["raw_action"] are the commanded ones."""
    from guidewire_rl.env import build_env
    action = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    advanced, seen, raw = [], [], []
    for execution in (None, {"gain": [0.5, 0.5]}):
        overrides = ["env.shadow_expert=false", "env.randomize=false"]
        cfg = load_config(FRICTION, overrides)
        if execution:
            cfg["env"]["randomize"] = {"execution": execution}
        env = build_env(cfg["env"])
        try:
            env.reset(seed=21)
            before = np.array(env.intervention.instrument_lengths_inserted, dtype=float)
            for _ in range(3):
                obs, _, _, _, info = env.step(action)
            advanced.append(np.array(env.intervention.instrument_lengths_inserted, dtype=float) - before)
            lo, hi = env.action_slice
            seen.append(obs[lo:hi].copy())
            raw.append(info["raw_action"].copy())
            if execution:
                assert info["scenario"]["gain_0"] == 0.5 and info["scenario"]["friction"] == pytest.approx(0.1)
        finally:
            env.close()
    np.testing.assert_allclose(advanced[1], 0.5 * advanced[0], rtol=0.05)
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_array_equal(raw[0], raw[1])


def test_uniform_training_target_covers_the_window_and_leaves_the_exam_alone():
    from guidewire_rl.env.two_device.route import Route
    from guidewire_rl.env.two_device.targets import RouteWindowTarget
    from guidewire_rl.env.vessels.biro_phantom import BiroPhantomArch
    tree = BiroPhantomArch(version="rebuilt_cut")
    fluoro = SimpleNamespace(image_rot_zx=[61.0, -21.0], image_center=[0, 0, 0], field_of_view=None)
    kwargs = dict(past_fork="v8", depth_min_mm=35.0, depth_max_mm=45.0)
    uniform = RouteWindowTarget(tree, fluoro, split="train", sampling="uniform", **kwargs)
    route = Route.of(tree)
    depths = []
    for seed in range(200):
        uniform.reset(seed=seed)
        depths.append(uniform.depth_mm)
        at = route.s_of_fork("v8") + uniform.depth_mm
        np.testing.assert_allclose(uniform.coordinates_vessel, route.point_at(at), atol=1e-9)
    assert min(depths) >= 35.0 and max(depths) <= 45.0 and len(set(depths)) == 200
    assert min(depths) < 35.5 and max(depths) > 44.5                # the whole window, ends included
    grid = RouteWindowTarget(tree, fluoro, split="eval", **kwargs)
    same = RouteWindowTarget(tree, fluoro, split="eval", sampling="uniform", **kwargs)
    for seed in range(20):                                          # the exam's points do not move
        grid.reset(seed=seed)
        same.reset(seed=seed)
        assert grid.depth_mm == same.depth_mm and grid.depth_mm in grid.candidate_depths
    with pytest.raises(ValueError):
        RouteWindowTarget(tree, fluoro, sampling="uniform", jitter_mm=0.5)
    with pytest.raises(ValueError):
        RouteWindowTarget(tree, fluoro, sampling="random")


def test_friction_config_is_the_delivered_runs_settings_plus_the_two_changes():
    config = load_config(FRICTION)
    parent = load_config("configs/two_devices_ica_dagger.yaml")
    env = config["env"]
    assert env["randomize"]["friction"] == {"range": [0.01, 0.3], "log": True, "nominal_fraction": 0.3}
    assert env["randomize"]["execution"] == {
        "delay_steps": [0.0, 2.0], "gain": [0.8, 1.2], "jitter": [0.0, 0.1], "deadband_push_mm_s": [0.0, 1.0],
        "deadband_rot_rad_s": [0.0, 0.1], "nominal_fraction": 0.3}
    assert config["evaluation"]["end_when_stalled"] is True and parent["evaluation"]["end_when_stalled"] is False
    assert parent["env"]["randomize"] is False
    assert env["scene"]["target"] == {**parent["env"]["scene"]["target"], "sampling": "uniform"}
    assert env["info"] == parent["env"]["info"] + ["ScenarioInfo"]
    assert env["observation"] == parent["env"]["observation"]       # the checkpoint's 120 values
    referee, parent_referee = env["monitors"][0], parent["env"]["monitors"][0]
    assert referee["thresholds"] == {**parent_referee["thresholds"], "target_mm": 3.0, "tier2_cath_past_split": 8.0}
    assert env["monitors"][1:] == parent["env"]["monitors"][1:]
    assert config["model"]["log_std_max"] == -2.3


@pytest.mark.sofa
@pytest.mark.upstream
def test_the_drawn_friction_is_the_solvers_and_the_episodes():
    from guidewire_rl.env import build_env
    cfg = load_config(FRICTION, ["env.shadow_expert=false"])
    cfg["env"]["randomize"] = {"friction": cfg["env"]["randomize"]["friction"]}
    env = build_env(cfg["env"])
    try:
        seen = []
        for seed in (11, 12, 13, 11):
            _, info = env.reset(seed=seed)
            mu = float(env.simulation.root.getObject("LCP").mu.value)
            assert mu == pytest.approx(info["scenario"]["friction"])
            assert 0.01 <= mu <= 0.3
            seen.append(mu)
        assert seen[0] == seen[3] and len(set(seen[:3])) > 1
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["scenario"]["friction"] == pytest.approx(seen[3])
        from guidewire_rl.train.metrics import EpisodeStats
        stats = EpisodeStats(info)
        stats.add(np.zeros(4), info)
        assert stats.fields()["scenario"] == {"friction": pytest.approx(seen[3])}
    finally:
        env.close()
