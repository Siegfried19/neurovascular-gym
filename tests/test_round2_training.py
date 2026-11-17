"""Round two of the two-device task, the trainer side, without SOFA (todo_list/two_devices_round2_changes.md
sections E-G): the start curriculum, the replay's materials / reward recomputation / protected
self-demonstrations, the scripted prefix booked into an episode (one env and worker processes), the
auxiliary heads' target scale, and the episode statistics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from guidewire_rl.env.two_device import reward_formula as rf
from guidewire_rl.train.curriculum import StartCurriculum
from guidewire_rl.train.env_pool import EnvPool, reset_command
from guidewire_rl.train.metrics import EpisodeStats, window_summary
from guidewire_rl.train.replay import (SOURCE_DEMO, SOURCE_POLICY, SOURCE_PREFIX, SOURCE_SELF_DEMO,
                                       EpisodeReplay)
from guidewire_rl.train.setup import seed_everything, start_run
from guidewire_rl.train.trainer import crossed

from conftest import tiny_config

GOAL = 60.0


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --------------------------------------------------------------------------- fake env ----
class PrefixController:
    """Stands in for the hand-written procedure: pushes until the env's ratio is reached, or gives
    up at ``give_up_at`` (an attribute of the env)."""

    def __init__(self, env) -> None:
        self.env, self.failed, self.failure_reason, self.stage = env, False, "", "approach"

    def reset(self, env=None) -> None:
        self.failed, self.failure_reason = False, ""

    def act(self, env=None) -> np.ndarray:
        if self.env.slow_prefix_s:
            time.sleep(self.env.slow_prefix_s)
        if self.env.give_up_at is not None and self.env.progress / GOAL >= self.env.give_up_at:
            self.failed, self.failure_reason = True, "budget:approach"
            return np.zeros(4, np.float32)
        return np.array([1.0, 0.0, 0.5, 0.0], np.float32)


class Round2Env:
    """Wire + catheter along a 1-D route with the round-two bookkeeping: progress_ratio, the reward's
    materials, the first tier, the lead; success at GOAL. ``slow_prefix_s``: seconds each prefix step
    takes (a SOFA prefix takes minutes; the pool must not count it as a hung worker)."""

    label_names = ["wire_tip_force", "catheter_tip_force"]

    def __init__(self, max_episode_steps: int = 40, give_up_at=None, sim_error_at_calls=(),
                 end_in_prefix: bool = False, slow_prefix_s: float = 0.0, shadow_until=None) -> None:
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.give_up_at = give_up_at
        self.sim_error_at_calls = set(sim_error_at_calls)
        self.end_in_prefix = end_in_prefix
        self.slow_prefix_s = float(slow_prefix_s)
        self.prefix_controller = PrefixController(self)
        self.calls = 0
        # a stand-in shadow expert (env.shadow_expert): before each step it "would do" shadow_label(t),
        # t = the episode's step count, and has no label from step ``shadow_until`` on (finished)
        self.shadow_until = shadow_until

    @staticmethod
    def shadow_label(t: int, until: int) -> np.ndarray:
        if t >= until:
            return np.full(4, np.nan, np.float32)
        return np.array([t / 100.0, -0.5, 0.25, 0.0], np.float32)

    def _obs(self):
        return np.array([self.progress / 100, self.d / 100, float(self.tier1), 0.0], np.float32).clip(-1, 1)

    def _info(self, materials=None, success=False, sim_error=False):
        return {"is_success": success, "simulation_error": sim_error, "phase": 0, "max_phase": 0,
                "progress_ratio": min(self.progress / GOAL, 1.0), "tier1": self.tier1, "d": self.d,
                "landmark_grid": 1 + min(9, int(10 * self.progress / GOAL)), "landmark_grids": 10,
                "events": {"wrong_vessel": 0, "dislodged": 0, "tier1": 0},
                "executed_action": np.zeros(4, np.float32),
                "labels": {"wire_tip_force": 0.1, "catheter_tip_force": 0.2},
                "reward_materials": (np.zeros(len(rf.FIELDS), np.float32) if materials is None else materials)}

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.progress, self.d, self.t, self.tier1 = 0.0, float(rng.uniform(0, 5)), 0, False
        return self._obs(), self._info()

    def step(self, action):
        self.calls += 1
        shadow = None if self.shadow_until is None else self.shadow_label(self.t, self.shadow_until)
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        gain = 3.0 + 2.0 * a[0]
        self.progress += gain
        self.d = float(np.clip(self.d + 2 * (a[0] - a[2]), -100, 100))
        self.t += 1
        materials = np.zeros(len(rf.FIELDS), np.float32)
        materials[rf.INDEX["progress_mm"]] = gain
        materials[rf.INDEX["cath_force_N"]] = 2.0 + 2.0 * max(a[2], 0.0)
        materials[rf.INDEX["lead_mm"]] = self.d
        if not self.tier1 and self.progress >= 0.7 * GOAL:
            self.tier1 = True
            materials[rf.INDEX["tier1"]] = 1.0
        success = self.progress >= GOAL
        materials[rf.INDEX["success"]] = float(success)
        sim_error = self.calls in self.sim_error_at_calls
        materials[rf.INDEX["sim_error"]] = float(sim_error)
        truncated = self.t >= self.max_episode_steps or (self.end_in_prefix and self.t >= 3)
        reward = float(rf.compute(materials, rf.Coefficients()))
        info = self._info(materials, success, sim_error)
        if shadow is not None:
            info.update(expert_action=shadow, expert_stage="done" if np.isnan(shadow).all() else "approach")
        return self._obs(), reward, bool(success), bool(truncated), info

    def close(self) -> None:
        pass


def round2_config(**training):
    config = tiny_config(**training)
    config["env"]["truncation"] = [{"type": "MaxSteps", "max_steps": 40}]
    config["env"]["reward_formula"] = {"type": "TwoDeviceFormula", "progress_ruler": "nearest"}
    config["aux_heads"] = [{"name": "wire_force", "label": "wire_tip_force", "loss": "huber", "weight": 1.0,
                            "output": "softplus"},
                           {"name": "cath_force", "label": "catheter_tip_force", "loss": "huber", "weight": 1.0,
                            "output": "identity", "target_scale": 10.0}]
    config["training"]["start_curriculum"] = {"bins": 10, "baseline_fraction": 0.3, "frontier_threshold": 0.5,
                                              "min_episodes": 2, "window_episodes": 10, "update_every": 20,
                                              "floor_weight": 0.05, "prefix_max_steps": 30}
    config["training"]["self_demos"] = {"enabled": True, "capacity": 2}
    config["training"]["total_steps"] = 200
    config["logging"]["print_interval"] = 50
    config["evaluation"]["interval"] = 100
    config["checkpoint"]["interval"] = 100
    return config


# ---------------------------------------------------------------------- curriculum ----
def test_start_curriculum_weights_frontier_and_grids():
    c = StartCurriculum(bins=10, baseline_fraction=0.3, floor_weight=0.05, update_every=20, min_episodes=3, seed=1)
    w = c.weights()
    assert w[0] == pytest.approx(0.3) and w[10] == pytest.approx(0.7) and np.all(w[1:10] == 0.0)
    assert c.grid_of(0.0) == 0 and c.grid_of(0.05) == 1 and c.grid_of(0.95) == 10 and c.grid_of(1.0) == 10
    grids = [c.sample()[0] for _ in range(400)]
    assert set(grids) == {0, 10} and 0.2 < np.mean(np.asarray(grids) == 0) < 0.4
    for _ in range(5):
        c.record(10, True)
    assert c.maybe_update(19) is None                           # not due yet
    move = c.maybe_update(25)
    assert move["frontier_before"] == 10 and move["frontier"] == 9 and c.frontier == 9
    w = c.weights()
    assert w[10] == pytest.approx(0.05) and w[9] == pytest.approx(0.65) and w[0] == pytest.approx(0.3)
    grid, ratio = next(s for s in (c.sample() for _ in range(200)) if s[0] == 9)
    assert 0.8 <= ratio < 0.9
    for _ in range(3):
        c.record(9, False)
    assert c.maybe_update(50) is None and c.frontier == 9      # not learned: stays
    state = c.state_dict()
    again = StartCurriculum(bins=10, baseline_fraction=0.3, floor_weight=0.05, update_every=20, min_episodes=3, seed=5)
    again.load_state_dict(state)
    assert again.frontier == 9 and again.rates()["9"] == pytest.approx(0.0)
    c.frontier = 1
    for _ in range(3):
        c.record(1, True)
    c.next_update = 0
    assert c.maybe_update(100)["frontier"] == 0 and c.weights()[0] == 1.0   # the curriculum is over
    with pytest.raises(ValueError):                  # the saved-state supply is built as 'self' (test_start_library.py)
        StartCurriculum(supply="snapshot")
    assert StartCurriculum(grid="landmarks").prefix_key == "landmark_grid"     # built 2026-11-10
    assert StartCurriculum.from_config(None, 0) is None
    with pytest.raises(KeyError):
        StartCurriculum.from_config({"bin": 4}, 0)


def test_crossed_catches_multiples_inside_a_burst():
    assert crossed(99, 100, 100) and crossed(90, 350, 100) and not crossed(100, 150, 100)
    assert not crossed(100, 100, 100)


# -------------------------------------------------------------------------- replay ----
def episode(length, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    materials = np.zeros((length, len(rf.FIELDS)), np.float32)
    materials[:, rf.INDEX["progress_mm"]] = scale * rng.uniform(0, 5, length)
    rewards = rf.compute(materials, rf.Coefficients())
    return dict(observations=rng.normal(size=(length + 1, 4)).astype(np.float32),
                actions=rng.uniform(-1, 1, (length, 4)).astype(np.float32), rewards=rewards,
                terminated=np.zeros(length, np.float32), labels={"f": np.ones(length, np.float32)},
                materials=materials)


def test_replay_recomputes_rewards_from_materials_with_the_coefficients_of_the_day():
    replay = EpisodeReplay(capacity=1000, observation_dim=4, action_dim=4, horizon=3, batch_size=16,
                           label_names=["f"], seed=0)
    stored = episode(20)
    replay.add(**stored)
    doubled = rf.Coefficients(progress=0.02)
    replay.configure(reward_fn=lambda m: rf.compute(m, doubled), n_materials=len(rf.FIELDS))
    batch = replay.sample()
    expected = rf.compute(stored["materials"], doubled)
    starts = replay._flat["start_in_episode"]
    assert batch.rewards.shape == (3, 16, 1)
    # every sampled reward is the recomputed one, not the stored one
    recomputed = set(np.round(expected, 6).tolist())
    assert all(round(float(r), 6) in recomputed for r in batch.rewards.reshape(-1))
    assert not np.allclose(replay._flat["rewards"], np.concatenate([stored["rewards"]]))
    with pytest.raises(ValueError):
        replay.add(**dict(episode(5), materials=None))              # materials are required now
    with pytest.raises(ValueError):
        replay.add(**dict(episode(5), materials=np.zeros((5, 3), np.float32)))
    state = replay.state_dict()
    back = EpisodeReplay.from_state(state)
    assert back.episodes[0]["materials"].shape == (20, len(rf.FIELDS))
    assert back.episodes[0]["source"].tolist() == [SOURCE_POLICY] * 20


def test_protected_episodes_stay_and_self_demos_are_capped():
    replay = EpisodeReplay(capacity=60, observation_dim=4, action_dim=4, horizon=3, batch_size=8,
                           label_names=["f"], seed=0)
    replay.configure(self_demo_capacity=2, demo_fraction=0.5)
    replay.add(**episode(10, seed=1), demo=True)                                     # a scripted demo
    for k in range(4):
        replay.add(**episode(10, seed=2 + k), protected=True, outcome="terminated")  # self demos
    kept = [e["protected"] and not e["demo"] for e in replay.episodes]
    assert replay.num_demo_episodes == 1 and replay.num_self_demo_episodes == 2 and sum(kept) == 2
    assert all(e["demo"] or e["protected"] for e in replay.episodes[-2:])             # the latest two kept
    for k in range(6):
        replay.add(**episode(10, seed=10 + k))                                      # plain episodes
    assert replay.size <= 60 and replay.num_demo_episodes == 1 and replay.num_self_demo_episodes == 2
    assert replay.episodes[0]["demo"]                                                 # the demo never leaves
    assert len(replay.episodes) == 6 and sum(not e["protected"] for e in replay.episodes) == 3   # the oldest plain ones went
    # the quota draws from scripted and self demos alike
    episodes_idx, _ = replay.sample_indices(64)
    quota = [replay.episodes[i]["demo"] or replay.episodes[i]["protected"] for i in episodes_idx[:32]]
    assert all(quota)
    sources = replay.episodes[1]["source"]
    assert sources.dtype == np.int8 and sources.tolist() == [SOURCE_POLICY] * 10
    # in a batch the kept self-demonstrations read SOURCE_SELF_DEMO (training.self_bc_coef), the
    # scripted demo and the plain episodes keep theirs; the stored episodes are not rewritten
    flat = replay._flat["source"]
    offsets = replay._flat["trans_offsets"]
    for i, e in enumerate(replay.episodes):
        want = SOURCE_SELF_DEMO if (e["protected"] and not e["demo"]) else int(e["source"][0])
        assert set(flat[offsets[i]: offsets[i] + len(e["actions"])].tolist()) == {want}
    assert sum(int(flat[offsets[i]]) == SOURCE_SELF_DEMO for i in range(len(replay.episodes))) == 2
    replay.add(**episode(4, seed=99), source=np.full(4, SOURCE_PREFIX, np.int8))
    assert replay.episodes[-1]["source"].tolist() == [SOURCE_PREFIX] * 4
    assert replay.episodes[0]["source"].tolist() == [SOURCE_DEMO] * 10


# ------------------------------------------------------------------------- trainer ----
def test_prefix_episodes_are_booked_and_the_curriculum_moves(tmp_path):
    config = round2_config()
    config["training"]["start_curriculum"]["baseline_fraction"] = 0.0
    env = Round2Env(max_episode_steps=40)
    trainer = start_run(config, env, tmp_path / "run", quiet=True)
    assert trainer.start_curriculum is not None and trainer.replay.reward_fn is not None
    trainer.train()
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    prefixes = [e for e in events if e["event"] == "prefix"]
    # in the last grid the script may finish the task before the handover: such an episode is a
    # scripted demonstration, the others are handed over
    assert prefixes and {p["outcome"] for p in prefixes} <= {"handover", "ended"}
    handed = [e for e in episodes if not e["scripted"]]
    assert handed and all(p["grid"] == 10 and 0.9 <= p["ratio"] <= 1.0 for p in prefixes)   # the frontier grid
    assert all(e["start_grid"] == 10 and e["prefix_steps"] > 0 for e in episodes)
    assert all(e["prefix_steps"] < e["length"] for e in handed)                 # the policy went on
    assert all(e["prefix_steps"] == e["length"] for e in episodes if e["scripted"])
    # the prefix's steps are in the replay, marked, and the step count includes them
    sources = np.concatenate([e["source"] for e in trainer.replay.episodes])
    assert (sources == SOURCE_PREFIX).sum() == sum(e["prefix_steps"] for e in episodes)
    assert trainer.step >= config["training"]["total_steps"]
    # scripted episodes are demonstrations (protected), not the policy's successes
    scripted = [e for e in trainer.replay.episodes if e["demo"]]
    assert len(scripted) == sum(e["scripted"] for e in episodes)
    assert trainer.success_count == sum(e["success"] for e in handed)
    # successes from the frontier grid moved it
    assert any(e["event"] == "start_frontier" for e in events)
    assert trainer.start_curriculum.frontier < 10
    # self-demonstrations: the policy's successful episodes kept, capped at 2
    assert trainer.replay.num_self_demo_episodes == 2
    # every episode carries materials of the formula's width, and the stored reward is the formula's
    e0 = trainer.replay.episodes[0]
    assert e0["materials"].shape == (len(e0["actions"]), len(rf.FIELDS))
    np.testing.assert_allclose(e0["rewards"], rf.compute(e0["materials"], rf.Coefficients()), atol=1e-6)
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert "start_frontier" in metrics[-1] and "tier1_rate" in metrics[-1]
    assert "success_by_start_grid" in metrics[-1]


def test_a_prefix_that_gives_up_hands_over_where_it_stopped(tmp_path):
    config = round2_config()
    config["training"]["start_curriculum"]["baseline_fraction"] = 0.0
    trainer = start_run(config, Round2Env(give_up_at=0.5), tmp_path / "run", quiet=True)
    trainer.train()
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    prefixes = [e for e in events if e["event"] == "prefix"]
    assert prefixes and all(p["outcome"] == "gave_up" and p["script_failure"] == "budget:approach" for p in prefixes)
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert all(e["start_grid"] == 6 for e in episodes)              # booked where it really started (0.5-0.6)


def test_a_prefix_that_ends_or_breaks_the_episode_is_closed_and_the_env_reset_again(tmp_path):
    config = round2_config()
    config["training"]["start_curriculum"]["baseline_fraction"] = 0.0
    config["training"]["total_steps"] = 60
    trainer = start_run(config, Round2Env(end_in_prefix=True), tmp_path / "ended", quiet=True)
    trainer.train()
    events = read_jsonl(tmp_path / "ended" / "events.jsonl")
    assert all(p["outcome"] == "ended" for p in events if p["event"] == "prefix")
    episodes = read_jsonl(tmp_path / "ended" / "episodes.jsonl")
    assert episodes and all(e["end"] == "truncated" and e["length"] == 3 and e["scripted"] for e in episodes)
    assert trainer.replay.num_demo_episodes == len(episodes) and trainer.success_count == 0
    trainer = start_run(config, Round2Env(sim_error_at_calls=range(2, 10_000, 2)), tmp_path / "broken", quiet=True)
    trainer.train()
    events = read_jsonl(tmp_path / "broken" / "events.jsonl")
    assert any(p["event"] == "prefix" and p["outcome"] == "simulation_error" for p in events)
    assert any(p["event"] == "dropped_step" and p["reason"] == "simulation_error" for p in events)
    assert trainer.env_errors > 0


def test_prefixes_run_in_the_worker_processes_too(tmp_path):
    config = round2_config()
    config["training"]["start_curriculum"].update(baseline_fraction=0.0, bins=4)   # grid 4: ratios 0.75-1
    config["training"]["total_steps"] = 120
    pool = EnvPool(Round2Env, kwargs={"max_episode_steps": 40}, num_envs=2, seed=0, timeout=120.0)
    try:
        trainer = start_run(config, pool, tmp_path / "pool", quiet=True)
        trainer.train()
    finally:
        pool.close()
    events = read_jsonl(tmp_path / "pool" / "events.jsonl")
    prefixes = [e for e in events if e["event"] == "prefix"]
    assert prefixes and all(p["outcome"] in ("handover", "ended") and "env" in p for p in prefixes)
    assert any(p["outcome"] == "handover" for p in prefixes)
    episodes = read_jsonl(tmp_path / "pool" / "episodes.jsonl")
    # booked in the frontier grid (4, then 3 once the frontier moved on the policy's successes)
    assert all(e["prefix_steps"] > 0 and e["start_grid"] in (3, 4) for e in episodes)
    assert episodes[0]["start_grid"] == 4
    assert reset_command(3, None, 0.5, 30) == ("reset", {"seed": 3, "prefix_ratio": 0.5, "prefix_max_steps": 30})
    assert reset_command(3) == ("reset", 3)


def test_landmark_grids_draw_hand_over_and_book_by_the_referees_grid(tmp_path):
    """``grid: landmarks``: a drawn grid k is handed over when the env's ``landmark_grid`` reaches
    k, the episode is booked under the grid reached, in one env and in the worker processes."""
    curriculum = StartCurriculum(bins=10, grid="landmarks", baseline_fraction=0.0, seed=1)
    assert curriculum.prefix_key == "landmark_grid"
    k, value = curriculum.sample()
    assert k == 10 and value == 10.0                                    # the frontier starts at the last grid
    assert curriculum.grid_of(7.0) == 7 and curriculum.grid_of(0.0) == 0 and curriculum.grid_of(12.0) == 10
    curriculum.check_grids({"landmark_grids": 10})
    with pytest.raises(ValueError, match="bins: 9"):
        curriculum.check_grids({"landmark_grids": 9})
    with pytest.raises(ValueError):
        StartCurriculum(grid="milestones")
    assert reset_command(3, None, 4.0, 30, "landmark_grid") == (
        "reset", {"seed": 3, "prefix_ratio": 4.0, "prefix_max_steps": 30, "prefix_key": "landmark_grid"})
    config = round2_config()
    config["training"]["start_curriculum"].update(baseline_fraction=0.0, grid="landmarks")
    config["training"]["total_steps"] = 120
    trainer = start_run(config, Round2Env(max_episode_steps=40), tmp_path / "one", quiet=True)
    trainer.train()
    prefixes = [e for e in read_jsonl(tmp_path / "one" / "events.jsonl") if e["event"] == "prefix"]
    assert prefixes and prefixes[0]["ratio_asked"] == 10.0                # the last grid first
    assert all(p["ratio_asked"] == int(p["ratio_asked"]) >= 1 for p in prefixes)   # whole grids asked
    handed = [p for p in prefixes if p["outcome"] == "handover"]
    assert handed and all(p["ratio"] >= p["ratio_asked"] and p["grid"] == min(10, int(round(p["ratio"])))
                          for p in handed)
    pool = EnvPool(Round2Env, kwargs={"max_episode_steps": 40}, num_envs=2, seed=0, timeout=120.0)
    try:
        trainer = start_run(config, pool, tmp_path / "pool", quiet=True)
        trainer.train()
    finally:
        pool.close()
    prefixes = [e for e in read_jsonl(tmp_path / "pool" / "events.jsonl") if e["event"] == "prefix"]
    assert prefixes and any(p["outcome"] == "handover" for p in prefixes)
    assert all(p["grid"] == min(10, int(round(p["ratio"]))) for p in prefixes if p["outcome"] == "handover")


def test_a_slow_prefix_is_not_a_hung_worker(tmp_path):
    """A scripted prefix in SOFA takes minutes; the pool gives a reset that runs one
    ``timeout + prefix_max_steps x prefix_step_seconds`` seconds, not the plain step timeout
    (the first launch of round two lost every prefix reset to the 120 s step timeout)."""
    from guidewire_rl.train.env_pool import EnvPool
    config = round2_config()
    config["training"]["start_curriculum"].update(baseline_fraction=0.0, bins=2, prefix_max_steps=12)
    config["training"]["total_steps"] = 40
    config["training"]["step_timeout_s"] = 1.0
    # 0.25 s a prefix step, up to 12 steps: 3 s, far past the 1 s step timeout; allowed = 1 + 12 x 0.5
    pool = EnvPool(Round2Env, kwargs={"max_episode_steps": 40, "slow_prefix_s": 0.25}, num_envs=2, seed=0,
                   timeout=1.0, prefix_step_seconds=0.5)
    try:
        assert pool._allowed_seconds("reset", {"seed": 1, "prefix_ratio": 0.9, "prefix_max_steps": 12}) == 7.0
        assert pool._allowed_seconds("reset", 1) == 1.0 and pool._allowed_seconds("step", np.zeros(4)) == 1.0
        trainer = start_run(config, pool, tmp_path / "slow", quiet=True)
        trainer.train()
        assert pool.restarts == 0
    finally:
        pool.close()
    events = read_jsonl(tmp_path / "slow" / "events.jsonl")
    assert not any(e["event"] in ("worker_restarted", "reset_failed") for e in events)
    assert any(e["event"] == "prefix" for e in events)


def test_without_the_round_two_parts_the_trainer_books_as_before(tmp_path):
    config = tiny_config()
    config["env"]["truncation"] = [{"type": "MaxSteps", "max_steps": 40}]
    config["aux_heads"][0]["label"] = "wire_tip_force"
    trainer = start_run(config, Round2Env(), tmp_path / "plain", quiet=True)
    assert trainer.start_curriculum is None and trainer.replay.reward_fn is None and not trainer.self_demos
    trainer.train()
    episodes = read_jsonl(tmp_path / "plain" / "episodes.jsonl")
    assert episodes and all("start_grid" not in e for e in episodes)
    assert all(e["protected"] is False for e in trainer.replay.episodes)


# ------------------------------------------------- shadow expert, pretraining (2026-11-13) ----
SHADOW_UNTIL = 15


def check_targets(replay, until=SHADOW_UNTIL):
    """Every stored step's target: the executed action on scripted (prefix) steps, else the env's
    shadow label of that step (step j of an episode = the env's step count j before it), NaN from
    ``until`` on."""
    checked = {"prefix": 0, "label": 0, "none": 0}
    for e in replay.episodes:
        for j, (source, action, target) in enumerate(zip(e["source"], e["actions"], e["expert"])):
            if source == SOURCE_PREFIX:
                np.testing.assert_array_equal(target, action)
                checked["prefix"] += 1
            elif j < until:
                np.testing.assert_array_equal(target, Round2Env.shadow_label(j, until))
                checked["label"] += 1
            else:
                assert np.isnan(target).all()
                checked["none"] += 1
    return checked


def test_shadow_labels_travel_from_the_env_through_the_trainer_to_the_batch(tmp_path):
    # seeded here: which episodes start from a prefix depends on the global random state, and without
    # this the test took whatever the tests before it left behind (it failed once tests were added
    # ahead of it, 2026-11-15: no episode from the start that outlived its labels)
    seed_everything(0)
    config = round2_config()
    config["training"]["start_curriculum"]["baseline_fraction"] = 0.5     # some prefixes, some from the start
    trainer = start_run(config, Round2Env(shadow_until=SHADOW_UNTIL), tmp_path / "one", quiet=True)
    trainer.train()
    checked = check_targets(trainer.replay)
    assert all(count > 0 for count in checked.values()), checked
    trainer.replay.batch_size = 256
    batch = trainer.replay.sample()
    assert batch.expert.shape == batch.actions.shape
    finite = torch.isfinite(batch.expert).all(dim=-1)
    assert finite.any() and (~finite).any()


@pytest.mark.parametrize("async_envs", [False, True])
def test_shadow_labels_travel_through_the_worker_processes(tmp_path, async_envs):
    config = round2_config(async_envs=async_envs)
    config["training"]["start_curriculum"]["baseline_fraction"] = 0.5
    config["training"]["total_steps"] = 120
    pool = EnvPool(Round2Env, kwargs={"max_episode_steps": 40, "shadow_until": 5}, num_envs=2,
                   seed=0, timeout=120.0)
    try:
        trainer = start_run(config, pool, tmp_path / "pool", quiet=True)
        trainer.train()
    finally:
        pool.close()
    checked = check_targets(trainer.replay, until=5)
    assert checked["label"] > 0 and checked["none"] > 0, checked


def write_demos(folder: Path, episodes: int = 3, length: int = 10) -> None:
    """Scripted episodes of the fake env in the demonstration file format (no ``expert_action``: as
    every file recorded before 2026-11-13)."""
    folder.mkdir(parents=True)
    env = Round2Env(max_episode_steps=40)
    for k in range(episodes):
        obs, info = env.reset(seed=k)
        rows = dict(obs=[obs], action=[], reward=[], terminated=[], materials=[], f=[], c=[])
        for _ in range(length):
            action = np.array([0.2, 0.0, 0.1, 0.0], np.float32)
            obs, reward, terminated, _, info = env.step(action)
            rows["obs"].append(obs)
            rows["action"].append(action)
            rows["reward"].append(reward)
            rows["terminated"].append(terminated)
            rows["materials"].append(info["reward_materials"])
            rows["f"].append(info["labels"]["wire_tip_force"])
            rows["c"].append(info["labels"]["catheter_tip_force"])
        np.savez(folder / f"episode_{k}.npz", obs=np.stack(rows["obs"]), action=np.stack(rows["action"]),
                 reward=np.asarray(rows["reward"]), terminated=np.asarray(rows["terminated"]),
                 materials=np.stack(rows["materials"]), **{"labels/wire_tip_force": np.asarray(rows["f"]),
                                                             "labels/catheter_tip_force": np.asarray(rows["c"])})


def pretrain_config(demos: Path, **training):
    config = round2_config(**training)
    config["training"]["seed_steps"] = 0
    config["training"].pop("start_curriculum")
    config["training"]["demos"] = {"dir": str(demos), "fraction": 0.25}
    config["training"]["total_steps"] = 30
    return config


def test_pretrain_updates_run_before_the_first_env_step_and_the_policy_acts_from_step_one(tmp_path):
    """training.pretrain_updates N: N updates on the demonstrations before the first env step, which
    replace the initial_updates burst; seed_steps 0: every action comes from the agent."""
    write_demos(tmp_path / "demos")
    config = pretrain_config(tmp_path / "demos", pretrain_updates=7)
    env = Round2Env(shadow_until=SHADOW_UNTIL)
    trainer = start_run(config, env, tmp_path / "run", quiet=True)
    # demonstration files without targets: a demonstration step's target is its action
    demos = [e for e in trainer.replay.episodes if e["demo"]]
    assert len(demos) == 3 and all(np.array_equal(e["expert"], e["actions"]) for e in demos)
    seen, acted = [], []
    plain_step, plain_act = env.step, trainer._act
    env.step = lambda action: (seen.append(trainer.agent.update_count), plain_step(action))[1]
    trainer._act = lambda *a, **k: (acted.append(1), plain_act(*a, **k))[1]
    trainer.train()
    assert seen[0] == 7                                        # all 7 before the first step
    assert seen == [7 + k for k in range(len(seen))]           # then one per step: no initial burst
    assert len(acted) == len(seen) == trainer.step             # the agent chose every action
    assert trainer.agent.update_count == 7 + trainer.step
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    pretrain = [e for e in events if e["event"] == "pretrain"]
    assert len(pretrain) == 1 and pretrain[0]["step"] == 0 and pretrain[0]["updates"] == 7
    assert pretrain[0]["demo_episodes"] == 3


def test_seed_steps_0_without_pretraining_bursts_after_the_first_step_and_pretraining_needs_a_replay(tmp_path):
    write_demos(tmp_path / "demos")
    config = pretrain_config(tmp_path / "demos")
    env = Round2Env()
    trainer = start_run(config, env, tmp_path / "run", quiet=True)
    seen, plain_step = [], env.step
    env.step = lambda action: (seen.append(trainer.agent.update_count), plain_step(action))[1]
    trainer.train()
    initial = config["training"]["initial_updates"]
    assert seen[:3] == [0, initial, initial + 1]                # the burst right after the first step
    assert not any(e["event"] == "pretrain" for e in read_jsonl(tmp_path / "run" / "events.jsonl"))
    # nothing to learn from before the first step: fail loudly
    config = pretrain_config(tmp_path / "nothing_here", pretrain_updates=5)
    trainer = start_run(config, Round2Env(), tmp_path / "empty", quiet=True)
    with pytest.raises(RuntimeError, match="pretrain_updates"):
        trainer.train()


@pytest.mark.parametrize("async_envs", [False, True])
def test_pretrain_updates_in_the_parallel_trainers(tmp_path, async_envs):
    write_demos(tmp_path / "demos")
    config = pretrain_config(tmp_path / "demos", pretrain_updates=6, async_envs=async_envs)
    config["training"]["total_steps"] = 40
    pool = EnvPool(Round2Env, kwargs={"max_episode_steps": 40}, num_envs=2, seed=0, timeout=120.0)
    try:
        trainer = start_run(config, pool, tmp_path / "pool", quiet=True)
        trainer.train()
    finally:
        pool.close()
    events = read_jsonl(tmp_path / "pool" / "events.jsonl")
    assert [e["step"] for e in events if e["event"] == "pretrain"] == [0]
    assert trainer.agent.update_count == 6 + trainer.step          # no initial_updates burst


# ----------------------------------------------------------------------- aux heads ----
def test_aux_head_target_scale_divides_the_label():
    from guidewire_rl.model.aux_heads import AuxHeadSpec, head_loss
    spec = AuxHeadSpec(name="c", label="clearance", loss="huber", target_scale=10.0)
    prediction = torch.full((3, 2, 1), 0.5)
    target = torch.full((3, 2, 1), 5.0)
    assert torch.allclose(head_loss(spec, prediction, target), torch.zeros(3, 2, 1))
    with pytest.raises(ValueError):
        AuxHeadSpec(name="c", label="clearance", target_scale=0.0)


# --------------------------------------------------------------------------- stats ----
def test_episode_stats_carry_the_round_two_numbers():
    stats = EpisodeStats({"progress_ratio": 0.05, "tier1": False})
    stats.add(np.zeros(4), {"progress_ratio": 0.4, "tier1": False, "d": 20.0})
    stats.add(np.zeros(4), {"progress_ratio": 0.8, "tier1": True, "d": 33.0})
    fields = stats.fields()
    assert fields["tier1"] is True and fields["progress_end"] == pytest.approx(0.8) and fields["lead_end"] == 33.0
    records = [dict(success=False, **{"return": 0.0}, length=5, tier1=True, progress_end=0.55, lead_end=20.0,
                    start_grid=6),
               dict(success=True, **{"return": 1.0}, length=5, tier1=True, progress_end=1.0, lead_end=35.0,
                    start_grid=6),
               dict(success=False, **{"return": 0.0}, length=5, tier1=False, progress_end=0.12, lead_end=5.0,
                    start_grid=0)]
    summary = window_summary(records)
    assert summary["tier1_rate"] == pytest.approx(2 / 3) and summary["lead_end_p50"] == 20.0
    assert summary["progress_end_hist"] == {"2": 1, "6": 1, "done": 1}
    assert summary["success_by_start_grid"] == {"0": 0.0, "6": 0.5}
