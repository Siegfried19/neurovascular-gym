"""Trainer changes for the two-device round (todo_list/two_devices_training.md, WP-C).

Everything here runs on fake envs (no SOFA): the cutoff mixture and how the
cutoff reaches the env (one env and worker processes), phase-stratified
sampling, demonstrations in the replay, the step timeout replacing a hung
worker, NaN episodes, the metrics files, the evaluation report and episode
files, the residual box; and, with all of it switched off, the training run is
bit for bit the one the code gave before these changes (golden fingerprints)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.model import build_agent
from guidewire_rl.train.checkpoint import load_checkpoint
from guidewire_rl.train.curriculum import (CutoffMixture, ResidualSchedule, frontier_index,
                                           mixture_weights)
from guidewire_rl.train.env_pool import EnvPool, reset_command, step_timeout
from guidewire_rl.train.evaluate import report, run_episode, wilson_interval
from guidewire_rl.train.metrics import EpisodeStats, reach_rates, window_summary
from guidewire_rl.train.parallel_trainer import ParallelTrainer
from guidewire_rl.train.replay import EpisodeReplay, phase_stratum
from guidewire_rl.train.setup import seed_everything, start_run

from conftest import tiny_config
from fakes import FakeEnv

CHECKPOINTS = (10.0, 25.0, 45.0, 70.0)      # progress (mm) where phases 1-4 begin
GOAL = 90.0


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# ------------------------------------------------------------------ fake env --
class FakeTwoDeviceEnv:
    """Wire + catheter along a 1-D route. Progress grows ~3 mm per step plus the
    pushes; the phase is how many of ``CHECKPOINTS`` it passed; success at
    ``GOAL`` (only possible when nothing cuts the episode first). Reports what
    WP-A's env reports: phase, max_phase, per-step events, shield_delta, d,
    executed and nominal action; truncates at ``phase >= cutoff``. Labels echo
    the cutoff so tests can see which one the env really had."""

    label_names = ["wire_tip_force", "catheter_tip_force", "cutoff_seen"]

    def __init__(self, max_episode_steps: int = 40, error_at_calls=(), hang_at_calls=(),
                 sim_error_at_calls=(), default_cutoff: int = 5, hang_once_file=None,
                 breach_at_calls=()) -> None:
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.error_at_calls = set(error_at_calls)
        self.hang_at_calls = set(hang_at_calls)
        self.sim_error_at_calls = set(sim_error_at_calls)
        self.breach_at_calls = set(breach_at_calls)       # steps reported as a wall breach
        self.hang_once_file = hang_once_file      # hang only while this file does not exist yet
        self.cutoff = int(default_cutoff)
        self.cutoff_calls = []
        self.executed = []
        self.calls = 0

    def set_cutoff(self, k: int) -> None:
        self.cutoff = int(k)
        self.cutoff_calls.append(int(k))

    def _phase(self) -> int:
        return int(sum(self.progress >= c for c in CHECKPOINTS))

    def _obs(self) -> np.ndarray:
        return np.array([self.progress / 100, self.phase / 4, self.d / 100, self.noise, 0.0, 0.0],
                        dtype=np.float32).clip(-1, 1)

    def _nominal(self):
        return [0.2, -0.3, 0.1, 0.5] if self.phase == 1 else None

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.progress, self.d, self.t = 0.0, float(rng.uniform(-5, 5)), 0
        self.noise = float(rng.uniform(-0.5, 0.5))
        self.phase = self.max_phase = 0
        info = {"is_success": False, "simulation_error": False, "phase": 0, "max_phase": 0,
                "cutoff": self.cutoff, "nominal_action": np.zeros(4, np.float32),
                "nominal_available": False}
        return self._obs(), info

    def step(self, action):
        self.calls += 1
        if self.calls in self.error_at_calls:
            raise RuntimeError("fake simulator failure")
        if self.calls in self.hang_at_calls:
            if self.hang_once_file is None or not os.path.exists(self.hang_once_file):
                if self.hang_once_file is not None:
                    Path(self.hang_once_file).touch()
                time.sleep(3600)
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        executed = a.copy()
        executed[2] = min(executed[2], 0.8)              # the shield caps catheter pushes
        self.executed.append(executed)
        self.progress = max(0.0, self.progress + 3.0 + 2.0 * a[0] + 1.0 * executed[2])
        self.d = float(np.clip(self.d + 4 * (a[0] - executed[2]), -100, 100))
        self.t += 1
        self.phase = self._phase()
        self.max_phase = max(self.max_phase, self.phase)
        success = self.phase == 4 and self.progress >= GOAL
        terminated = bool(success)
        truncated = bool(self.t >= self.max_episode_steps or self.phase >= self.cutoff)
        info = {"is_success": success, "simulation_error": self.calls in self.sim_error_at_calls,
                "wall_breached": self.calls in self.breach_at_calls,
                "phase": self.phase, "max_phase": self.max_phase, "cutoff": self.cutoff,
                "failure_reason": "outside_wire" if self.t == self.max_episode_steps - 1 else "",
                "nominal_action": np.zeros(4, np.float32), "nominal_available": False,
                "events": {"wrong_vessel": int(a[1] > 0.9), "naked_advance": int(a[3] < -0.9)},
                "shield_delta": float(a[2] - executed[2]), "d": self.d,
                "executed_action": executed,
                "labels": {"wire_tip_force": float(abs(a[0])), "catheter_tip_force": float(abs(a[2])),
                           "cutoff_seen": float(self.cutoff)}}
        if self._nominal() is not None:
            info["nominal_action"], info["nominal_available"] = self._nominal(), True
        terminated = terminated or bool(info["failure_reason"])
        return self._obs(), 0.01 * (3.0 + 2.0 * a[0]) + float(success), terminated, truncated, info

    def close(self) -> None:
        pass


def two_device_config(**training):
    config = tiny_config(**training)
    config["env"]["truncation"] = [{"type": "MaxSteps", "max_steps": 40}]
    config["aux_heads"][0]["label"] = "wire_tip_force"
    config["training"]["cutoff_mixture"] = {"phases": [2, 3, 4, 5], "min_weight": 0.1,
                                            "frontier_min": 0.4, "window_episodes": 50,
                                            "update_every": 40}
    config["training"]["total_steps"] = 120
    config["logging"]["print_interval"] = 30
    config["evaluation"]["interval"] = 60
    config["checkpoint"]["interval"] = 60
    return config


def pool_of(fake=None, num_envs=2, config=None, **kwargs):
    timeout = step_timeout(config) if config is not None else 120.0
    return EnvPool(FakeTwoDeviceEnv, kwargs=fake or {}, num_envs=num_envs, seed=0,
                   timeout=timeout, **kwargs)


def train_and_close(trainer):
    try:
        return trainer.train()
    finally:
        if hasattr(trainer.env, "close"):
            trainer.env.close()


# ------------------------------------------------------- cutoff mixture --
def test_mixture_weights_start_with_the_frontier_at_its_floor():
    np.testing.assert_allclose(mixture_weights([0, 0, 0, 0], min_weight=0.1, frontier_min=0.4),
                               [0.4, 0.2, 0.2, 0.2])
    # everything learned: equal raw weights, the earliest is the frontier
    np.testing.assert_allclose(mixture_weights([1, 1, 1, 1], min_weight=0.1, frontier_min=0.4),
                               [0.4, 0.2, 0.2, 0.2])


def test_mixture_weights_follow_one_minus_reach_with_floors():
    # raw 1 - rate = [0.1, 0.5, 0.9, 1.0]; frontier = cutoff 5 (lowest rate);
    # cutoff 2 is lifted to 0.1, cutoff 5 to 0.4, the rest share 0.5 as 0.5 : 0.9
    w = mixture_weights([0.9, 0.5, 0.1, 0.0], min_weight=0.1, frontier_min=0.4)
    np.testing.assert_allclose(w, [0.1, 0.5 * 0.5 / 1.4, 0.5 * 0.9 / 1.4, 0.4])
    assert abs(w.sum() - 1) < 1e-12
    # no floor binds: plain proportions
    w = mixture_weights([0.5, 0.5, 0.5, 0.2], min_weight=0.1, frontier_min=0.3)
    np.testing.assert_allclose(w, np.array([0.5, 0.5, 0.5, 0.8]) / 2.3)


def test_frontier_is_the_lowest_reach_rate_ties_to_the_earliest():
    assert frontier_index([0.9, 0.0, 0.0, 0.0]) == 1
    assert frontier_index([0.9, 0.3, 0.1, 0.05]) == 3
    assert frontier_index([0.0, 0.0, 0.0, 0.0]) == 0
    with pytest.raises(ValueError):
        mixture_weights([0, 0, 0, 0], min_weight=0.3, frontier_min=0.4)    # floors > 1


def test_mixture_reach_rates_only_count_episodes_that_could_get_there():
    mix = CutoffMixture(window_episodes=100, update_every=10, seed=0)
    for _ in range(10):
        mix.record(2, 2, False)      # cut at 2: says nothing about 3, 4, success
    for max_phase, success in [(1, False), (2, False), (3, False), (4, False), (4, True)]:
        mix.record(5, max_phase, success)
    rates, counts = mix.reach_rates()
    assert counts == [15, 5, 5, 5]
    assert rates == pytest.approx([14 / 15, 3 / 5, 2 / 5, 1 / 5])
    assert mix.maybe_update(9) is None                        # not due yet
    change = mix.maybe_update(10)
    assert change["frontier"] == 5 and mix.next_update == 20
    np.testing.assert_allclose(mix.weights, mixture_weights(rates, min_weight=0.1, frontier_min=0.4))
    assert mix.maybe_update(15) is None and mix.maybe_update(23) is not None and mix.next_update == 30


def test_mixture_draws_follow_the_weights_and_round_trip():
    mix = CutoffMixture(seed=3)
    draws = [mix.sample() for _ in range(4000)]
    shares = [draws.count(k) / len(draws) for k in (2, 3, 4, 5)]
    np.testing.assert_allclose(shares, [0.4, 0.2, 0.2, 0.2], atol=0.03)
    mix.record(5, 3, False)
    copy_ = CutoffMixture(seed=99)
    copy_.load_state_dict(mix.state_dict())
    assert [copy_.sample() for _ in range(20)] == [mix.sample() for _ in range(20)]
    assert list(copy_.window) == list(mix.window)


def test_cutoff_reaches_the_env_before_every_reset_one_env(tmp_path):
    seed_everything(0)
    env = FakeTwoDeviceEnv()
    trainer = start_run(two_device_config(), env, tmp_path / "run", quiet=True)
    summary = trainer.train()
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert summary["total_env_steps"] == 120 and episodes
    for e in episodes:                      # the env had the cutoff the trainer drew
        assert e["cutoff"] in (2, 3, 4, 5) and e["cutoff_seen_max"] == e["cutoff"]
        if e["end"] == "truncated" and e["length"] < 40:
            assert e["max_phase"] == e["cutoff"]       # cut exactly at its cutoff
    # one set_cutoff per training episode started, plus 5 (never cut) before each evaluation episode
    evals = read_jsonl(tmp_path / "run" / "eval.jsonl")
    assert env.cutoff_calls.count(5) >= 2 * len(evals)
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    updates = [e for e in events if e["event"] == "cutoff_weights"]
    assert [e["step"] for e in updates] == [40, 80, 120]
    assert abs(sum(updates[0]["weights"].values()) - 1) < 1e-9
    payload = load_checkpoint(tmp_path / "run" / "checkpoints" / "step_120.pt")
    assert payload["cutoff_mixture"]["next_update"] == 160


def test_cutoff_reaches_the_worker_processes(tmp_path):
    seed_everything(0)
    trainer = start_run(two_device_config(), pool_of(num_envs=3), tmp_path / "run", quiet=True)
    assert isinstance(trainer, ParallelTrainer)
    train_and_close(trainer)
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert len({e["cutoff"] for e in episodes}) > 1
    assert all(e["cutoff_seen_max"] == e["cutoff"] for e in episodes)
    evals = read_jsonl(tmp_path / "run" / "eval.jsonl")
    assert evals and "max_phase_hist" in evals[0]
    resumed = start_run(two_device_config(), pool_of(num_envs=2), tmp_path / "run",
                        resume=tmp_path / "run" / "checkpoints" / "step_60.pt", quiet=True)
    assert resumed.mixture.next_update == 80 and resumed.mixture.window
    train_and_close(resumed)


def test_async_trainer_passes_cutoffs_too(tmp_path):
    seed_everything(0)
    config = two_device_config()
    config["training"]["async_envs"] = True
    train_and_close(start_run(config, pool_of(num_envs=2), tmp_path / "run", quiet=True))
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert all(e["cutoff_seen_max"] == e["cutoff"] for e in episodes)


class IgnoresCutoff(FakeTwoDeviceEnv):
    def set_cutoff(self, k: int) -> None:
        self.cutoff_calls.append(int(k))          # takes it, keeps cutting at 5


def test_a_cutoff_the_env_does_not_take_is_logged(tmp_path):
    seed_everything(0)
    start_run(two_device_config(), IgnoresCutoff(), tmp_path / "run", quiet=True).train()
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    mismatches = [e for e in events if e["event"] == "cutoff_mismatch"]
    assert mismatches and all(e["env_reports"] == 5 and e["drawn"] != 5 for e in mismatches)
    seed_everything(0)
    start_run(two_device_config(), FakeTwoDeviceEnv(), tmp_path / "ok", quiet=True).train()
    assert not [e for e in read_jsonl(tmp_path / "ok" / "events.jsonl") if e["event"] == "cutoff_mismatch"]


def test_mixture_needs_an_env_that_takes_cutoffs(tmp_path):
    config = two_device_config()
    config["aux_heads"][0]["label"] = "distal_force"
    with pytest.raises(ValueError, match="set_cutoff"):
        start_run(config, FakeEnv(max_episode_steps=40), tmp_path / "run", quiet=True)


def test_reset_command_is_unchanged_without_a_cutoff():
    assert reset_command(7) == ("reset", 7)
    assert reset_command(7, 3) == ("reset", {"seed": 7, "cutoff": 3})


# ----------------------------------------------------- stratified replay --
def make_phase_replay(lengths_by_phase, **configure):
    replay = EpisodeReplay(capacity=100_000, observation_dim=2, action_dim=1, horizon=3,
                           batch_size=64, seed=0)
    for phase, (count, length) in lengths_by_phase.items():
        for _ in range(count):
            obs = np.full((length + 1, 2), phase, dtype=np.float32)
            replay.add(obs, np.zeros((length, 1)), np.zeros(length), np.zeros(length),
                       phase=np.full(length + 1, phase))
    if configure:
        replay.configure(**configure)
    return replay


def start_phases(replay, batch_size=None):
    episodes, starts = replay.sample_indices(batch_size)
    return np.array([replay.episodes[e]["phase"][s] for e, s in zip(episodes, starts)])


def test_phase_strata_are_the_stretches_between_cutoffs():
    assert phase_stratum([0, 1, 2, 3, 4, 7, -1]).tolist() == [0, 0, 1, 2, 3, 3, 0]


def test_stratified_batches_give_each_big_stratum_a_quarter():
    # starts per stratum: {0,1}: 2*(40-2)+38 = 114, {2}: 38, {3}: 38, {4}: 38
    replay = make_phase_replay({0: (2, 40), 1: (1, 40), 2: (1, 40), 3: (1, 40), 4: (1, 40)},
                               stratify_by="phase", stratify_min_samples=30)
    assert replay.stratum_sizes() == [114, 38, 38, 38]
    phases = start_phases(replay)
    counts = np.bincount(phase_stratum(phases), minlength=4)
    assert counts.tolist() == [16, 16, 16, 16]


def test_a_small_stratum_is_not_blown_up():
    # stratum {4} has 18 starts (< 30): its quarter is drawn from the whole replay
    replay = make_phase_replay({0: (6, 40), 2: (1, 40), 3: (1, 40), 4: (1, 20)},
                               stratify_by="phase", stratify_min_samples=30)
    strata = np.concatenate([phase_stratum(start_phases(replay)) for _ in range(200)])
    share = np.bincount(strata, minlength=4) / len(strata)
    total = sum(replay.stratum_sizes())
    assert share[3] == pytest.approx(0.25 * 18 / total, abs=0.01)       # natural share of a quarter
    assert share[1] == pytest.approx(0.25 + 0.25 * 38 / total, abs=0.015)


def test_without_stratification_the_sampler_draws_what_it_always_drew():
    replay = make_phase_replay({0: (3, 30), 2: (2, 25), 4: (1, 12)})
    twin = np.random.default_rng(0)
    episodes, starts = replay.sample_indices(500)
    picks = twin.integers(0, replay.num_valid_starts, size=500)
    np.testing.assert_array_equal(episodes, replay._flat["episode_of_start"][picks])
    np.testing.assert_array_equal(starts, replay._flat["start_in_episode"][picks])


def test_phase_is_stored_per_step_and_survives_a_checkpoint(tmp_path):
    seed_everything(0)
    config = two_device_config()
    config["replay"] = {"stratify_by": "phase", "stratify_min_samples": 50}
    trainer = start_run(config, FakeTwoDeviceEnv(), tmp_path / "run", quiet=True)
    trainer.train()
    for e in trainer.replay.episodes:
        assert e["phase"].dtype == np.int8 and len(e["phase"]) == len(e["actions"]) + 1
        assert e["phase"][0] == 0 and np.all(np.diff(e["phase"]) >= 0)
    assert max(int(e["phase"].max()) for e in trainer.replay.episodes) >= 2
    restored = EpisodeReplay.from_state(load_checkpoint(
        tmp_path / "run" / "checkpoints" / "step_120.pt")["replay"])
    for a, b in zip(restored.episodes, trainer.replay.episodes):
        np.testing.assert_array_equal(a["phase"], b["phase"])
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert len(metrics[-1]["stratum_starts"]) == 4


# ----------------------------------------------------------- demonstrations --
def write_demos(folder, count, length=12, obs_dim=6, act_dim=4, labels=True, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(count):
        arrays = {"obs": rng.uniform(-1, 1, (length + 1, obs_dim)).astype(np.float32),
                  "action": rng.uniform(-1, 1, (length, act_dim)).astype(np.float32),
                  "reward": rng.uniform(0, 0.1, length).astype(np.float32),
                  "phase": np.minimum(np.arange(length + 1) // 3, 4).astype(np.int8),
                  "terminated": np.r_[np.zeros(length - 1), 1].astype(bool),
                  "truncated": np.zeros(length, dtype=bool)}
        if labels:
            for name in FakeTwoDeviceEnv.label_names:
                arrays[f"labels/{name}"] = rng.uniform(0, 1, length).astype(np.float32)
        np.savez(folder / f"episode_{k:03d}.npz", **arrays)


def test_demos_go_into_the_replay_and_hold_their_share(tmp_path):
    write_demos(tmp_path / "demos", 3)
    config = two_device_config()
    config["training"]["demos"] = {"dir": str(tmp_path / "demos"), "fraction": 0.25}
    config["training"]["replay_capacity"] = 150          # forces evictions during the run
    seed_everything(0)
    trainer = start_run(config, FakeTwoDeviceEnv(), tmp_path / "run", quiet=True)
    replay = trainer.replay
    assert replay.num_demo_episodes == 3 and replay.size == 36
    episodes, _ = replay.sample_indices()
    from_demos = sum(replay.episodes[e]["demo"] for e in episodes)
    assert from_demos >= math.ceil(0.25 * replay.batch_size)        # 2 of 8
    trainer.train()
    assert replay.num_demo_episodes == 3                   # never evicted
    assert replay.size <= 150 + 40
    events = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl") if e["event"] == "demos"]
    assert events[0]["loaded"] == 3 and events[0]["per_batch"] == 2
    # a resume finds them in the replay and does not load them again
    resumed = start_run(config, FakeTwoDeviceEnv(), tmp_path / "run",
                        resume=tmp_path / "run" / "checkpoints" / "step_120.pt", quiet=True)
    assert resumed.replay.num_demo_episodes == 3


def test_demo_files_are_checked_and_missing_labels_filled(tmp_path):
    replay = EpisodeReplay(capacity=1000, observation_dim=6, action_dim=4, horizon=3,
                           batch_size=8, label_names=FakeTwoDeviceEnv.label_names)
    from guidewire_rl.train.demos import load_demos
    write_demos(tmp_path / "nolabels", 2, labels=False)
    said = []
    outcome = load_demos(replay, {"dir": str(tmp_path / "nolabels"), "fraction": 0.1}, say=said.append)
    assert outcome["loaded"] == 2 and outcome["missing_labels"] == sorted(FakeTwoDeviceEnv.label_names)
    assert any("WARNING" in line for line in said)
    other = EpisodeReplay(capacity=1000, observation_dim=5, action_dim=4, horizon=3, batch_size=8)
    with pytest.raises(ValueError, match="re-record"):
        load_demos(other, {"dir": str(tmp_path / "nolabels"), "fraction": 0.1}, say=said.append)


def test_no_demo_directory_is_skipped_and_said(tmp_path):
    config = two_device_config()
    config["training"]["demos"] = {"dir": str(tmp_path / "nowhere"), "fraction": 0.08}
    seed_everything(0)
    trainer = start_run(config, FakeTwoDeviceEnv(), tmp_path / "run", quiet=True)
    assert trainer.replay.num_demo_episodes == 0 and trainer.replay.demo_fraction == 0.08
    assert "training without demonstrations" in (tmp_path / "run" / "console.log").read_text()
    events = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl") if e["event"] == "demos"]
    assert events[0]["loaded"] == 0 and events[0]["reason"] == "no such directory"


# -------------------------------------------------------------- robustness --
def test_step_timeout_from_config_replaces_a_hung_worker_and_logs_it(tmp_path):
    config = two_device_config()
    config["training"]["step_timeout_s"] = 3
    config["evaluation"]["interval"] = 1000
    seed_everything(0)
    pool = pool_of({"hang_at_calls": {7}, "hang_once_file": str(tmp_path / "hung")}, num_envs=2,
                   config=config)
    assert pool.timeout == 3.0
    summary = train_and_close(start_run(config, pool, tmp_path / "run", quiet=True))
    assert summary["total_env_steps"] == 120 and summary["env_restarts"] >= 1
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    restarts = [e for e in events if e["event"] == "worker_restarted"]
    assert len(restarts) == summary["env_restarts"]
    assert all("timed out" in e["reason"] for e in restarts)
    dropped = [e for e in events if e["event"] == "dropped_step"]
    assert len(dropped) == summary["env_errors"] >= 1
    assert step_timeout({"training": {}}) == 120.0


def test_nan_steps_end_the_episode_as_truncated_and_are_counted(tmp_path):
    config = two_device_config()
    config["evaluation"]["interval"] = 1000
    seed_everything(0)
    trainer = start_run(config, FakeTwoDeviceEnv(sim_error_at_calls={5, 33}), tmp_path / "run",
                        quiet=True)
    summary = trainer.train()
    assert summary["env_errors"] == 2
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    errors = [e for e in episodes if e["end"] == "error"]
    assert len(errors) == 2 and all(e["length"] == 4 for e in errors[:1])
    first = trainer.replay.episodes[0]                     # stored, no terminal flag: value bootstrapped
    assert len(first["actions"]) == 4 and first["terminated"].sum() == 0
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert metrics[0]["error_rate"] > 0
    assert sum(metrics[0]["error_phase_hist"].values()) >= 1          # which phase crashed
    dropped = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl") if e["event"] == "dropped_step"]
    assert len(dropped) == 2 and all("phase" in e for e in dropped)


def test_wall_breach_steps_are_dropped_and_the_episode_ends_as_breach(tmp_path):
    """A step flagged ``wall_breached`` (a centerline outside the vessel mesh) is not stored and
    closes the episode as 'breach' (no terminal flag: bootstrapped, no -1), counted apart from NaN
    errors; the same through worker processes (the key travels with the pool's info)."""
    config = two_device_config()
    config["evaluation"]["interval"] = 1000
    seed_everything(0)
    # the 5th call breaches (episode 1, its 5th step), the 7th is a NaN (episode 2, its 2nd step)
    trainer = start_run(config, FakeTwoDeviceEnv(breach_at_calls={5}, sim_error_at_calls={7}),
                        tmp_path / "run", quiet=True)
    summary = trainer.train()
    assert summary["wall_breaches"] == 1 and summary["env_errors"] == 1
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert (episodes[0]["end"], episodes[0]["length"]) == ("breach", 4)
    assert (episodes[1]["end"], episodes[1]["length"]) == ("error", 1)
    first = trainer.replay.episodes[0]                      # stored up to the step before, no terminal flag
    assert len(first["actions"]) == 4 and first["terminated"].sum() == 0
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert metrics[0]["breach_rate"] > 0 and metrics[0]["error_rate"] > 0
    assert metrics[-1]["wall_breaches"] == 1 and metrics[-1]["env_errors"] == 1
    dropped = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl") if e["event"] == "dropped_step"]
    assert [e["reason"] for e in dropped] == ["wall_breach", "simulation_error"]
    payload = trainer.checkpoint_payload()
    assert payload["wall_breaches"] == 1 and payload["env_errors"] == 1
    # through worker processes: each of the two envs breaches at its own 5th step
    seed_everything(0)
    pool = pool_of({"breach_at_calls": {5}}, num_envs=2, config=config)
    summary = train_and_close(start_run(config, pool, tmp_path / "pool", quiet=True))
    assert summary["wall_breaches"] == 2 and summary["env_errors"] == 0
    assert [e["end"] for e in read_jsonl(tmp_path / "pool" / "episodes.jsonl")].count("breach") == 2


# ----------------------------------------------------------------- metrics --
def test_metrics_files_carry_the_task_numbers(tmp_path):
    seed_everything(0)
    train_and_close(start_run(two_device_config(), pool_of(num_envs=2), tmp_path / "run", quiet=True))
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    keys = {"max_phase", "cutoff", "events", "shield_delta_mean", "shield_active_frac", "d_p10",
            "d_p50", "d_p90", "catheter_moving_frac", "success"}
    assert all(keys <= set(e) for e in episodes)
    assert set(episodes[0]["events"]) == {"naked_advance", "wrong_vessel"}
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [30, 60, 90, 120]
    last = metrics[-1]
    assert set(last["reach"]) == {"P1", "P2", "P3", "P4", "success"}
    for key in ("success_rate", "error_rate", "shield_delta_mean", "catheter_moving_frac", "d_p50",
                "events_per_episode", "cutoff_weights", "steps_per_s", "env_restarts",
                "max_phase_hist", "cutoff_share"):
        assert key in last, key
    console = (tmp_path / "run" / "console.log").read_text()
    assert "reach from insertion" in console and "cutoff weights" in console


def test_episode_stats_from_info():
    stats = EpisodeStats({"phase": 0, "max_phase": 0}, cutoff=3)
    infos = [{"phase": 1, "events": {"wrong_vessel": 1}, "shield_delta": [0.3, 0.4], "d": -10.0,
              "executed_action": [0.1, 0.0, 0.5, 0.0]},
             {"phase": 2, "events": {"wrong_vessel": 0}, "shield_delta": 0.0, "d": 20.0,
              "executed_action": [0.1, 0.0, 0.01, 0.0]},
             {"phase": 1, "events": {"wrong_vessel": 1, "exit": 1}, "d": 30.0,
              "executed_action": [30.0, 0.0, 30.0, 0.0]}]      # raw units: ignored, policy action used
    for info in infos:
        stats.add(np.array([0.0, 0.0, 0.0, 0.0]), info)
    f = stats.fields()
    assert f["max_phase"] == 2 and f["cutoff"] == 3 and stats.phases == [0, 1, 2, 1]
    assert f["last_phase"] == 1 and stats.where() == {"phase": 1}
    assert f["events"] == {"exit": 1, "wrong_vessel": 2}
    assert f["shield_delta_mean"] == pytest.approx(0.25) and f["shield_active_frac"] == 0.5
    assert f["d_p50"] == pytest.approx(20.0)
    assert f["catheter_moving_frac"] == pytest.approx(1 / 3)
    assert EpisodeStats({}).fields() == {}          # an env without task info adds nothing


def test_reach_rates_respect_cutoffs():
    records = [{"max_phase": 2, "cutoff": 2, "success": False},
               {"max_phase": 3, "cutoff": 5, "success": False},
               {"max_phase": 4, "cutoff": 5, "success": True},
               {"max_phase": 1, "cutoff": 4, "success": False}]
    reach = reach_rates(records)
    assert reach == {"P1": 1.0, "P2": 0.75, "P3": 2 / 3, "P4": 1 / 3, "success": 0.5}
    assert "reach" not in window_summary([{"success": False, "return": 0.0, "length": 3}])


# -------------------------------------------------------------- evaluation --
def load_evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_script",
                                                  WORKSPACE_ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_agent(obs_dim=6, act_dim=4):
    torch.manual_seed(0)
    return build_agent(tiny_config(), observation_dim=obs_dim, action_dim=act_dim,
                       episode_length=40, device="cpu")


def test_wilson_interval_known_values():
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def test_evaluation_report_from_episodes():
    agent = tiny_agent()
    env = FakeTwoDeviceEnv()
    records = [run_episode(env, agent, seed, cutoff=5) for seed in range(6)]
    assert env.cutoff_calls == [5] * 6
    out = report(records, step_seconds=1 / 7.5)
    assert out["episodes"] == 6 and len(out["success_ci95"]) == 2
    assert sum(out["max_phase_hist"].values()) == 6 and set(out["max_phase_hist"]) == set("012345")
    assert set(out["events_per_episode"]) == {"naked_advance", "wrong_vessel"}
    assert set(out["peak_force"]) == {"catheter_tip_force", "wire_tip_force"}
    assert out["time"]["seconds_mean"] == pytest.approx(out["time"]["steps_mean"] / 7.5)
    assert sum(out["end_reasons"].values()) == 6
    assert set(out["end_reasons"]) <= {"success", "failure:outside_wire", "simulation_error", "wall_breach",
                                       "other"}
    assert out["wall_breaches"] == 0
    for key in ("shield_delta_mean", "shield_active_frac", "catheter_moving_frac"):
        assert key in out


def test_evaluation_ends_a_breached_episode_and_names_it():
    agent = tiny_agent()
    record = run_episode(FakeTwoDeviceEnv(breach_at_calls={3}), agent, 0, cutoff=5)
    assert record["wall_breach"] and not record["success"] and record["length"] == 3
    out = report([record])
    assert out["wall_breaches"] == 1 and out["end_reasons"] == {"wall_breach": 1}


def test_freeze_catheter_sends_zero_catheter_actions():
    script = load_evaluate_script()
    env = FakeTwoDeviceEnv()
    run_episode(env, tiny_agent(), 3, action_hook=script.freeze_catheter)
    executed = np.stack(env.executed)
    assert np.all(executed[:, 2:] == 0) and np.any(executed[:, :2] != 0)


def test_target_split_override_only_when_the_target_has_one(tmp_path):
    script = load_evaluate_script()
    assert script.target_split_overrides(str(WORKSPACE_ROOT / "configs" / "g13.yaml"), "eval") == []
    child = tmp_path / "split.yaml"
    child.write_text(f"extends: {WORKSPACE_ROOT / 'configs' / 'g13.yaml'}\n"
                     "env: {scene: {target: {split: train}}}\n")
    assert script.target_split_overrides(str(child), "eval") == ["env.scene.target.split=eval"]
    assert script.target_split_overrides(str(child), "keep") == []


class _Branch:
    def __init__(self, name, coordinates):
        self.name, self.coordinates = name, np.asarray(coordinates, dtype=np.float32)
        self.radii = np.full(len(coordinates), 2.0, dtype=np.float32)


class FakeSceneEnv(FakeTwoDeviceEnv):
    """``FakeTwoDeviceEnv`` with the scene attributes the workbench reads."""

    def __init__(self, mesh_path, **kwargs):
        super().__init__(**kwargs)
        from types import SimpleNamespace as NS
        self.observation_names = [f"obs[{i}]" for i in range(6)]
        self.observation_layout = [("obs", 6)]
        tree = NS(mesh_path=str(mesh_path), branches=[_Branch("aorta", [[0, 0, 0], [0, 0, 50]])],
                  insertion=NS(position=np.zeros(3), direction=np.array([0.0, 0.0, 1.0])))
        fluoro = NS(image_frequency=7.5, image_rot_zx=(0.0, 0.0), image_center=(0.0, 0.0, 0.0))
        self.intervention = NS(vessel_tree=tree, fluoroscopy=fluoro, instruments=[],
                               target=NS(threshold=5.0, coordinates3d=np.array([0.0, 0.0, 40.0])),
                               simulation=NS(dof_positions=np.zeros((5, 3))),
                               instrument_lengths_inserted=[0.0, 0.0], instrument_rotations=[0.0, 0.0])
        self._fluoro = fluoro

    @property
    def raw_action_limits(self):
        return np.full(4, -1.0, np.float32), np.full(4, 1.0, np.float32)

    def _move(self):
        tip = np.array([0.0, 0.0, self.progress / 2])
        self.intervention.simulation.dof_positions = np.stack([tip - [0, 0, k] for k in range(5)])
        self._fluoro.instrument_trackings3d = [tip[None], (tip - [0, 0, 3])[None]]

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self._move()
        return out

    def step(self, action):
        out = super().step(action)
        self._move()
        return out


def test_saved_episodes_load_in_the_workbench(tmp_path):
    pytest.importorskip("trimesh")          # the workbench lives in the guidewire_rl env only
    script = load_evaluate_script()
    mesh = tmp_path / "vessel.obj"
    mesh.write_text("v 0 0 0\nv 10 0 0\nv 0 10 0\nv 0 0 10\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")
    env = FakeSceneEnv(mesh)
    recorder = script.EpisodeRecorder(tmp_path / "episodes", "test")
    record = run_episode(env, tiny_agent(), 11, cutoff=5, hooks=recorder)
    path = Path(recorder.save(0, record))
    assert path.name.startswith("episode_000_seed11_")
    from parts.frame import EpisodeLog            # the workbench's loader (ReplaySource uses it)
    log = EpisodeLog.load(path)
    assert len(log) == record["length"] + 1
    last = log.frames[-1]
    assert last.extras["phase"] == record["max_phase"]
    assert len(last.extras["instrument_tips_mm"]) == 2 and "d" in last.extras
    assert log.frames[1].labels.keys() == set(FakeTwoDeviceEnv.label_names)


# ------------------------------------------------------------ residual box --
def test_residual_box_keeps_every_planned_action_inside(monkeypatch):
    agent = tiny_agent()
    low = np.array([-0.2, -0.1, 0.3, -1.0], dtype=np.float32)
    high = np.array([0.4, 0.1, 0.9, -0.5], dtype=np.float32)
    seen = []
    original = agent._estimate_value

    def spy(latent, actions):
        seen.append(actions.clone())
        return original(latent, actions)

    monkeypatch.setattr(agent, "_estimate_value", spy)
    obs = np.linspace(-0.5, 0.5, 6).astype(np.float32)
    for eval_mode in (False, True):
        action = agent.act(obs, first_step=True, eval_mode=eval_mode, bounds=(low, high))
        assert np.all(action >= low - 1e-6) and np.all(action <= high + 1e-6)
    every = torch.cat([a.reshape(-1, 4) for a in seen])
    assert torch.all(every >= torch.as_tensor(low) - 1e-6) and torch.all(every <= torch.as_tensor(high) + 1e-6)
    # batched: env 0 boxed, env 1 free
    seen.clear()
    actions, _ = agent.act_batch(np.stack([obs, -obs]), first_step=[True, True],
                                 previous_means=torch.zeros(2, agent.horizon, 4), eval_mode=False,
                                 bounds=[(low, high), None])
    assert np.all(actions[0] >= low - 1e-6) and np.all(actions[0] <= high + 1e-6)
    per_env = seen[0].view(agent.horizon, 2, -1, 4)
    assert torch.all(per_env[:, 0] >= torch.as_tensor(low) - 1e-6)
    assert (per_env[:, 1].abs() > 0.95).any()           # the free env still samples the full range


def test_residual_box_off_plans_exactly_as_before():
    agent = tiny_agent()
    obs = np.linspace(-0.5, 0.5, 6).astype(np.float32)
    torch.manual_seed(5)
    plain = agent.act(obs, first_step=True, eval_mode=False)
    torch.manual_seed(5)
    unboxed = agent.act(obs, first_step=True, eval_mode=False, bounds=None)
    np.testing.assert_array_equal(plain, unboxed)
    torch.manual_seed(6)
    a, m = agent.act_batch(np.stack([obs, obs]), first_step=[True, False],
                           previous_means=torch.zeros(2, agent.horizon, 4))
    torch.manual_seed(6)
    b, n = agent.act_batch(np.stack([obs, obs]), first_step=[True, False],
                           previous_means=torch.zeros(2, agent.horizon, 4), bounds=[None, None])
    np.testing.assert_array_equal(a, b)
    torch.testing.assert_close(m, n, rtol=0, atol=0)


def test_residual_box_needs_an_available_suggestion():
    schedule = ResidualSchedule(delta=0.5)
    assert schedule.box_for({"phase": 1, "nominal_action": [0.0] * 4, "nominal_available": False}) is None
    assert schedule.box_for({"phase": 1}) is None
    assert schedule.box_for({"phase": 2, "nominal_action": [0.0] * 4, "nominal_available": True}) is None
    low, high = schedule.box_for({"phase": 1, "nominal_action": [0.0] * 4, "nominal_available": True})
    np.testing.assert_allclose(low, -0.5)
    np.testing.assert_allclose(high, 0.5)


def test_residual_schedule_grows_when_phase_one_is_passed():
    schedule = ResidualSchedule(delta=0.5, grow_every_episodes=10, grow_if_phase1_rate=0.7,
                                grow_step=0.25, max=1.0)
    assert schedule.box(0, [0.0] * 4) is None and schedule.box(1, None) is None
    low, high = schedule.box(1, [0.9, -0.9, 0.0, 0.2])
    np.testing.assert_allclose(low, [0.4, -1.0, -0.5, -0.3], atol=1e-6)
    np.testing.assert_allclose(high, [1.0, -0.4, 0.5, 0.7], atol=1e-6)
    for k in range(10):                     # 8 of 9 that entered phase 1 got through
        assert (schedule.record(0 if k == 0 else (1 if k == 1 else 3), False) is None) == (k < 9)
    assert schedule.delta == 0.75
    for _ in range(10):                     # 0 of 10 got through: stays
        schedule.record(1, False)
    assert schedule.delta == 0.75
    for _ in range(20):
        schedule.record(4, True)
    assert schedule.delta == 1.0            # capped at max


def test_residual_box_in_training_follows_phase_one(tmp_path):
    config = two_device_config()
    config["planning"]["residual"] = {"delta": 0.25, "grow_every_episodes": 3,
                                      "grow_if_phase1_rate": 0.5, "grow_step": 0.25, "max": 1.0}
    seed_everything(0)
    trainer = start_run(config, FakeTwoDeviceEnv(), tmp_path / "run", quiet=True)
    trainer.train()
    delta = load_checkpoint(tmp_path / "run" / "checkpoints" / "step_120.pt")["residual"]["delta"]
    assert 0.25 <= delta <= 1.0
    nominal = np.array([0.2, -0.3, 0.1, 0.5])
    seed_steps, step, boxed, free = config["training"]["seed_steps"], 0, 0, 0
    for e in trainer.replay.episodes:          # action t was planned at observation t
        for t, action in enumerate(e["actions"]):
            if step >= seed_steps:
                if e["phase"][t] == 1:          # the box only grows: inside the last one
                    boxed += 1
                    assert np.all(np.abs(action - nominal) <= delta + 1e-5)
                else:
                    free += int(np.any(np.abs(action - nominal) > delta + 1e-5))
            step += 1
    assert boxed > 0 and free > 0
    events = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl") if e["event"] == "residual_check"]
    assert events and all("phase1_pass_rate" in e for e in events)


# --------------------------------------------------- reproducible when on --
def test_with_every_new_part_on_lockstep_training_still_reproduces(tmp_path):
    write_demos(tmp_path / "demos", 2)
    for name in ("a", "b"):
        config = two_device_config()
        config["replay"] = {"stratify_by": "phase", "stratify_min_samples": 20}
        config["training"]["demos"] = {"dir": str(tmp_path / "demos"), "fraction": 0.25}
        config["planning"]["residual"] = {"delta": 0.5, "grow_every_episodes": 4}
        seed_everything(0)
        train_and_close(start_run(config, pool_of(num_envs=3), tmp_path / name, quiet=True))
    for stream in ("episodes", "eval"):
        assert (tmp_path / "a" / f"{stream}.jsonl").read_text() == \
            (tmp_path / "b" / f"{stream}.jsonl").read_text()


# --------------------------------------------- off = the code as it was --
GOLDEN = {   # the pre-change code (2026-10-31), torch on one thread; tests/fakes.FakeEnv
    "single": {"replay": "cccbf1ce40ccb6bc", "model": "4923168c1a9c9f74", "updates": 50,
               "episodes_jsonl": "843d7cb05abe2a3b", "eval_jsonl": "ca89dc6944faf38c",
               "picks": "59fa5f04b2260c82"},
    "parallel": {"replay": "b43cfbbdb98fbd7a", "model": "68edfcdbfdb045d1", "updates": 50,
                 "episodes_jsonl": "f5795526c6e24c35", "eval_jsonl": "03f8fa449e667366",
                 "picks": "fd0bc730c229929a"},
}


def _sha(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _fingerprint(trainer, run_dir):
    r = trainer.replay
    replay = _sha([e[k] for e in r.episodes for k in ("observations", "actions", "rewards", "terminated")]
                  + [e["labels"][n] for e in r.episodes for n in r.label_names])
    model = _sha([v.detach().cpu().numpy() for v in trainer.agent.model.state_dict().values()])
    episodes, starts = r.sample_indices(64)
    digest = lambda name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()[:16]  # noqa: E731
    return {"replay": replay, "model": model, "updates": trainer.agent.update_count,
            "episodes_jsonl": digest("episodes.jsonl"), "eval_jsonl": digest("eval.jsonl"),
            "picks": _sha([episodes, starts])}


def test_with_every_new_part_off_training_is_bit_for_bit_the_old_code(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)               # float sums depend on the thread count
    try:
        seed_everything(0)
        trainer = start_run(tiny_config(), FakeEnv(), tmp_path / "single", quiet=True)
        trainer.train()
        assert _fingerprint(trainer, tmp_path / "single") == GOLDEN["single"]
        seed_everything(0)
        config = tiny_config()
        config["training"]["async_envs"] = False
        pool = EnvPool(FakeEnv, kwargs={"max_episode_steps": 5}, num_envs=3, seed=0)
        try:
            trainer = start_run(config, pool, tmp_path / "parallel", quiet=True)
            trainer.train()
        finally:
            pool.close()
        assert _fingerprint(trainer, tmp_path / "parallel") == GOLDEN["parallel"]
    finally:
        torch.set_num_threads(threads)
