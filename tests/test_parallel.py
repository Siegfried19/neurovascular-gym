"""Parallel training: env pool, batched planning, the N-env training loop.

The pool tests start real worker processes (FakeEnv, no SOFA)."""

import copy
import json
import os

import pytest

import numpy as np
import torch

from guidewire_rl.model import build_agent
from guidewire_rl.train.checkpoint import load_checkpoint
from guidewire_rl.env import build_env
from guidewire_rl.train.env_pool import WORKER_OOM_SCORE_ADJ, EnvPool, split_cpus
from guidewire_rl.train.evaluate import evaluate, evaluate_pool
from guidewire_rl.train.parallel_trainer import AsyncTrainer, ParallelTrainer
from guidewire_rl.train.setup import seed_everything, start_run

from conftest import tiny_config
from fakes import FakeEnv

STEP = np.full(2, 0.5, dtype=np.float32)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------- pool --
def test_pool_steps_and_reports_errors_without_losing_the_env():
    pool = EnvPool(FakeEnv, kwargs={"error_at_calls": {2}}, num_envs=2, seed=0)
    try:
        assert len(pool) == 2 and pool.label_names == ["distal_force"]
        assert pool.max_episode_steps == 15 and pool.action_space.shape == (2,)
        pool.send({0: ("reset", 5), 1: ("reset", 6)})
        reset = pool.receive()
        np.testing.assert_array_equal(reset[0].value[0], FakeEnv().reset(seed=5)[0])
        pool.send({0: ("step", STEP), 1: ("step", STEP)})
        step = pool.receive()
        assert step[0].ok and set(step[0].value[4]) == {"is_success", "simulation_error", "labels"}
        pool.send({0: ("step", STEP)})                 # env 0's second step raises
        failed = pool.receive()
        assert not failed[0].ok and "fake simulator failure" in failed[0].error
        pool.send({0: ("step", STEP)})                 # same env, still there
        assert pool.receive()[0].ok and pool.restarts == 0 and pool.recycles == 0
    finally:
        pool.close()


def test_pool_recycles_a_grown_worker_only_at_a_reset():
    """Above max_rss_mb a worker is swapped for a fresh one before its reset, never
    during an episode; the fresh one gives the same episode for the same seed."""
    pool = EnvPool(FakeEnv, num_envs=2, seed=0, max_rss_mb=1.0)    # every worker is above 1 MB
    try:
        pids = [proc.pid for proc in pool._procs]
        assert all(rss > 1.0 for rss in pool.worker_rss_mb())
        pool.send({0: ("reset", 5), 1: ("reset", 6)})
        reset = pool.receive()
        np.testing.assert_array_equal(reset[0].value[0], FakeEnv().reset(seed=5)[0])
        assert pool.recycles == 2 and pool.restarts == 0
        assert all(proc.pid not in pids for proc in pool._procs)
        events = pool.drain_events()
        assert [e["event"] for e in events] == ["worker_recycled"] * 2
        assert events[0]["rss_mb"] > 1.0 and events[0]["max_rss_mb"] == 1.0
        pids = [proc.pid for proc in pool._procs]
        for _ in range(3):                             # steps: the same workers
            pool.send({0: ("step", STEP), 1: ("step", STEP)})
            assert all(r.ok for r in pool.receive().values())
        assert [proc.pid for proc in pool._procs] == pids and pool.recycles == 2
    finally:
        pool.close()
    pool = EnvPool(FakeEnv, num_envs=1, max_rss_mb=1e6)              # below the limit: kept
    try:
        pid = pool._procs[0].pid
        pool.send({0: ("reset", 5)})
        assert pool.receive()[0].ok and pool.recycles == 0 and pool._procs[0].pid == pid
    finally:
        pool.close()


def test_workers_go_first_when_the_machine_runs_out_of_memory():
    pool = EnvPool(FakeEnv, num_envs=1)
    try:
        with open(f"/proc/{pool._procs[0].pid}/oom_score_adj") as adj:
            assert int(adj.read()) == WORKER_OOM_SCORE_ADJ
    finally:
        pool.close()


def test_pool_replaces_a_dead_or_hung_worker():
    for fake, timeout, word in (({"crash_at_calls": {2}}, 600.0, "died"),
                                ({"hang_at_calls": {2}}, 3.0, "timed out")):
        pool = EnvPool(FakeEnv, kwargs=fake, num_envs=1, timeout=timeout)
        try:
            pool.send({0: ("reset", 1)})
            pool.receive()
            pool.send({0: ("step", STEP)})
            assert pool.receive()[0].ok
            pool.send({0: ("step", STEP)})
            result = pool.receive()[0]
            assert not result.ok and word in result.error and pool.restarts == 1
            pool.send({0: ("reset", 1)})             # the new worker works
            assert pool.receive()[0].ok
        finally:
            pool.close()


def test_receive_ready_returns_the_fast_env_while_the_slow_one_steps():
    pool = EnvPool(FakeEnv, kwargs={"delay_at_calls": {2: 1.5}}, num_envs=2)
    try:
        pool.send({0: ("reset", 1), 1: ("reset", 1)})
        pool.receive()
        pool.send({0: ("step", STEP)})                 # env 0's first step: fast
        pool.receive()
        pool.send({0: ("step", STEP), 1: ("step", STEP)})   # env 0's second step: 1.5 s
        ready = pool.receive_ready(timeout=0.5)
        assert list(ready) == [1] and ready[1].ok and pool.pending == [0]
        with pytest.raises(RuntimeError):              # env 0 still owes an answer
            pool.send({0: ("step", STEP)})
        pool.send({1: ("step", STEP)})                 # env 1 can go on meanwhile
        rest = pool.receive()
        assert set(rest) == {0, 1} and all(r.ok for r in rest.values()) and pool.pending == []
    finally:
        pool.close()


def test_cpu_split_keeps_workers_off_the_main_core():
    split = split_cpus()
    if split is None:                       # no readable CPU layout: nothing gets pinned
        return
    main, workers = split
    assert main and workers and not main & workers
    assert main | workers == os.sched_getaffinity(0)
    pool = EnvPool(FakeEnv, num_envs=1, cpus=workers)
    try:
        assert os.sched_getaffinity(pool._procs[0].pid) == workers
    finally:
        pool.close()


# ------------------------------------------------------ batched planning --
def make_agent(**model):
    config = tiny_config()
    config["model"].update(model)
    torch.manual_seed(0)
    agent = build_agent(config, observation_dim=4, action_dim=2, episode_length=15, device="cpu")
    return config, agent


def test_act_batch_with_one_env_matches_act():
    _, agent = make_agent()
    observations = np.random.default_rng(0).uniform(-1, 1, (3, 4)).astype(np.float32)
    means = torch.zeros(1, agent.horizon, 2)
    for eval_mode in (False, True):
        for k, first in enumerate((True, False, False)):
            torch.manual_seed(10 + k)
            single = agent.act(observations[k], first_step=first, eval_mode=eval_mode)
            torch.manual_seed(10 + k)
            batch, means = agent.act_batch(observations[k:k + 1], first_step=[first],
                                           previous_means=means, eval_mode=eval_mode)
            np.testing.assert_allclose(batch[0], single, atol=1e-6)
            torch.testing.assert_close(means[0], agent.previous_mean)


def test_act_batch_keeps_envs_apart(monkeypatch):
    # Policy samples collapse onto the policy mean, so the only randomness left
    # is the planner's own noise, which the test feeds in.
    config, agent = make_agent(log_std_min=-30.0, log_std_max=-30.0)
    p = config["planning"]
    shape = (p["iterations"], agent.horizon, 3, p["num_samples"] - p["num_pi_trajs"], 2)
    noise = torch.randn(shape)
    rng = np.random.default_rng(1)
    observations = rng.uniform(-1, 1, (3, 4)).astype(np.float32)
    warm = torch.as_tensor(rng.uniform(-1, 1, (3, agent.horizon, 2)), dtype=torch.float32)
    first = [True, False, False]

    def feed(chunks):
        chunks = iter(chunks)
        monkeypatch.setattr(agent, "_planning_noise", lambda _shape: next(chunks))

    feed(noise)
    together, together_means = agent.act_batch(observations, first_step=first,
                                               previous_means=warm, eval_mode=True)
    for k in range(3):
        feed(noise[:, :, k:k + 1])
        alone, alone_means = agent.act_batch(observations[k:k + 1], first_step=first[k:k + 1],
                                             previous_means=warm[k:k + 1], eval_mode=True)
        np.testing.assert_allclose(together[k], alone[0], atol=1e-5)
        torch.testing.assert_close(together_means[k], alone_means[0], atol=1e-5, rtol=0)


# ------------------------------------------------------------ evaluation --
def test_evaluate_pool_equals_evaluate_bit_for_bit():
    """Same episodes, same planner samples, same numbers, whatever the pool size."""
    _, agent = make_agent()
    seeds = [1000, 1001, 1002, 1003, 1004]
    sequential = evaluate(FakeEnv(), agent, seeds)
    assert len({e["return"] for e in sequential["episodes"]}) > 1       # not a trivial match
    for n in (1, 2, 3):
        pool = EnvPool(FakeEnv, num_envs=n)
        try:
            assert evaluate_pool(pool, agent, seeds) == sequential
        finally:
            pool.close()


@pytest.mark.sofa
@pytest.mark.upstream
def test_evaluate_pool_equals_evaluate_on_the_real_simulator(env_config):
    """Two SOFA scenes in worker processes give the same evaluation numbers as
    one scene in this process (two 25-step episodes, planner mean actions).

    Upstream only: with Wilson's eve (SOFA 23.06) the scene is not rebuilt
    between episodes, and the same seed gives a different initial wire rotation
    as a process's first and as its second episode (measured 2026-10-28), so
    there a pool's fresh first episodes cannot match a sequential run."""
    config = copy.deepcopy(env_config)
    config["env"]["truncation"] = [{"type": "MaxSteps", "max_steps": 25}]
    env = build_env(config["env"])
    try:
        agent = build_agent(tiny_config(), observation_dim=env.observation_space.shape[0],
                            action_dim=2, episode_length=25, device="cpu")
        sequential = evaluate(env, agent, [1000, 1001])
    finally:
        env.close()
    assert sequential["summary"]["length_mean"] > 0
    pool = EnvPool(build_env, (config["env"],), num_envs=2)
    try:
        assert evaluate_pool(pool, agent, [1000, 1001]) == sequential
    finally:
        pool.close()


# ------------------------------------------------------- training loop --
def parallel_trainer(tmp_path, name, config=None, fake=None, num_envs=3, async_envs=False,
                     max_rss_mb=None, **kwargs):
    seed_everything(0)
    config = config or tiny_config()
    config["training"]["async_envs"] = async_envs
    pool = EnvPool(FakeEnv, kwargs=fake or {"max_episode_steps": 5}, num_envs=num_envs, seed=0,
                   max_rss_mb=max_rss_mb)
    try:
        return start_run(config, pool, tmp_path / name, quiet=True, **kwargs)
    except BaseException:
        pool.close()
        raise


def train_and_close(trainer):
    try:
        return trainer.train()
    finally:
        trainer.env.close()


def test_parallel_schedule_logs_and_checkpoints(tmp_path):
    # 60 steps: 20 random, burst of 10 updates, then 1 per env step
    trainer = parallel_trainer(tmp_path, "run")
    assert isinstance(trainer, ParallelTrainer)
    summary = train_and_close(trainer)
    assert summary["total_env_steps"] == 60
    assert summary["updates"] == 10 + (60 - 20)
    checkpoints = tmp_path / "run" / "checkpoints"
    assert load_checkpoint(checkpoints / "step_30.pt")["agent"]["update_count"] == 10 + (30 - 20)
    assert (checkpoints / "step_60.pt").is_file()
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert sum(e["length"] for e in episodes) == trainer.replay.size
    evals = read_jsonl(tmp_path / "run" / "eval.jsonl")
    assert [e["step"] for e in evals] == [30, 60] and evals[0]["episodes"] == 2
    assert read_jsonl(tmp_path / "run" / "updates.jsonl")


def test_parallel_resume_continues_counters_and_replay(tmp_path):
    train_and_close(parallel_trainer(tmp_path, "a"))
    resumed = parallel_trainer(tmp_path, "a", resume=tmp_path / "a" / "checkpoints" / "step_30.pt")
    assert resumed.step == 30 and resumed.replay.size > 0 and resumed.agent.update_count == 20
    summary = train_and_close(resumed)
    assert summary["total_env_steps"] == 60 and summary["updates"] == 20 + 30


def test_parallel_errors_and_dead_workers_do_not_stop_training(tmp_path):
    config = tiny_config()
    config["evaluation"]["interval"] = 1000
    fake = {"max_episode_steps": 5, "error_at_calls": {3}, "crash_at_calls": {9}}
    summary = train_and_close(parallel_trainer(tmp_path, "run", config, fake, num_envs=2))
    assert summary["total_env_steps"] == 60 and summary["env_restarts"] >= 2
    dropped = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl")
               if e["event"] == "dropped_step"]
    assert len(dropped) == summary["env_errors"]
    reasons = " ".join(e["reason"] for e in dropped)
    assert "fake simulator failure" in reasons and "died" in reasons


def test_parallel_training_is_reproducible(tmp_path):
    for name in ("a", "b"):
        train_and_close(parallel_trainer(tmp_path, name))
    for stream in ("episodes", "eval"):        # events carry the run folder's path
        assert ((tmp_path / "a" / f"{stream}.jsonl").read_text()
                == (tmp_path / "b" / f"{stream}.jsonl").read_text())


def test_recycling_workers_leaves_lockstep_training_unchanged(tmp_path):
    """A fresh worker at every reset (limit 1 MB): same episodes, evaluations, updates."""
    plain = train_and_close(parallel_trainer(tmp_path, "a"))
    recycled = train_and_close(parallel_trainer(tmp_path, "b", max_rss_mb=1.0))
    assert plain["env_recycles"] == 0 and recycled["env_recycles"] >= 3
    for stream in ("episodes", "eval"):
        assert ((tmp_path / "a" / f"{stream}.jsonl").read_text()
                == (tmp_path / "b" / f"{stream}.jsonl").read_text())
    assert {k: v for k, v in plain.items() if k not in ("env_recycles", "final_checkpoint")} == \
        {k: v for k, v in recycled.items() if k not in ("env_recycles", "final_checkpoint")}
    recycles = [e for e in read_jsonl(tmp_path / "b" / "events.jsonl")
                if e["event"] == "worker_recycled"]
    assert len(recycles) == recycled["env_recycles"]


def test_evaluation_pool_is_closed_after_each_evaluation_and_memory_is_logged(tmp_path,
                                                                             monkeypatch):
    spawned = []
    original = EnvPool.spawn

    def spy(self, num_envs):
        spawned.append(original(self, num_envs))
        return spawned[-1]

    monkeypatch.setattr(EnvPool, "spawn", spy)
    config = tiny_config()
    config["logging"]["print_interval"] = 20
    trainer = parallel_trainer(tmp_path, "run", config)
    train_and_close(trainer)
    assert len(spawned) == 2                       # evaluations at steps 30 and 60
    assert all(proc is None for pool in spawned for proc in pool._procs)
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [20, 40, 60]
    memory = metrics[0]["memory_mb"]
    assert memory["main"] > 0 and memory["available"] > 0
    assert 0 < memory["workers_max"] <= memory["workers_total"]
    assert metrics[0]["env_recycles"] == 0


# ---------------------------------------------------- async training loop --
def test_async_schedule_logs_and_checkpoints(tmp_path):
    # the slow env (its 2nd, 5th, ... call sleeps) must not hold the others back
    fake = {"max_episode_steps": 5, "delay_at_calls": {c: 0.2 for c in range(2, 200, 3)}}
    trainer = parallel_trainer(tmp_path, "run", fake=fake, async_envs=True)
    assert isinstance(trainer, AsyncTrainer)
    summary = train_and_close(trainer)
    assert summary["total_env_steps"] == 60
    assert summary["updates"] == 10 + (60 - 20)
    checkpoints = tmp_path / "run" / "checkpoints"
    assert load_checkpoint(checkpoints / "step_30.pt")["agent"]["update_count"] == 10 + (30 - 20)
    assert (checkpoints / "step_60.pt").is_file()
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert sum(e["length"] for e in episodes) == trainer.replay.size
    evals = read_jsonl(tmp_path / "run" / "eval.jsonl")
    assert [e["step"] for e in evals] == [30, 60] and evals[0]["episodes"] == 2


def test_async_resume_from_a_lockstep_checkpoint(tmp_path):
    train_and_close(parallel_trainer(tmp_path, "a"))
    resumed = parallel_trainer(tmp_path, "a", resume=tmp_path / "a" / "checkpoints" / "step_30.pt",
                               num_envs=2, async_envs=True)
    assert resumed.step == 30 and resumed.agent.update_count == 20
    summary = train_and_close(resumed)
    assert summary["total_env_steps"] == 60 and summary["updates"] == 20 + 30


def test_async_errors_and_dead_workers_do_not_stop_training(tmp_path):
    config = tiny_config()
    config["evaluation"]["interval"] = 1000
    fake = {"max_episode_steps": 5, "error_at_calls": {3}, "crash_at_calls": {9}}
    summary = train_and_close(parallel_trainer(tmp_path, "run", config, fake, num_envs=2,
                                               async_envs=True))
    assert summary["total_env_steps"] == 60 and summary["env_restarts"] >= 2
    dropped = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl")
               if e["event"] == "dropped_step"]
    assert len(dropped) == summary["env_errors"]
    reasons = " ".join(e["reason"] for e in dropped)
    assert "fake simulator failure" in reasons and "died" in reasons


# ------------------------------------------- evaluation beside training --
def test_background_evaluation_gives_the_pausing_evaluations_numbers_for_the_agent_at_that_step(tmp_path):
    """evaluation.background: the agent is copied at the evaluation's step and its episodes run on their
    own pool while training goes on. The numbers are those of evaluating that step's checkpoint."""
    fake = {"max_episode_steps": 5, "delay_at_calls": {c: 0.05 for c in range(2, 400, 3)}}
    config = tiny_config()
    config["evaluation"]["background"] = True
    trainer = parallel_trainer(tmp_path, "run", config, fake=fake, async_envs=True)
    summary = train_and_close(trainer)
    assert summary["total_env_steps"] == 60
    run = tmp_path / "run"
    evals = read_jsonl(run / "eval.jsonl")
    assert [e["step"] for e in evals] == [30, 60], "every evaluation is finished before training returns"
    events = [e for e in read_jsonl(run / "events.jsonl") if e["event"] == "evaluation"]
    assert [e["evaluated_step"] for e in events] == [30, 60] and all(e["background"] for e in events)
    assert events[0]["step"] >= 30, "reported when it arrives"
    assert not list((run / "checkpoints").glob("exam_step_*.pt")), "no pending checkpoint left behind"

    # the same episodes, evaluated the pausing way from the checkpoint written at step 30
    checkpoint = load_checkpoint(run / "checkpoints" / "step_30.pt")
    agent = trainer.agent.acting_copy()
    agent.load_state_dict(checkpoint["agent"], load_optimizers=False)
    ev = config["evaluation"]
    seeds = [int(ev["seed"]) + i for i in range(int(ev["episodes"]))]
    pool = EnvPool(FakeEnv, kwargs={"max_episode_steps": 5}, num_envs=2)
    try:
        reference = evaluate_pool(pool, agent, seeds)["summary"]
    finally:
        pool.close()
    for key, value in reference.items():
        assert evals[0][key] == value, key


def test_background_evaluation_touches_neither_trainings_random_numbers_nor_its_agent(tmp_path):
    """The evaluation's episodes use their own RNG streams and their own copy of the agent."""
    config = tiny_config()
    config["evaluation"]["background"] = True
    trainer = parallel_trainer(tmp_path, "run", config, async_envs=True)
    try:
        trainer._exam, trainer._exams, trainer._exam_polled = None, [], 0.0
        trainer.timing["evaluation beside training"] = 0.0
        before = {k: v.clone() for k, v in trainer.agent.state_dict()["model"].items()}
        warm_start = trainer.agent.previous_mean.clone()
        torch.manual_seed(7)
        rng = torch.get_rng_state()
        trainer._queue_evaluation()
        trainer._settle_evaluations()
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(trainer.agent.previous_mean, warm_start)
        for key, value in trainer.agent.state_dict()["model"].items():
            assert torch.equal(value, before[key]), key
        assert [e["step"] for e in read_jsonl(tmp_path / "run" / "eval.jsonl")] == [0]
    finally:
        trainer.env.close()


def test_background_evaluation_keeps_the_evaluated_agent_as_best_and_drops_on_an_interrupt(tmp_path):
    config = tiny_config()
    config["evaluation"].update(background=True, keep_best=True)
    trainer = parallel_trainer(tmp_path, "run", config, async_envs=True)
    train_and_close(trainer)
    run = tmp_path / "run"
    best = [e for e in read_jsonl(run / "events.jsonl") if e["event"] == "best_checkpoint"]
    assert best and load_checkpoint(run / "checkpoints" / "best.pt")["total_env_steps"] == best[-1]["evaluated_step"]
    assert not list((run / "checkpoints").glob("exam_step_*.pt"))

    # an interrupt while an evaluation is outstanding: dropped, said so, nothing left behind
    config = tiny_config()
    config["evaluation"].update(background=True, keep_best=True)
    trainer = parallel_trainer(tmp_path, "stopped", config, async_envs=True)
    original = trainer._queue_evaluation

    def queue_and_interrupt():
        original()
        raise KeyboardInterrupt

    trainer._queue_evaluation = queue_and_interrupt
    with pytest.raises(KeyboardInterrupt):
        train_and_close(trainer)
    stopped = tmp_path / "stopped"
    dropped = [e for e in read_jsonl(stopped / "events.jsonl") if e["event"] == "evaluation_dropped"]
    assert dropped and dropped[0]["evaluated_steps"] == [30]
    assert not (stopped / "eval.jsonl").exists() or not read_jsonl(stopped / "eval.jsonl")
    assert not list((stopped / "checkpoints").glob("exam_step_*.pt"))


def test_background_evaluation_needs_the_async_trainer(tmp_path):
    config = tiny_config()
    config["evaluation"]["background"] = True
    with pytest.raises(ValueError, match="evaluation.background"):
        parallel_trainer(tmp_path, "run", config, async_envs=False)


def test_evaluation_time_limit_ends_the_unfinished_episodes_as_simulation_errors(tmp_path):
    """evaluation.max_minutes: episodes still running then are recorded as simulation errors, the
    evaluation reports as usual; in the pausing evaluation and in the one beside training."""
    slow = {"max_episode_steps": 5, "delay_at_calls": {c: 1.0 for c in range(3, 400)}}   # crawls from the 3rd call on
    for background in (False, True):
        config = tiny_config()
        config["evaluation"].update(background=background, max_minutes=1.5 / 60.0, interval=30)
        config["training"]["total_steps"] = 30
        trainer = parallel_trainer(tmp_path, f"run{background}", config, fake=slow, num_envs=2, async_envs=True)
        train_and_close(trainer)
        run = tmp_path / f"run{background}"
        evals = read_jsonl(run / "eval.jsonl")
        assert [e["step"] for e in evals] == [30] and evals[0]["episodes"] == 2
        cut = [e for e in read_jsonl(run / "events.jsonl") if e["event"] == "evaluation_cut"]
        assert cut and cut[0]["evaluated_step"] == 30 and len(cut[0]["seeds"]) >= 1
        assert evals[0]["end_reasons"].get("simulation_error", 0) == len(cut[0]["seeds"])


def test_at_most_one_background_evaluation_waits(tmp_path):
    config = tiny_config()
    config["evaluation"]["background"] = True
    trainer = parallel_trainer(tmp_path, "run", config, async_envs=True)
    try:
        trainer._exam, trainer._exams, trainer._exam_polled = None, [], 0.0
        trainer.timing["evaluation beside training"] = 0.0
        trainer._queue_evaluation()
        trainer._poll_evaluation()                        # the first one starts
        for step in (10, 20, 30):
            trainer.step = step
            trainer._queue_evaluation()                   # 10 waits, 20 replaces it, 30 replaces 20
        assert trainer._exam["step"] == 0 and [job["step"] for job in trainer._exams] == [30]
        trainer._settle_evaluations()
        assert [e["step"] for e in read_jsonl(tmp_path / "run" / "eval.jsonl")] == [0, 30]
        dropped = [e for e in read_jsonl(tmp_path / "run" / "events.jsonl") if e["event"] == "evaluation_dropped"]
        assert [e["evaluated_steps"] for e in dropped] == [[10], [20]]
    finally:
        trainer.env.close()


def test_evaluation_episodes_end_when_stalled_and_training_episodes_do_not(tmp_path):
    """evaluation.end_when_stalled: only the evaluation episodes are asked to end once the devices stand
    still (the fake env "stalls" at step 2 when asked); they are reported with the end reason "stalled",
    not as simulation errors; training episodes on the same simulators run on. Pausing and beside training."""
    fake = {"max_episode_steps": 6, "stall_step": 2}
    for background in (False, True):
        for on in (True, False):
            config = tiny_config()
            config["evaluation"].update(background=background, end_when_stalled=on, interval=30, episodes=4)
            config["training"]["total_steps"] = 30
            name = f"run{background}{on}"
            train_and_close(parallel_trainer(tmp_path, name, config, fake=fake, num_envs=2, async_envs=True))
            evaluation = read_jsonl(tmp_path / name / "eval.jsonl")[0]
            stalled = evaluation["end_reasons"].get("stalled", 0)
            assert evaluation["end_reasons"].get("simulation_error", 0) == 0
            lengths = [e["length"] for e in read_jsonl(tmp_path / name / "episodes.jsonl") if not e["success"]]
            assert lengths and all(length == 6 for length in lengths)        # training: to the step limit
            if on:
                assert stalled >= 1 and stalled + evaluation["end_reasons"].get("success", 0) == 4
            else:
                assert stalled == 0


def test_evaluate_on_one_env_ends_stalled_episodes_only_when_asked():
    agent = build_agent(tiny_config(), observation_dim=4, action_dim=2, episode_length=6)
    env = FakeEnv(max_episode_steps=6, stall_step=2)
    asked = evaluate(env, agent, [3, 4, 5], end_when_stalled=True)
    plain = evaluate(env, agent, [3, 4, 5])
    assert asked["summary"]["end_reasons"].get("stalled", 0) >= 1
    assert "stalled" not in plain["summary"]["end_reasons"]
    assert all(e["length"] <= 2 or e["success"] for e in asked["episodes"])
