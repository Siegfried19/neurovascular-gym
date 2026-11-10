import json

from guidewire_rl.train.setup import seed_everything, start_run

from conftest import tiny_config
from fakes import FakeEnv


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_schedule_logs_and_checkpoints(tmp_path):
    config = tiny_config()            # 60 steps: 20 random, burst of 10 updates, then 1/step
    seed_everything(0)
    trainer = start_run(config, FakeEnv(), tmp_path / "run", quiet=True)
    summary = trainer.train()
    assert summary["total_env_steps"] == 60
    assert summary["updates"] == 10 + (60 - 20)
    for name in ("step_30.pt", "step_60.pt"):
        assert (tmp_path / "run" / "checkpoints" / name).is_file()
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    assert sum(e["length"] for e in episodes) == trainer.replay.size
    assert {"distal_force_max", "distal_force_mean"} <= set(episodes[0])
    assert read_jsonl(tmp_path / "run" / "eval.jsonl")
    updates = read_jsonl(tmp_path / "run" / "updates.jsonl")
    assert updates and "aux_distal_force_loss" in updates[0]
    assert (tmp_path / "run" / "config.yaml").is_file()


def test_resume_continues_counters_and_replay(tmp_path):
    seed_everything(0)
    start_run(tiny_config(), FakeEnv(), tmp_path / "a", quiet=True).train()
    checkpoint = tmp_path / "a" / "checkpoints" / "step_30.pt"
    seed_everything(0)
    resumed = start_run(tiny_config(), FakeEnv(), tmp_path / "a", resume=checkpoint, quiet=True)
    assert resumed.step == 30 and resumed.replay.size > 0
    before = resumed.agent.update_count
    summary = resumed.train()
    assert summary["total_env_steps"] == 60
    assert summary["updates"] == before + 30


def test_simulator_failure_drops_the_step_and_keeps_going(tmp_path):
    config = tiny_config()
    config["evaluation"]["interval"] = 1000      # keep env calls == training steps
    seed_everything(0)
    trainer = start_run(config, FakeEnv(error_at_calls={7, 41}), tmp_path / "run", quiet=True)
    summary = trainer.train()
    assert summary["total_env_steps"] == 60 and summary["env_errors"] == 2
    events = [e["event"] for e in read_jsonl(tmp_path / "run" / "events.jsonl")]
    assert events.count("dropped_step") == 2


def test_aux_head_must_name_an_env_label(tmp_path):
    config = tiny_config()
    config["aux_heads"][0]["label"] = "tip_speed"
    try:
        start_run(config, FakeEnv(), tmp_path / "run", quiet=True)
    except ValueError as exc:
        assert "tip_speed" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
