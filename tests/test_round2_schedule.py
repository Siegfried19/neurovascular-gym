"""The round-two training schedule (guidewire_rl/train/schedule.py): ramps under the guard, the
stages, the stop rule, the ladder's level 1, and the trainer wiring (best checkpoint, knobs into the
replay, stopping), without SOFA."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from guidewire_rl.env.two_device import reward_formula as rf
from guidewire_rl.train.schedule import Ramp, Schedule
from guidewire_rl.train.setup import start_run

from test_round2_training import Round2Env, round2_config


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def guard(ramp, step, rate):
    return ramp.on_evaluation(step, rate, drop=0.15, floor=0.30, hold_evals=3)


def test_ramp_goes_down_linearly_and_the_guard_rolls_back_holds_and_halves():
    ramp = Ramp(start=0.10, steps=100)
    assert guard(ramp, 0, 0.8) == "start" and ramp.baseline == 0.8 and ramp.value == pytest.approx(0.10)
    assert guard(ramp, 25, 0.8) == "advance" and ramp.value == pytest.approx(0.075)
    assert guard(ramp, 50, 0.7) == "advance" and ramp.value == pytest.approx(0.05)    # -10 points: fine
    assert guard(ramp, 75, 0.6) == "rollback" and ramp.value == pytest.approx(0.075)  # -20: back one evaluation
    assert [guard(ramp, s, 0.8) for s in (100, 125)] == ["hold", "hold"]
    assert guard(ramp, 150, 0.8) == "resume_half_speed" and ramp.speed == 0.5
    assert guard(ramp, 175, 0.8) == "advance" and ramp.value == pytest.approx(0.075 - 0.0125)
    assert guard(ramp, 200, 0.25) == "frozen" and ramp.over                          # below 30 %: second trigger
    frozen = ramp.value
    assert guard(ramp, 225, 0.9) is None and ramp.value == frozen
    # still bad when the hold ends: hold on
    ramp = Ramp(start=1.0, steps=10)
    guard(ramp, 0, 0.8)
    guard(ramp, 1, 0.5)
    assert [guard(ramp, s, 0.5) for s in (2, 3, 4, 5)] == ["hold"] * 4
    assert guard(ramp, 6, 0.8) == "resume_half_speed"


def test_stages_move_on_the_evaluations_and_the_stop_rule_ends_the_run():
    curriculum = SimpleNamespace(frontier=3, rates=lambda: {})
    s = Schedule(demo_fraction=0.10, demo_ramp_steps=100, stop_patience=2, ladder=False)
    assert s.on_evaluation(0, 0.9, curriculum) == [] and s.stage == "curriculum"   # frontier not home yet
    curriculum.frontier = 0
    moves = s.on_evaluation(25, 0.4, curriculum)                                    # merged 0.65: ramp starts
    assert moves[0] == {"what": "stage", "from": "curriculum", "to": "demo_ramp", "step": 25, "rate": 0.65}
    assert s.demo_value == pytest.approx(0.10)
    s.on_evaluation(75, 0.9, curriculum)
    assert s.demo_value == pytest.approx(0.05)
    moves = s.on_evaluation(125, 0.9, curriculum)
    assert s.demo_value == 0.0 and s.stage == "wrap_up" and not s.stop
    assert s.best == 0.9                                  # the evaluation that opens the wrap-up counts
    s.on_evaluation(150, 0.85, curriculum)                # no better: 1
    assert not s.stop
    moves = s.on_evaluation(175, 0.85, curriculum)        # no better: 2 = patience
    assert s.stop and s.stage == "done" and moves[-1]["to"] == "done"
    # the optional hint ramp sits between the demonstration ramp and the wrap-up
    s = Schedule(demo_fraction=0.0, hint_ramp=True, hint_ramp_steps=50, ladder=False)
    s.on_evaluation(0, 0.9)
    assert s.stage == "hint_ramp" and s.hint_scale == 1.0
    s.on_evaluation(25, 0.9)
    assert s.hint_scale == pytest.approx(0.5)
    s.on_evaluation(50, 0.9)
    assert s.hint_scale == 0.0 and s.stage == "wrap_up"
    with pytest.raises(KeyError):
        Schedule.from_config({"stop_patiance": 3}, 0.1)
    assert Schedule.from_config(None, 0.1) is None


def test_the_ladder_turns_the_arch_hint_on_for_a_stuck_grid_and_off_when_it_passes():
    state = {"frontier": 4, "rates": {"4": 0.05, "5": 0.8}}
    curriculum = SimpleNamespace(frontier=4, rates=lambda: dict(state["rates"]))
    s = Schedule(demo_fraction=0.1, stuck_evals=3, stuck_steps=100, level_steps=200)
    for step in (50, 100, 150):
        assert not s.arch_hint and s.on_evaluation(step, 0.2, curriculum) == []
    moves = s.on_evaluation(200, 0.2, curriculum)
    assert s.arch_hint and s.ladder_level == 1 and moves[0]["what"] == "ladder_level_1" and moves[0]["grid"] == 4
    moves = s.on_evaluation(450, 0.2, curriculum)                       # level 1 spent its steps
    assert moves[0]["what"] == "ladder_needs_advisor" and s.arch_hint
    state["rates"]["4"] = 0.6                                           # the grid is learned: hint off
    moves = s.on_evaluation(500, 0.3, curriculum)
    assert moves[0]["what"] == "ladder_off" and not s.arch_hint and s.ladder_level == 0
    restored = Schedule(demo_fraction=0.1)
    restored.load_state_dict(json.loads(json.dumps(s.state_dict())))
    assert restored.state_dict() == s.state_dict()


def test_the_trainer_keeps_the_best_checkpoint_applies_the_knobs_and_stops(tmp_path):
    config = round2_config()
    config["training"].pop("start_curriculum")
    config["training"]["schedule"] = {"pass_rate": 0.0, "stop_patience": 1, "ladder": False}
    config["training"]["total_steps"] = 2000
    config["evaluation"].update(interval=60, episodes=2, keep_best=True)
    trainer = start_run(config, Round2Env(max_episode_steps=40), tmp_path / "run", quiet=True)
    summary = trainer.train()
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    stages = [e for e in events if e["event"] == "schedule" and e["what"] == "stage"]
    assert [e["to"] for e in stages][:2] == ["demo_ramp", "wrap_up"] and stages[-1]["to"] == "done"
    assert summary["total_env_steps"] < 2000                            # the stop rule ended it
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    assert any(e["event"] == "best_checkpoint" for e in events)
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert metrics[-1]["schedule_stage"] in ("wrap_up", "done")
    # the knobs: the arch hint and the hint factor go into the replay's reward recomputation
    materials = np.zeros((1, len(rf.FIELDS)), np.float32)
    materials[0, rf.INDEX["progress_mm"]] = 5.0
    materials[0, rf.INDEX["hook_progress_mm"]] = 2.0
    before = float(trainer.replay.reward_fn(materials)[0])
    trainer.schedule.arch_hint = True
    trainer._apply_schedule()
    assert float(trainer.replay.reward_fn(materials)[0]) == pytest.approx(before - 0.03)
    trainer.schedule.arch_hint = False
    trainer.schedule.hint = SimpleNamespace(value=0.5)
    trainer._apply_schedule()
    assert float(trainer.replay.reward_fn(materials)[0]) == pytest.approx(before - 0.025)


def test_the_force_factor_starts_light_and_climbs_to_one_under_the_guard():
    """training.schedule force_start (user 2026-11-13, "先轻后重"): the factor on the force penalty's cap
    stays at force_start until the merged success rate reaches the line, then climbs linearly to 1
    beside the stages; the guard takes it back down; the stop rule waits while it climbs."""
    s = Schedule(demo_fraction=0.1, ladder=False)                     # no force keys: no factor, no events
    assert s.force_scale == 1.0 and s.on_evaluation(0, 0.9) and s.force is None
    assert not any(m["what"].startswith("force_ramp") for m in s.on_evaluation(25, 0.9))
    s = Schedule.from_config({"force_start": 0.25, "force_ramp_steps": 100, "ladder": False}, 0.1)
    curriculum = SimpleNamespace(frontier=3, rates=lambda: {})        # the stages stay in "curriculum"
    assert s.force_scale == 0.25
    assert s.on_evaluation(0, 0.2, curriculum) == [] and s.force_scale == 0.25      # below the line: light
    moves = s.on_evaluation(25, 0.9, curriculum)                                    # merged 0.55: it begins
    assert len(moves) == 1 and moves[0]["what"] == "force_ramp_start" and moves[0]["step"] == 25
    assert moves[0]["rate"] == pytest.approx(0.55) and moves[0]["force_scale"] == 0.25
    s.on_evaluation(50, 0.9, curriculum)
    assert s.force_scale == pytest.approx(0.25 + 0.75 * 25 / 100) and s.stage == "curriculum"
    s.on_evaluation(75, 0.9, curriculum)
    assert s.force_scale == pytest.approx(0.625) and s.force_climbing
    s.on_evaluation(100, 0.2, curriculum)                             # merged (0.9 + 0.2) / 2 = 0.55, the
    assert s.force_scale == pytest.approx(0.625 + 0.1875)             # baseline: no drop, it advances
    moves = s.on_evaluation(125, 0.2, curriculum)                     # merged 0.2: below the floor, back one
    assert moves[-1]["what"] == "force_ramp_rollback" and s.force_scale == pytest.approx(0.625)
    assert s.guard_trips == 1
    restored = Schedule.from_config({"force_start": 0.25, "force_ramp_steps": 100, "ladder": False}, 0.1)
    restored.load_state_dict(json.loads(json.dumps(s.state_dict())))
    assert restored.state_dict() == s.state_dict() and restored.force_scale == pytest.approx(0.625)
    # to the end: exactly 1, the ramp is over
    s = Schedule(demo_fraction=0.0, force_start=0.5, force_ramp_steps=50, pass_rate=0.0, stop_patience=1,
                 ladder=False)
    s.on_evaluation(0, 0.9)                                           # stages: demo_ramp over at once -> wrap_up
    assert s.stage == "wrap_up" and s.force_scale == 0.5 and s.force_climbing
    s.on_evaluation(25, 0.9)                                          # no better: patience spent, but climbing
    assert s.force_scale == pytest.approx(0.75) and not s.stop
    s.on_evaluation(50, 0.9)
    assert s.force_scale == 1.0 and not s.force_climbing and s.stop   # at 1: the stop rule may fire
    with pytest.raises(ValueError):
        Schedule(force_start=1.5)


def test_the_trainer_scales_the_force_penalty_of_the_replay_from_the_first_batch(tmp_path):
    config = round2_config()
    config["training"].pop("start_curriculum")
    config["training"]["schedule"] = {"ladder": False, "force_start": 0.25}
    trainer = start_run(config, Round2Env(max_episode_steps=40), tmp_path / "run", quiet=True)
    c = rf.Coefficients.from_config(config["env"]["reward_formula"])
    materials = np.zeros((1, len(rf.FIELDS)), np.float32)
    materials[0, rf.INDEX["cath_force_N"]] = c.cath_knee_N + c.cath_width_N + 1.0      # at the cap
    full = float(rf.compute(materials, c)[0])
    assert float(trainer.replay.reward_fn(materials)[0]) == pytest.approx(full + 0.75 * c.force_cap)
    trainer.schedule.force = SimpleNamespace(value=0.0, triggers=0, over=True)         # the ramp is done
    trainer._apply_schedule()
    assert float(trainer.replay.reward_fn(materials)[0]) == pytest.approx(full)
    trainer._print_progress(0.0, 0)
    assert read_jsonl(tmp_path / "run" / "metrics.jsonl")[-1]["force_scale"] == 1.0
