"""The training monitor (docs/wandb_metrics.md): the wandb forwarder's flattening and its inert mode,
the fuller evaluation summary (interval, end reasons, success steps, force peaks, progress histogram),
Q against the realized return, the reward broken into its terms, the prefix share and the termination
head's recall, all without wandb's servers (a fake wandb module) and without SOFA."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from guidewire_rl.env.two_device import reward_formula as rf
from guidewire_rl.train.evaluate import q_versus_return, summarize
from guidewire_rl.train.logger import MetricAverager, RunLogger
from guidewire_rl.train.metrics import window_summary
from guidewire_rl.train.wandb_sink import CORE_CURVES, DASHBOARD, PROGRESS_BINS, WandbSink, dashboard, flatten


def test_flatten_keeps_numbers_nests_with_slashes_and_drops_text():
    record = {"step": 10, "success": True, "end": "truncated", "reach": {"P2": 0.5, "P3": None},
              "start_weights": {"0": 0.3, "10": 0.7}, "stratum_starts": [1, 2], "memory_mb": {"main": 1.5},
              "nan": float("nan"), "events": {"wrong_vessel": 2}, "array": np.array([0.5, 1.5])}
    flat = flatten(record, "metrics")
    assert flat == {"metrics/step": 10.0, "metrics/success": 1.0, "metrics/reach/P2": 0.5,
                    "metrics/start_weights/0": 0.3, "metrics/start_weights/10": 0.7,
                    "metrics/stratum_starts/0": 1.0, "metrics/stratum_starts/1": 2.0,
                    "metrics/memory_mb/main": 1.5, "metrics/events/wrong_vessel": 2.0,
                    "metrics/array/0": 0.5, "metrics/array/1": 1.5}


class FakeRun:
    def __init__(self) -> None:
        self.logged, self.metrics, self.finished = [], [], False

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def log(self, data):
        self.logged.append(dict(data))

    def finish(self):
        self.finished = True


def _fake_wandb(monkeypatch):
    fake = types.ModuleType("wandb")
    created = {}

    def init(**kwargs):
        created.update(kwargs)
        return FakeRun()

    fake.init = init
    fake.Settings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return created


def test_sink_with_select_all_forwards_every_jsonl_line_with_the_env_step(tmp_path, monkeypatch):
    created = _fake_wandb(monkeypatch)
    config = {"logging": {"wandb": {"enabled": True, "project": "p", "mode": "offline", "tags": ["t"],
                                    "select": "all"}},
              "training": {"seed": 1}}
    sink = WandbSink.from_config(config, tmp_path / "run")
    assert sink.enabled and created["project"] == "p" and created["mode"] == "offline"
    assert created["dir"] == str(tmp_path / "run") and created["name"] == "run" and created["tags"] == ["t"]
    assert sink.run.metrics[0] == ("env_step", {"hidden": True})            # the x-axis: no chart of its own
    assert sink.run.metrics[1] == ("*", {"step_metric": "env_step"})
    assert created["settings"] == {"x_disable_stats": True}                 # no wandb hardware curves
    logger = RunLogger(tmp_path / "run", quiet=True, sink=sink)
    logger.write("eval", {"step": 100, "success_rate": 0.5, "end_reasons": {"success": 25, "other": 25}})
    logger.write("events", {"step": 101, "event": "checkpoint", "path": "x"})       # text only: nothing sent
    logger.write("updates", {"step": 102, "total_loss": 1.25})
    assert sink.sent == 2
    assert sink.run.logged[0] == {"eval/success_rate": 0.5, "eval/end_reasons/success": 25.0,
                                  "eval/end_reasons/other": 25.0, "env_step": 100.0}
    assert sink.run.logged[1]["updates/total_loss"] == 1.25 and sink.run.logged[1]["env_step"] == 102.0
    assert (tmp_path / "run" / "eval.jsonl").exists()                           # the jsonl is still written
    logger.close()
    assert sink.run.finished and not sink.enabled
    # off, or not installed: inert, nothing else changes
    assert not WandbSink.from_config({"logging": {"wandb": {"enabled": False}}}, tmp_path).enabled
    assert not WandbSink.from_config({"logging": {}}, tmp_path).enabled
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert not WandbSink.from_config(config, tmp_path / "run2").enabled
    with pytest.raises(KeyError):
        WandbSink.from_config({"logging": {"wandb": {"enabled": True, "projekt": "p"}}}, tmp_path)


def test_select_dashboard_sends_only_the_listed_curves(tmp_path, monkeypatch):
    """User 2026-11-11: wandb gets only the panels of docs/wandb_metrics.md; the jsonl keeps everything."""
    _fake_wandb(monkeypatch)
    sink = WandbSink.from_config({"logging": {"wandb": {"enabled": True, "project": "p", "select": "dashboard"}}},
                                 tmp_path / "run")
    assert sink.select == "dashboard"
    logger = RunLogger(tmp_path / "run", quiet=True, sink=sink)
    metrics = {"step": 3000, "episodes": 5, "success_rate": 0.2, "tier1_rate": 0.4, "return_mean": -1.0,
               "progress_end_mean": 0.5, "env_errors": 1, "wall_breaches": 0, "steps_per_s_recent": 8.0,
               "reward_parts": {"progress": 1.25, "follow": 1.5, "force_cath": -1.0, "force_wire": -0.5,
                                "step": -1.5},
               "events_per_episode": {"wrong_vessel": 0.5}, "memory_mb": {"available": 2048.0},
               "shield_active_frac": 0.25, "d_p50": 80.0, "catheter_moving_frac": 0.9,
               "reach": {"P1": 0.4}, "progress_end_hist": {"3": 2}}
    logger.write("metrics", metrics)
    logger.write("episodes", {"step": 3001, "return": 1.0, "length": 800})          # per episode: local only
    logger.write("events", {"step": 3002, "event": "checkpoint", "path": "x"})
    logger.write("updates", {"step": 3003, "consistency_loss": 0.01, "total_loss": 0.5, "model_grad_norm": 1.5})
    logger.write("eval", {"step": 3004, "success_rate": 0.1, "q_mean": 0.3, "g_mean": 0.2,
                          "end_reasons": {"success": 5}})
    assert sink.sent == 3
    sent_metrics, sent_updates, sent_eval = sink.run.logged
    bins = {f"train/progress_end/{name}": 0.0 for name in PROGRESS_BINS.values()}
    bins["train/progress_end/bin_20_30"] = 1.0                    # two episodes in bin 3: the whole share
    assert sent_metrics == {
        "train/train_tier2_rate": 0.2, "train/train_tier1_rate": 0.4, "train/progress_end_mean": 0.5,
        "train/return_mean": -1.0, **bins,
        "train/reward/progress": 1.25, "train/reward/follow": 1.5, "train/reward/force_cath": -1.0,
        "train/reward/force_wire": -0.5, "train/reward/step": -1.5, "train/reward/tier1": 0.0,
        "train/reward/tier2": 0.0, "train/reward/wrong_vessel": 0.0, "train/reward/failure": 0.0,
        "train/reward/smooth": 0.0,                                # smoothness term (2026-11-13)
        "train/reward_hint": 2.75, "train/reward_penalty": -3.0,   # no tier / failure terms: no finish total
        "train/wrong_vessel_per_ep": 0.5, "train/shield_active_frac": 0.25, "train/lead_p50": 80.0,
        "train/cath_moving_frac": 0.9,
        "sim/steps_per_s_recent": 8.0, "sim/sim_errors_total": 1.0, "sim/wall_breaches_total": 0.0,
        "sim/mem_available_gb": 2.0, "sim/episodes_total": 5.0,
        "env_step": 3000.0}                       # no crutches panel: the run has no curriculum / schedule
    assert sent_updates == {"learning/consistency_loss": 0.01, "learning/model_grad_norm": 1.5,
                            "learning/total_loss": 0.5, "env_step": 3003.0}
    assert sent_eval == {"eval/tier2_success": 0.1, "learning/q_mean": 0.3, "learning/g_mean": 0.2,
                         "env_step": 3004.0}      # end_reasons needs the episode count to become shares
    # the jsonl files still hold every field of every record
    assert (tmp_path / "run" / "episodes.jsonl").read_text().count("\n") == 1
    assert "progress_end_hist" in (tmp_path / "run" / "metrics.jsonl").read_text()
    logger.close()
    # end reasons as shares of the test episodes, all six layers always there (stacked chart)
    ends = dashboard("eval", {"step": 1, "episodes": 50, "end_reasons": {"success": 10, "failure:outside": 5,
                                                                        "other": 35}})
    assert ends == {"eval/end/success": 0.2, "eval/end/hard_failure": 0.1, "eval/end/sim_error": 0.0,
                    "eval/end/wall_breach": 0.0, "eval/end/stalled": 0.0, "eval/end/step_limit": 0.7}
    # the evaluation's wall-clock time comes from its event (eval.jsonl stays reproducible); other events: nothing
    assert dashboard("events", {"step": 5, "event": "evaluation", "minutes": 12.5}) == {"sim/eval_minutes": 12.5}
    assert dashboard("events", {"step": 5, "event": "checkpoint", "minutes": 3.0}) == {}
    # the crutches panel appears only when its numbers are there; five panels in all
    assert dashboard("metrics", {"step": 1, "start_frontier": 10, "demo_fraction": 0.1,
                                 "schedule_stage_index": 1}) == {
        "crutches/start_frontier": 10.0, "crutches/demo_fraction": 0.1, "crutches/schedule_stage": 1.0}
    assert {name.split("/")[0] for name, _, _ in DASHBOARD} == {"eval", "learning", "train", "crutches", "sim"}
    assert all(name.isascii() and not name[0].isdigit() for name, _, _ in DASHBOARD)  # English, no numbers
    with pytest.raises(ValueError):
        WandbSink.from_config({"logging": {"wandb": {"enabled": True, "select": "some"}}}, tmp_path)


def test_core_is_the_default_and_sends_only_the_core_views_curves(tmp_path, monkeypatch):
    """User 2026-11-11: wandb gets only what the core view draws; the jsonl files keep everything."""
    _fake_wandb(monkeypatch)
    sink = WandbSink.from_config({"logging": {"wandb": {"enabled": True, "project": "p"}}}, tmp_path / "run")
    assert sink.select == "core" and len(CORE_CURVES) == 52     # 34 + curriculum and schedule (user 2026-11-13) + eval/end/stalled
    logger = RunLogger(tmp_path / "run", quiet=True, sink=sink)
    logger.write("metrics", {"step": 3000, "episodes": 5, "success_rate": 0.2, "return_mean": -1.0,
                             "progress_end_mean": 0.5, "env_errors": 1, "wall_breaches": 0, "steps_per_s": 6.0,
                             "steps_per_s_recent": 8.0, "d_p10": 10.0, "d_p50": 80.0, "d_p90": 140.0,
                             "shield_active_frac": 0.25, "catheter_moving_frac": 0.9, "memory_mb": {"available": 2048.0},
                             "reward_parts": {"progress": 1.25, "follow": 1.5, "force_cath": -1.0, "force_wire": -0.5,
                                              "step": -1.5, "wrong_vessel": 0.0, "tier1": 1.0, "tier2": 0.0,
                                              "failure": 0.0}})
    logger.write("updates", {"step": 3001, "consistency_loss": 0.01, "reward_loss": 0.2, "value_loss": 0.3,
                             "model_grad_norm": 1.5, "policy_grad_norm": 0.01, "policy_entropy": -38.0,
                             "policy_loss": -1.0, "aux_cath_clearance_loss": 0.001})
    logger.write("events", {"step": 3002, "event": "evaluation", "minutes": 12.5})     # not a core curve
    logger.write("eval", {"step": 3003, "episodes": 50, "success_rate": 0.1, "tier1_rate": 0.2,
                          "success_ci95_low": 0.04, "success_ci95_high": 0.21, "length_mean": 800.0,
                          "end_reasons": {"success": 5, "other": 45}})
    sent_metrics, sent_updates, sent_eval = sink.run.logged
    assert sent_metrics == {
        "train/return_mean": -1.0, "train/progress_end_mean": 0.5, "train/reward_hint": 2.75,
        "train/reward_penalty": -3.0, "train/reward_finish": 1.0, "train/lead_p10": 10.0, "train/lead_p50": 80.0,
        "train/lead_p90": 140.0, "train/shield_active_frac": 0.25, "sim/steps_per_s_recent": 8.0,
        "sim/steps_per_s": 6.0, "sim/sim_errors_total": 1.0, "sim/wall_breaches_total": 0.0, "env_step": 3000.0}
    assert sent_updates == {"learning/consistency_loss": 0.01, "learning/reward_loss": 0.2,
                            "learning/model_grad_norm": 1.5, "learning/policy_grad_norm": 0.01,
                            "learning/policy_entropy": -38.0, "env_step": 3001.0}
    assert sent_eval == {"eval/tier2_success": 0.1, "eval/tier2_ci95_low": 0.04, "eval/tier2_ci95_high": 0.21,
                         "eval/tier1_rate": 0.2, "eval/end/success": 0.1, "eval/end/hard_failure": 0.0,
                         "eval/end/sim_error": 0.0, "eval/end/wall_breach": 0.0, "eval/end/stalled": 0.0,
                         "eval/end/step_limit": 0.9,
                         "env_step": 3003.0}
    assert sink.sent == 3
    logger.close()
    assert len(set(CORE_CURVES)) == len(CORE_CURVES)                            # no curve listed twice


def test_the_core_view_sends_the_curriculum_and_schedule_of_a_run_that_has_them():
    """User 2026-11-13: frontier, schedule stage, demonstration quota, guard trips, ladder level and the
    training success by start grid are core curves. A grid without episodes sends nothing (no 0 fill);
    the names are the ones the charts use (grid_00 ... grid_10)."""
    from guidewire_rl.train.wandb_sink import core
    sent = core("metrics", {"step": 9, "start_frontier": 9, "schedule_stage_index": 0, "demo_fraction": 0.25,
                            "guard_trips": 0, "ladder_level": 0, "hint_scale": 1.0, "force_scale": 0.25,
                            "start_weights": {"0": 0.3},
                            "success_by_start_grid": {"0": 0.0, "5": None, "9": 0.5, "10": 0.83}})
    assert sent == {"crutches/start_frontier": 9.0, "crutches/schedule_stage": 0.0, "crutches/demo_fraction": 0.25,
                    "crutches/force_scale": 0.25, "crutches/guard_trips": 0.0, "crutches/ladder_level": 0.0,
                    "crutches/success_by_start/grid_00": 0.0, "crutches/success_by_start/grid_09": 0.5,
                    "crutches/success_by_start/grid_10": 0.83}
    assert core("metrics", {"step": 9, "success_rate": 0.1}) == {}        # a run without them: none of these


def test_metric_averager_averages_a_sometimes_missing_key_over_its_own_count():
    averager = MetricAverager()
    averager.add({"loss": 1.0, "termination_recall": None})
    averager.add({"loss": 3.0, "termination_recall": 0.5})
    averager.add({"loss": 2.0, "termination_recall": 1.0})
    means = averager.pop()
    assert means["loss"] == pytest.approx(2.0) and means["termination_recall"] == pytest.approx(0.75)
    assert averager.pop() == {}


def test_q_versus_return_is_the_discounted_tail():
    out = q_versus_return([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], discount=0.5)
    np.testing.assert_allclose([out["g0"], out["g_mean"]], [0.25, (0.25 + 0.5 + 1.0) / 3])
    assert out["q0"] == 1.0 and out["q_mean"] == 1.0
    assert out["q_minus_g_mean"] == pytest.approx(1.0 - (0.25 + 0.5 + 1.0) / 3)


def test_reward_components_add_up_to_the_formula():
    rng = np.random.default_rng(0)
    m = np.zeros((7, len(rf.FIELDS)))
    m[:, rf.INDEX["progress_mm"]] = rng.uniform(-2, 5, 7)
    m[:, rf.INDEX["follow_mm"]] = rng.uniform(-2, 5, 7)
    m[:, rf.INDEX["cath_force_N"]] = rng.uniform(0, 6, 7)
    m[:, rf.INDEX["wire_force_N"]] = rng.uniform(0, 2, 7)
    m[2, rf.INDEX["wrong_vessel"]] = 1.0
    m[4, rf.INDEX["tier1"]] = 1.0
    m[6, rf.INDEX["success"]] = 1.0
    m[3, rf.INDEX["sim_error"]] = 1.0
    c = rf.Coefficients()
    parts = rf.components(m, c)
    assert set(parts) == set(rf.COMPONENTS)
    np.testing.assert_allclose(sum(parts.values()), rf.compute(m, c), atol=1e-12)
    assert all(v[3] == 0.0 for v in parts.values())                     # the dropped step pays nothing
    sums = rf.component_sums(m, c)
    assert sums["tier1"] == pytest.approx(1.0) and sums["tier2"] == pytest.approx(1.0)
    assert sums["wrong_vessel"] == pytest.approx(-0.1) and sums["step"] == pytest.approx(-0.002 * 6)
    assert sums["force_cath"] <= 0.0 and sums["force_wire"] <= 0.0


def test_evaluation_summary_has_the_dashboard_numbers():
    episodes = []
    for k in range(10):
        success = k < 6
        episodes.append({"seed": k, "return": 1.0 if success else 0.0, "length": 300 + 10 * k, "success": success,
                         "simulation_error": False, "wall_breach": k == 9, "true_end": success,
                         "distal_force_max": 0.1 * k, "catheter_tip_force_max": 0.5 * k,
                         "max_phase": 4, "progress_end": 1.0 if success else 0.55, "lead_end": 20.0 + k,
                         "tier1": True, "q_mean": 2.0, "g_mean": 1.5, "q_minus_g_mean": 0.5, "q0": 3.0, "g0": 2.0,
                         "reward_parts": {"progress": 3.0, "force_cath": -0.02}})
    summary = summarize(episodes)
    assert summary["success_rate"] == pytest.approx(0.6)
    assert summary["success_ci95_low"] < 0.6 < summary["success_ci95_high"]
    assert summary["end_reasons"] == {"other": 3, "success": 6, "wall_breach": 1}
    assert summary["success_steps_mean"] == pytest.approx(325.0) and summary["success_steps_median"] == 325.0
    assert summary["distal_force_peak_max"] == pytest.approx(0.9) and summary["catheter_tip_force_peak_p90"] > 0
    assert summary["progress_end_hist"] == {"6": 4, "done": 6}
    assert summary["lead_end_below_30_rate"] == pytest.approx(1.0)
    assert summary["q_versus_return_episodes"] == 6 and summary["q_minus_g_mean"] == pytest.approx(0.5)
    assert summary["reward_parts"] == {"progress": pytest.approx(3.0), "force_cath": pytest.approx(-0.02)}
    assert summary["tier1_rate"] == 1.0
    assert summary["tier1_ci95_low"] < 1.0 and summary["tier1_ci95_high"] == pytest.approx(1.0)


def test_q_versus_return_only_from_episodes_that_really_ended():
    """WORK_LOG 2026-11-11 18:45: with every test episode cut at the step limit there is nothing to
    compare (the cut episodes' realized return misses its tail), so no Q numbers at all."""
    cut = [{"seed": k, "return": 0.5, "length": 800, "success": False, "simulation_error": False,
            "wall_breach": False, "true_end": False, "q_mean": 5.0, "g_mean": -0.2, "q_minus_g_mean": 5.2,
            "q0": 6.0, "g0": -0.1} for k in range(4)]
    summary = summarize(cut)
    assert not any(k.startswith(("q_", "g_")) or k in ("q0", "g0") for k in summary)
    ended = dict(cut[0], true_end=True, success=True, q_mean=1.0, g_mean=0.9, q_minus_g_mean=0.1, q0=1.2, g0=1.0)
    summary = summarize(cut + [ended])
    assert summary["q_versus_return_episodes"] == 1 and summary["q_mean"] == pytest.approx(1.0)
    assert summary["q_minus_g_mean"] == pytest.approx(0.1)


def test_window_summary_has_prefix_share_and_reward_parts():
    records = [dict(success=True, **{"return": 5.0}, length=100, prefix_steps=40, prefix_outcome="handover",
                    reward_parts={"progress": 3.0, "step": -0.2}),
               dict(success=False, **{"return": 1.0}, length=60, prefix_steps=20, prefix_outcome="gave_up",
                    reward_parts={"progress": 1.0, "step": -0.12})]
    out = window_summary(records)
    assert out["prefix_step_share"] == pytest.approx(60 / 160)
    assert out["prefix_gave_up_rate"] == pytest.approx(0.5)
    assert out["reward_parts"] == {"progress": 2.0, "step": pytest.approx(-0.16)}


def test_the_wandb_layout_puts_every_dashboard_curve_in_a_chart():
    """scripts/wandb_layout.py lays the DASHBOARD curves out (charts, stacked shares, log scale)."""
    import importlib.util
    from guidewire_rl import WORKSPACE_ROOT
    spec = importlib.util.spec_from_file_location("wandb_layout", WORKSPACE_ROOT / "scripts" / "wandb_layout.py")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    assert layout.check(crutches=True) == ([], [])
    assert [panel for panel, _ in layout.charts(crutches=False)] == ["eval", "learning", "train", "sim"]
    # the core view (user 2026-11-11): 16 charts in four sections, the two tiers on separate charts; since
    # 2026-11-13 a fifth section, curriculum and schedule (5 charts)
    assert sum(len(rows) for _, rows in layout.CORE) == 22
    assert [section for section, _ in layout.CORE] == ["Task (test episodes)", "Training episodes (last 100)",
                                                       "World model and value", "Simulator",
                                                       "Curriculum and schedule"]
    titles = [title for _, rows in layout.CORE for title, _, _, _ in rows]
    assert titles[:2] == ["Tier 2 success, 95% interval", "Tier 1 rate, 95% interval"]


def test_a_resumed_run_continues_its_last_wandb_run(tmp_path, monkeypatch):
    """Resuming from a checkpoint (several times, e.g. to change the number of simulators) keeps one
    wandb run: the id behind <run_dir>/wandb/latest-run, resume="allow"; a fresh run gets a new one."""
    from guidewire_rl.train.wandb_sink import previous_run_id
    created = _fake_wandb(monkeypatch)
    run_dir = tmp_path / "run"
    config = {"logging": {"wandb": {"enabled": True, "project": "p"}}}
    WandbSink.from_config(config, run_dir)
    assert "id" not in created and "resume" not in created
    (run_dir / "wandb" / "run-20261112_014521-abc123").mkdir(parents=True)
    (run_dir / "wandb" / "latest-run").symlink_to("run-20261112_014521-abc123")
    assert previous_run_id(run_dir) == "abc123"
    created.clear()
    WandbSink.from_config(config, run_dir, resume=True)
    assert created["id"] == "abc123" and created["resume"] == "allow"
    created.clear()
    WandbSink.from_config(config, run_dir)                      # not a resume: a new wandb run
    assert "id" not in created
    assert previous_run_id(tmp_path / "elsewhere") is None
