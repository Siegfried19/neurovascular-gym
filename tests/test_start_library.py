"""The start library (user 2026-11-12). Supply 'self', the policy's own: training episodes save a state
per new progress grid, later episodes start from those states, the curriculum draws only grids the
library can serve. Supply 'demos': a fixed library made beforehand from the scripted demonstrations
(scripts/record_demos.py --starts), read and never changed. The four training methods are four configs.
Everything without SOFA except the last test (one real save and restore through GuidewireEnv.reset)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from guidewire_rl.train.curriculum import StartCurriculum
from guidewire_rl.train.env_pool import reset_command
from guidewire_rl.train.start_library import StartLibrary


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_library_keeps_the_newest_states_per_grid_and_reads_itself_back(tmp_path):
    lib = StartLibrary(tmp_path / "starts", bins=10, cap=3, min_entries=2, seed=0)
    assert lib.available() == set() and lib.sample(4) is None
    paths = [touch(tmp_path / "starts" / "grid_04" / f"e0_{k}.npz") for k in range(5)]
    for p in paths:
        lib.add({"path": str(p), "grid": 4, "progress": 0.35, "step": 100})
    lib.add({"path": str(touch(tmp_path / "starts" / "grid_07" / "e1_0.npz")), "grid": 7})
    assert lib.counts()["4"] == 3 and lib.counts()["7"] == 1
    assert [p.exists() for p in paths] == [False, False, True, True, True]   # the oldest went, files too
    assert lib.available() == {4}                                            # grid 7 has 1 < min_entries
    assert lib.sample(4) in {str(p) for p in paths[2:]}
    lib.add({"path": "x", "grid": 0}) or lib.add({"path": "y", "grid": 11})  # out of range: ignored
    assert sum(lib.counts().values()) == 4
    # after a restart the folder is the library (oldest first by file time)
    for k, p in enumerate(paths[2:]):
        os.utime(p, (time.time() + k, time.time() + k))
    again = StartLibrary(tmp_path / "starts", bins=10, cap=3, min_entries=2)
    assert again.counts()["4"] == 3 and again.counts()["7"] == 1
    assert StartLibrary.from_config({"supply": "prefix"}, tmp_path, 0) is None
    made = StartLibrary.from_config({"supply": "self", "library_cap": 7, "library_min": 3}, tmp_path / "run", 0)
    assert made.cap == 7 and made.min_entries == 3 and made.directory == tmp_path / "run" / "starts"


def test_self_supply_draws_only_grids_the_library_can_serve_and_walks_back():
    c = StartCurriculum(supply="self", bins=10, baseline_fraction=0.3, min_episodes=4, update_every=10,
                        floor_weight=0.05, seed=0)
    c.set_available(set())
    assert c.training_grid == 0 and c.weights()[0] == 1.0 and c.sample() == (0, None)
    c.set_available({3, 5})                          # the policy has reached grid 5 so far
    assert c.training_grid == 5
    w = c.weights()
    assert w[0] == pytest.approx(0.3) and w[5] == pytest.approx(0.7) and w.sum() == pytest.approx(1.0)
    draws = {c.sample() for _ in range(200)}
    assert draws <= {(0, None), (5, None)} and (5, None) in draws         # no ratio: the library gives the state
    for _ in range(4):
        c.record(5, True)
    move = c.maybe_update(10)
    assert move["frontier_before"] == 10 and move["frontier"] == 4         # back from the grid trained
    assert c.training_grid == 3                                            # the highest served grid <= 4
    c.set_available({3, 5, 9})                                             # a far state appears later
    w = c.weights()
    assert w[9] == pytest.approx(0.05) and w[5] == pytest.approx(0.05) and w[3] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        StartCurriculum(supply="snapshotz")
    with pytest.raises(ValueError):
        StartCurriculum(supply="self", grid="landmarks")
    # the prefix supply is unchanged: ratios inside the frontier grid
    p = StartCurriculum(supply="prefix", bins=10, baseline_fraction=0.0, seed=1)
    k, ratio = p.sample()
    assert k == 10 and 0.9 <= ratio <= 1.0


def test_fixed_library_is_read_and_never_changed(tmp_path):
    """Supply 'demos': a library made beforehand keeps every state it finds, deletes nothing and takes
    no state from the training episodes."""
    files = [touch(tmp_path / "lib" / "grid_10" / f"demo_{k}_s0400.npz") for k in range(6)]
    touch(tmp_path / "lib" / "grid_03" / "demo_0_s0100.npz")
    lib = StartLibrary(tmp_path / "lib", bins=10, cap=3, min_entries=2, fixed=True)
    assert lib.counts()["10"] == 6 and all(p.exists() for p in files)        # the cap does not apply
    assert lib.available() == {10} and lib.saving_to is None                 # the envs are not asked to save
    lib.add({"path": str(touch(tmp_path / "elsewhere.npz")), "grid": 10})    # a training episode's state
    assert lib.counts()["10"] == 6 and lib.added == 0
    assert lib.sample(10) in {str(p) for p in files}
    own = StartLibrary(tmp_path / "run" / "starts", bins=10)
    assert own.saving_to == str(tmp_path / "run" / "starts")
    made = StartLibrary.from_config({"supply": "demos", "library": str(tmp_path / "lib"), "library_min": 2},
                                    tmp_path / "run", 0)
    assert made.fixed and made.directory == tmp_path / "lib" and made.available() == {10}
    with pytest.raises(FileNotFoundError):                                  # nothing to start from: say so
        StartLibrary.from_config({"supply": "demos", "library": str(tmp_path / "nothing")}, tmp_path, 0)
    with pytest.raises(ValueError):
        StartLibrary.from_config({"supply": "demos"}, tmp_path, 0)


def test_demos_supply_starts_at_the_last_grid_and_walks_back():
    c = StartCurriculum(supply="demos", library="runs/x/starts", bins=10, baseline_fraction=0.3,
                        min_episodes=4, update_every=10, floor_weight=0.05, seed=0)
    assert c.uses_library and not StartCurriculum(supply="prefix").uses_library
    c.set_available(set(range(1, 11)))                 # the demonstrations cover every grid
    assert c.training_grid == 10
    w = c.weights()
    assert w[0] == pytest.approx(0.3) and w[10] == pytest.approx(0.7)
    assert {c.sample() for _ in range(200)} == {(0, None), (10, None)}      # the library gives the state
    for _ in range(4):
        c.record(10, True)
    assert c.maybe_update(10)["frontier"] == 9 and c.training_grid == 9
    w = c.weights()
    assert w[10] == pytest.approx(0.05) and w[9] == pytest.approx(0.65)
    with pytest.raises(ValueError):
        StartCurriculum(supply="demos")                # no library named
    with pytest.raises(ValueError):
        StartCurriculum(supply="self", library="runs/x/starts")
    with pytest.raises(ValueError):
        StartCurriculum(supply="demos", library="runs/x/starts", grid="landmarks")
    made = StartCurriculum.from_config({"supply": "demos", "library": "runs/x/starts"}, 0)
    assert made.supply == "demos" and made.library == "runs/x/starts"


def test_states_of_a_recorded_episode_go_into_the_library_only_when_it_is_kept(tmp_path):
    """scripts/record_demos.py --starts: the states saved along a successful demonstration are kept,
    those of a failed one are dropped; the staging folder goes either way."""
    from guidewire_rl.train.start_library import collect_episode_states
    lib = tmp_path / "starts"
    good = lib / "_staging" / "seed_1"
    touch(good / "grid_01" / "demo_1_s0010.npz")
    touch(good / "grid_02" / "demo_1_s0030.npz")
    assert collect_episode_states(good, lib, keep=True) == 2
    assert (lib / "grid_01" / "demo_1_s0010.npz").exists() and (lib / "grid_02" / "demo_1_s0030.npz").exists()
    assert not good.exists()
    bad = lib / "_staging" / "seed_2"
    touch(bad / "grid_01" / "demo_2_s0012.npz")
    assert collect_episode_states(bad, lib, keep=False) == 0
    assert not bad.exists() and not (lib / "grid_01" / "demo_2_s0012.npz").exists()
    assert collect_episode_states(lib / "_staging" / "seed_3", lib, keep=True) == 0    # it saved nothing


def test_trainer_starts_episodes_from_the_fixed_library_and_leaves_it_alone(tmp_path):
    """The whole wiring without SOFA: the curriculum draws a grid, the trainer takes a state of the
    fixed library, the env starts there; no prefix, nothing saved, the library untouched."""
    from guidewire_rl.train.setup import start_run
    from test_round2_training import GOAL, Round2Env, read_jsonl, round2_config

    class LibraryEnv(Round2Env):
        """reset(options={"snapshot": path}) starts at the progress written in that file."""

        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed)
            path = (options or {}).get("snapshot")
            if path:
                ratio = float(np.load(path)["progress"])
                self.progress = ratio * GOAL
                observation, info = self._obs(), self._info()
                info["snapshot_start"] = {"path": path, "progress": ratio, "grid": int(ratio * 10) + 1}
            return observation, info

    library = tmp_path / "demo_starts"
    for k in range(1, 11):
        for j in range(5):
            path = library / f"grid_{k:02d}" / f"demo_{j}_s{10 * k:04d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, progress=(k - 0.5) / 10)
    before = sorted(str(p) for p in library.rglob("*.npz"))
    config = round2_config()
    config["training"]["start_curriculum"].update(supply="demos", library=str(library), baseline_fraction=0.0)
    trainer = start_run(config, LibraryEnv(max_episode_steps=40), tmp_path / "run", quiet=True)
    assert trainer.start_library is not None and trainer.start_library.fixed
    trainer.train()
    episodes = read_jsonl(tmp_path / "run" / "episodes.jsonl")
    events = read_jsonl(tmp_path / "run" / "events.jsonl")
    assert episodes and episodes[0]["start_grid"] == 10                      # the last grid first
    assert all(e["start_grid"] >= 1 and e["prefix_steps"] == 0 for e in episodes)
    assert not [e for e in events if e["event"] in ("prefix", "snapshot_failed")]
    assert trainer.start_curriculum.frontier < 10                           # learned there: it walked back
    assert sorted(str(p) for p in library.rglob("*.npz")) == before         # the library is as it was
    assert not (tmp_path / "run" / "starts").exists() and trainer._saver is None   # nothing was saved


def test_a_restored_episode_is_booked_under_the_grid_drawn(tmp_path):
    """User 2026-11-13 (work log 2026-11-12 21:45): a state saved on entering grid k sits on its lower edge;
    re-measured against another target its progress can fall just below it, in grid k - 1. The episode is
    still booked under grid k (the grid drawn, the state's library folder); its start ratio stays the
    re-measured progress. Runs with the schedule on, as the demonstrations-as-starts config does."""
    from guidewire_rl.train.setup import start_run
    from test_round2_training import GOAL, Round2Env, read_jsonl, round2_config

    class EdgeEnv(Round2Env):
        """reset(options={"snapshot": path}) starts 0.003 below the lower edge of the file's grid."""

        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed)
            path = (options or {}).get("snapshot")
            if path:
                k = int(Path(path).parent.name.split("_")[1])
                ratio = max(0.001, (k - 1) / 10 - 0.003)
                self.progress = ratio * GOAL
                observation, info = self._obs(), self._info()
                info["snapshot_start"] = {"path": path, "progress": ratio, "grid": k}
            return observation, info

    library = tmp_path / "demo_starts"
    for k in range(1, 11):
        for j in range(5):
            touch(library / f"grid_{k:02d}" / f"demo_{j}_s{10 * k:04d}.npz")
    config = round2_config()
    config["training"]["start_curriculum"].update(supply="demos", library=str(library), baseline_fraction=0.0)
    config["training"]["schedule"] = {"demo_ramp_steps": 100, "pass_rate": 0.5, "ladder": True, "stuck_steps": 100}
    trainer = start_run(config, EdgeEnv(max_episode_steps=40), tmp_path / "run", quiet=True)
    trainer.train()
    episodes = [e for e in read_jsonl(tmp_path / "run" / "episodes.jsonl") if e["start_grid"] >= 1]
    assert episodes and episodes[0]["start_grid"] == 10
    for e in episodes:
        below = trainer.start_curriculum.grid_of(e["start_ratio"])
        assert e["start_grid"] == below + 1 or (e["start_grid"] == 1 and below == 1)
    metrics = read_jsonl(tmp_path / "run" / "metrics.jsonl")
    assert metrics and all(m["schedule_stage"] == "curriculum" for m in metrics)   # the schedule runs alongside


def test_record_demos_can_keep_the_states_of_the_episodes_that_failed(tmp_path, monkeypatch):
    """User 2026-11-13 ("也放进开局池"): ``--starts-from-failed`` keeps the states saved along the
    demonstrations the script gave up on too; without it only the successful ones (the default). The
    switch reaches the child processes that record."""
    import importlib.util
    import subprocess
    from guidewire_rl import WORKSPACE_ROOT
    spec = importlib.util.spec_from_file_location("record_demos", WORKSPACE_ROOT / "scripts" / "record_demos.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    seen = []
    monkeypatch.setattr(rd, "worker", lambda args: seen.append(args))
    out, starts = str(tmp_path / "out"), str(tmp_path / "starts")
    rd.main(["--child", "--seeds", "7", "--out", out, "--starts", starts, "--starts-from-failed"])
    rd.main(["--child", "--seeds", "7", "--out", out, "--starts", starts])
    assert [args[-1] for args in seen] == [True, False]
    commands = []

    class Done:                                    # a child that finished at once (and wrote nothing)
        def __init__(self, cmd, *a, **k):
            commands.append(cmd)

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Done)
    monkeypatch.setattr(rd.time, "sleep", lambda s: None)
    rd.main(["--seeds", "7", "--out", out, "--starts", starts, "--starts-from-failed"])
    assert commands and all("--starts-from-failed" in c and "--child" in c for c in commands)
    import json
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["starts_from_failed"] is True
    with pytest.raises(SystemExit):
        rd.main(["--seeds", "7", "--out", out, "--starts-from-failed"])      # needs --starts


def test_a_restored_episode_can_count_the_saved_steps_towards_the_step_limit():
    """User 2026-11-12 23:27: from a state saved at step n, an episode has the limit minus n steps left
    (env.start FromSnapshot count_saved_steps; GuidewireEnv.reset calls advance_step_limit). Only the
    MaxSteps entries count; off by default."""
    from eve.truncation import Combination, MaxSteps, TruncationDummy
    from guidewire_rl.env.guidewire_env import advance_step_limit
    from guidewire_rl.env.two_device.snapshot import FromSnapshot
    limit = MaxSteps(10)
    truncation = Combination([TruncationDummy(), Combination([limit])])
    truncation.reset()
    assert advance_step_limit(truncation, 7) == [limit]
    for _ in range(2):
        truncation.step()
        assert not truncation.truncated
    truncation.step()
    assert truncation.truncated                                  # 7 saved + 3 own = the limit of 10
    truncation.reset()                                           # the next episode starts from zero
    for _ in range(9):
        truncation.step()
    assert not truncation.truncated
    assert advance_step_limit(TruncationDummy(), 5) == []        # no MaxSteps: nothing to count
    assert FromSnapshot(None, fraction=0.0).count_saved_steps is False
    assert FromSnapshot(None, fraction=0.0, count_saved_steps=True).count_saved_steps is True


def test_the_four_training_methods_are_four_configs():
    """docs/training_framework.md, "四种训练方法": the same task, observation, action and reward; only how
    the policy is helped differs, and each way is one config file."""
    from guidewire_rl import WORKSPACE_ROOT
    from guidewire_rl.config import load_config
    names = ("two_devices_ica_nodemo", "two_devices_ica", "two_devices_ica_selfstart", "two_devices_ica_demostart")
    none, crutches, own, demo = (load_config(WORKSPACE_ROOT / "configs" / f"{n}.yaml") for n in names)
    for other in (crutches, own, demo):
        for part in ("observation", "action", "reward_formula", "terminal", "truncation", "monitors"):
            assert other["env"][part] == none["env"][part], part
    t = none["training"]
    assert not t.get("start_curriculum") and not t.get("demos") and not t.get("schedule")
    assert not t["self_demos"]["enabled"]
    t = crutches["training"]
    assert t["start_curriculum"]["supply"] == "prefix" and t["schedule"]
    t = own["training"]
    assert t["start_curriculum"]["supply"] == "self" and not t.get("demos") and not t.get("schedule")
    assert own["env"]["start"]["type"] == "FromSnapshot"
    t = demo["training"]
    assert t["start_curriculum"]["supply"] == "demos"
    assert t["schedule"] == crutches["training"]["schedule"]       # the existing schedule (user 2026-11-12 23:22)
    assert t["demos"] == {"dir": "runs/demos_round2c", "fraction": 0.25}           # user 2026-11-13 00:57
    assert t["demos"] == crutches["training"]["demos"]   # the same demonstrations and quota (user 2026-11-13)
    assert t["start_curriculum"]["library"] == "runs/demos_round2c/starts"        # made in the same recording
    assert t["self_demos"] == {"enabled": True, "capacity": 500}
    # starts restored the same way, except that a restored episode keeps its saved steps (user 2026-11-12 23:27)
    assert demo["env"]["start"] == {**own["env"]["start"], "count_saved_steps": True}
    assert {k: v for k, v in demo["env"].items() if k != "start"} == {k: v for k, v in own["env"].items() if k != "start"}
    assert not own["env"]["start"].get("count_saved_steps")                       # selfstart: the full 800 steps
    rest = {k: v for k, v in t["start_curriculum"].items() if k not in ("supply", "library")}
    assert rest == {k: v for k, v in own["training"]["start_curriculum"].items() if k != "supply"}


def test_episode_snapshots_save_once_per_new_grid(tmp_path):
    from guidewire_rl.env.two_device.snapshot import EpisodeSnapshots
    taken = []

    def fake_capture(env, **meta):
        taken.append(meta)
        return {"meta": meta}

    import guidewire_rl.env.two_device.snapshot as snapshot_module
    saved_files = []
    original_save = snapshot_module.save
    snapshot_module.save = lambda path, snap: saved_files.append(Path(path))
    try:
        s = EpisodeSnapshots(bins=10, capture_fn=fake_capture)
        s.begin(str(tmp_path), "e3_77")
        reports = [s.after_step(None, {"progress_ratio": r}) for r in (0.0, 0.05, 0.08, 0.12, 0.11, 0.35, 0.36)]
        assert [r["grid"] if r else None for r in reports] == [None, 1, None, 2, None, 4, None]
        assert reports[3]["step"] == 4 and reports[3]["path"].endswith("grid_02/e3_77_s0004.npz")
        assert s.after_step(None, {"progress_ratio": 0.55, "simulation_error": True}) is None
        assert s.after_step(None, {"progress_ratio": 0.95, "is_success": True}) is None   # the end: no start
        s.begin(str(tmp_path), "e3_78", start_ratio=0.42)                # a restored episode starts in grid 5
        assert s.after_step(None, {"progress_ratio": 0.45}) is None
        assert s.after_step(None, {"progress_ratio": 0.51})["grid"] == 6
        s.begin(None, "")                                                # an evaluation: nothing saved
        assert s.after_step(None, {"progress_ratio": 0.99}) is None
    finally:
        snapshot_module.save = original_save
    assert [m["grid"] for m in taken] == [1, 2, 4, 6] and len(saved_files) == 4


def test_reset_command_carries_the_start_and_the_saving():
    assert reset_command(5) == ("reset", 5)
    _, data = reset_command(5, snapshot="/runs/r/starts/grid_06/a.npz", save_dir="/runs/r/starts", save_tag="e2_5")
    assert data == {"seed": 5, "snapshot": "/runs/r/starts/grid_06/a.npz", "save_dir": "/runs/r/starts",
                    "save_tag": "e2_5", "save_bins": 10}
    _, data = reset_command(5, save_dir="/runs/r/starts", save_tag="e2_5")
    assert "snapshot" not in data and data["save_dir"] == "/runs/r/starts"


def test_referee_history_fields_are_exactly_what_measuring_leaves_out():
    """After a restore the referee keeps these from the snapshot and measures every other field."""
    import ast
    import dataclasses
    import inspect
    import textwrap

    from guidewire_rl.env.two_device import phases
    tree = ast.parse(textwrap.dedent(inspect.getsource(phases.PhaseDetector._measure)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "PhaseState"]
    measured = {kw.arg for kw in calls[0].keywords}
    every = {f.name for f in dataclasses.fields(phases.PhaseState)}
    assert set(phases.HISTORY_FIELDS) == every - measured


@pytest.mark.sofa
@pytest.mark.upstream          # the round-two scene needs the new env's stEVE (the Angled wire)
def test_an_episode_starts_from_a_state_saved_along_another_one(tmp_path):
    """Save along an episode (wire pushed out of the catheter), start an episode with another target from
    that state: same beam, the referee measured against the new target, the first step without a jump."""
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env
    from guidewire_rl.env.two_device.reward_formula import INDEX
    from guidewire_rl.env.two_device.snapshot import EpisodeSnapshots
    from guidewire_rl import WORKSPACE_ROOT
    cfg = load_config(WORKSPACE_ROOT / "configs" / "two_devices_ica_selfstart.yaml")
    env = build_env(cfg["env"])
    try:
        env.reset(seed=1000)
        saver = EpisodeSnapshots(bins=10)
        saver.begin(str(tmp_path), "t")
        saved = []
        for _ in range(80):                              # both forward, the wire faster: it leaves the catheter
            _, _, te, tr, info = env.step(np.array([1.0, 0.0, 0.6, 0.0], np.float32))
            s = saver.after_step(env, info)
            if s:
                saved.append((s, np.asarray(env.simulation.dof_positions, float)[0, :3].copy(), info["d"]))
            if te or tr:
                break
        assert saved, "the run reached no new grid"
        s, tip, lead = saved[-1]
        assert lead > 0                                  # the wire was out of the catheter
        obs, info = env.reset(seed=1777, options={"snapshot": s["path"]})
        assert info["snapshot_start"]["path"] == s["path"] and info["snapshot_start"]["grid"] == s["grid"]
        restored_tip = np.asarray(env.simulation.dof_positions, float)[0, :3]
        assert np.linalg.norm(restored_tip - tip) < 1.0  # the restore's one substep moves it a little
        assert info["progress_ratio"] == pytest.approx(info["snapshot_start"]["progress"])
        assert abs(info["progress_ratio"] - s["progress"]) < 0.05   # another target: a few mm of path
        assert info["snapshot_start"]["counted_steps"] == 0           # selfstart: the full step limit
        _, _, _, _, info = env.step(np.zeros(4, np.float32))
        assert abs(info["reward_materials"][INDEX["progress_mm"]]) < 2.0   # no jump on the first step
        # count_saved_steps (user 2026-11-12 23:27): the restored episode keeps the saved state's steps
        from guidewire_rl.env.guidewire_env import advance_step_limit
        env._env.start.count_saved_steps = True
        obs, info = env.reset(seed=1777, options={"snapshot": s["path"]})
        assert info["snapshot_start"]["counted_steps"] == s["step"] == info["snapshot_start"]["saved_step"] > 0
        (limit,) = advance_step_limit(env._env.truncation, 0)
        limit.max_steps = s["step"] + 2                  # so that two steps are left
        assert not env.step(np.zeros(4, np.float32))[3]
        assert env.step(np.zeros(4, np.float32))[3]      # truncated after the saved steps + 2 own
        limit.max_steps, env._env.start.count_saved_steps = 800, False
        obs, info = env.reset(seed=1000)                 # an ordinary reset afterwards: the opening
        assert "snapshot_start" not in info and info["progress_ratio"] < 0.1
    finally:
        env.close()
