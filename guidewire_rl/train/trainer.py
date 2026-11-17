"""Online training loop.

Same schedule as the official TD-MPC2 ``OnlineTrainer``:

1. the first ``seed_steps`` env steps use uniform random actions;
2. when ``seed_steps`` is reached, ``initial_updates`` updates run in one burst;
3. after that, ``updates_per_step`` updates per env step;
4. an episode enters the replay only when it ends;
5. evaluation is due every ``evaluation.interval`` steps and runs at the next
   episode boundary.

Additions for SOFA: a step that raises or flags ``simulation_error`` is
dropped, the episode so far is kept (as truncated: no terminal flag, so its last
value is bootstrapped) and counted (``end: error`` in ``episodes.jsonl``,
``error_rate`` in ``metrics.jsonl``), and a new one starts; a step that flags
``wall_breached`` (a device's centerline outside the vessel mesh, two_device/wall.py)
is treated the same way under its own name (``end: breach``, ``breach_rate``,
``wall_breaches``: the simulator's failure, not the policy's, so no -1; user
decision 2026-11-09); checkpoints hold the full state so a crashed run can resume.

Optional parts, each off unless configured; with all of them off a run is bit for
bit what this loop gave before they existed (tests/test_two_device_training.py):

* ``training.cutoff_mixture``: each episode's cutoff, set with
  ``env.set_cutoff(k)`` before its reset (``curriculum.CutoffMixture``);
* ``planning.residual``: the planner's box around the scripted suggestion in
  phase 1 (``curriculum.ResidualSchedule``);
* ``replay.stratify_by`` and ``training.demos``: see ``replay.py`` / ``demos.py``
  (set up in ``setup.start_run``);
* ``training.start_curriculum``: where each episode starts (``curriculum.StartCurriculum``,
  grid ``distance`` or ``landmarks``);
* ``training.schedule``: stages, guard, stop rule and ladder, moved at every evaluation
  (``schedule.Schedule``); ``evaluation.keep_best``: ``checkpoints/best.pt`` at every new best
  success rate;
* ``training.pretrain_updates`` (default 0): that many ordinary updates on the replay (the
  demonstrations) before the first env step of a fresh run (``_pretrain``; all three loops). The
  agent then counts as trained, so the ``initial_updates`` burst is skipped; with
  ``seed_steps: 0`` the policy acts from the first step.

Logs (``RunLogger``): ``episodes.jsonl`` one line per episode, with the task
metrics the env reports (``metrics.EpisodeStats``); ``metrics.jsonl`` one line
per ``logging.print_interval`` steps, averaged over the last ``METRICS_WINDOW``
episodes (reach rates per phase, success, errors, shield, catheter, step rate,
cutoff weights); ``events.jsonl`` checkpoints, dropped steps, worker
replacements, weight updates.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .checkpoint import capture_rng, restore_rng, save_checkpoint
from .curriculum import CutoffMixture, ResidualSchedule, StartCurriculum
from .evaluate import evaluate
from .logger import MetricAverager, RunLogger
from .metrics import EpisodeStats, progress_line, window_summary
from .replay import SOURCE_POLICY, SOURCE_PREFIX, EpisodeReplay
from .schedule import Schedule
from .start_library import StartLibrary
from .wandb_sink import WandbSink

METRICS_WINDOW = 100           # episodes the console's reach rates and metrics.jsonl average over
CUTOFF_SEED_OFFSET = 7_919     # the cutoff draws' own random stream: training.seed + this
START_SEED_OFFSET = 104_729    # the start curriculum's own random stream: training.seed + this


def crossed(before: int, after: int, every: int) -> bool:
    """Whether a multiple of ``every`` lies in (before, after]: steps may arrive in bursts (a
    scripted prefix of hundreds of steps lands with one reset), so checkpoints, evaluation and
    console lines are due when a multiple was passed, not only when the count lands on it."""
    return after // every > before // every


def step_failure(info: Mapping[str, Any]) -> Optional[str]:
    """Why a step is dropped and its episode closed: ``simulation_error`` (NaN: stEVE put the
    instruments back to the start) or ``wall_breach`` (``info["wall_breached"]``: a device's
    centerline outside the vessel mesh); None for a step to keep."""
    if info.get("simulation_error"):
        return "simulation_error"
    if info.get("wall_breached"):
        return "wall_breach"
    return None


class _Episode:
    def __init__(self, observation: np.ndarray, label_names: List[str],
                 info: Optional[Mapping[str, Any]] = None, cutoff: Optional[int] = None,
                 start_grid: int = 0, start_ratio: Optional[float] = None) -> None:
        self.observations = [observation]
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminated: List[float] = []
        self.labels: Dict[str, List[float]] = {n: [] for n in label_names}
        self.materials: List[np.ndarray] = []        # the reward's raw materials per step (round two)
        self.sources: List[int] = []                 # SOURCE_* per step
        self.experts: List[np.ndarray] = []          # the shadow expert's target per step (NaN: none)
        self.start_grid = int(start_grid)            # the start curriculum's grid (0: insertion point)
        self.start_ratio = start_ratio               # progress ratio handed over at (None: insertion)
        self.prefix_steps = 0
        self.prefix_outcome = ""                     # how the scripted prefix ended (prefix.py)
        self.scripted = False                        # the script played it to the end: a demonstration
        self.stats = EpisodeStats(info, cutoff)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def last_info(self) -> Mapping[str, Any]:
        return self.stats.last_info

    def add(self, action, reward, terminated, labels, next_observation,
            info: Optional[Mapping[str, Any]] = None, source: int = SOURCE_POLICY) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.terminated.append(float(terminated))
        for name in self.labels:
            self.labels[name].append(float(labels[name]))
        self.observations.append(next_observation)
        info = info or {}
        if info.get("reward_materials") is not None:
            self.materials.append(np.asarray(info["reward_materials"], dtype=np.float32))
        self.sources.append(int(source))
        expert = info.get("expert_action")
        self.experts.append(np.full(self.actions[-1].shape, np.nan, np.float32) if expert is None
                            else np.asarray(expert, dtype=np.float32).reshape(self.actions[-1].shape))
        self.stats.add(self.actions[-1], info)

    def materials_array(self) -> Optional[np.ndarray]:
        if not self.materials:
            return None
        if len(self.materials) != self.length:
            raise ValueError("reward materials missing on some steps of the episode")
        return np.stack(self.materials)


class Trainer:
    SUPPORTS_BACKGROUND_EVAL = False

    def __init__(self, config: Mapping[str, Any], env, agent, replay: EpisodeReplay,
                 run_dir: str | Path, *, resume_state: Optional[Mapping[str, Any]] = None,
                 quiet: bool = False) -> None:
        self.cfg = config
        self.env = env
        self.agent = agent
        self.replay = replay
        self.run_dir = Path(run_dir)
        self.logger = RunLogger(self.run_dir, quiet=quiet,
                                sink=WandbSink.from_config(config, self.run_dir, resume=resume_state is not None))
        self.label_names = list(env.label_names)
        # the reward broken into its terms per episode (docs/wandb_metrics.md, section 3), for configs
        # whose reward is the formula
        self.reward_parts = None
        formula = (config.get("env") or {}).get("reward_formula")
        if formula:
            from ..env.two_device import reward_formula as rf
            coefficients = rf.Coefficients.from_config(formula)
            self.reward_parts = lambda materials: rf.component_sums(materials, coefficients)
        self.rng = np.random.default_rng(int(config["training"]["seed"]))
        self.step = 0
        self.episode_index = 0
        self.success_count = 0
        self.env_errors = 0
        self.wall_breaches = 0
        self.updates = MetricAverager()
        self.recent = deque(maxlen=50)
        self.window = deque(maxlen=METRICS_WINDOW)
        self.last_update: Dict[str, float] = {}
        self.mixture = CutoffMixture.from_config(config["training"].get("cutoff_mixture"),
                                                 int(config["training"]["seed"]) + CUTOFF_SEED_OFFSET)
        self.residual = ResidualSchedule.from_config(config.get("planning", {}).get("residual"))
        self.start_curriculum = StartCurriculum.from_config(
            config["training"].get("start_curriculum"), int(config["training"]["seed"]) + START_SEED_OFFSET)
        # the start library (start_library.py): supply 'self' = the policy's own, read back from <run>/starts;
        # supply 'demos' = the fixed one made from the demonstrations (nothing saved during training)
        self.start_library = StartLibrary.from_config(
            config["training"].get("start_curriculum"), self.run_dir, int(config["training"]["seed"]) + START_SEED_OFFSET + 1)
        self._saver = None                               # single env: the episode's along-the-way snapshots
        if self.mixture is not None and self.start_curriculum is not None:
            raise ValueError("training.cutoff_mixture and training.start_curriculum exclude each other")
        # the schedule: stages, guard, stop rule, ladder (schedule.py); None without training.schedule
        self.schedule = Schedule.from_config(config["training"].get("schedule"), replay.demo_fraction)
        self.keep_best = bool(config["evaluation"].get("keep_best", False))
        # evaluation.background (2026-11-15): the evaluation runs beside training instead of pausing it;
        # only the trainer whose envs do not wait for each other can do that (parallel_trainer.AsyncTrainer)
        self.background_eval = bool(config["evaluation"].get("background", False))
        # evaluation.end_when_stalled: evaluation episodes end once the devices stand still (two_device Stalled)
        self.eval_end_when_stalled = bool(config["evaluation"].get("end_when_stalled", False))
        if self.background_eval and not self.SUPPORTS_BACKGROUND_EVAL:
            raise ValueError("evaluation.background needs training.num_envs > 1 with training.async_envs: true")
        self.best_rate: Optional[float] = None
        self._coefficients_key: Optional[tuple] = None
        # the policy's own successful episodes are kept as self-demonstrations (never evicted, up to
        # replay.self_demo_capacity) when the config says so (``training.self_demos``)
        self.self_demos = bool((config["training"].get("self_demos") or {}).get("enabled", False))
        if self.mixture is not None and not self._env_supports_cutoff():
            raise ValueError("training.cutoff_mixture needs an env with set_cutoff(k) "
                             "(a truncation that cuts the episode at a phase)")
        self._last_print = (time.time(), 0)
        if resume_state is not None:
            self.step = int(resume_state["total_env_steps"])
            self.episode_index = int(resume_state.get("episode_index", 0))
            self.success_count = int(resume_state.get("success_count", 0))
            self.env_errors = int(resume_state.get("env_errors", 0))
            self.wall_breaches = int(resume_state.get("wall_breaches", 0))
            if "rng_state" in resume_state:
                restore_rng(resume_state["rng_state"], self.rng)
            if self.mixture is not None and resume_state.get("cutoff_mixture"):
                self.mixture.load_state_dict(resume_state["cutoff_mixture"])
            if self.residual is not None and resume_state.get("residual"):
                self.residual.load_state_dict(resume_state["residual"])
            if self.start_curriculum is not None and resume_state.get("start_curriculum"):
                self.start_curriculum.load_state_dict(resume_state["start_curriculum"])
            if self.schedule is not None and resume_state.get("schedule"):
                self.schedule.load_state_dict(resume_state["schedule"])
            self.best_rate = resume_state.get("best_rate")
        if self.schedule is not None:
            self._apply_schedule()         # the schedule's values from the first batch on (the force factor)

    # ------------------------------------------------------------ helpers --
    def _env_supports_cutoff(self) -> bool:
        return callable(getattr(self.env, "set_cutoff", None))

    @property
    def eval_cutoff(self) -> Optional[int]:
        """Evaluation episodes are never cut (only when the run sets cutoffs at all)."""
        return None if self.mixture is None else self.mixture.full_episode_cutoff

    def _draw_seed_and_cutoff(self) -> tuple[int, Optional[int]]:
        seed = int(self.rng.integers(0, 2**31 - 1))
        return seed, (None if self.mixture is None else self.mixture.sample())

    def _draw_start(self) -> tuple[int, Optional[float]]:
        """(grid, prefix ratio) of the next episode from the start curriculum; (0, None) without.
        With the start library the ratio is None and ``_draw_snapshot`` gives the state."""
        if self.start_curriculum is None:
            return 0, None
        if self.start_library is not None:
            self.start_curriculum.set_available(self.start_library.available())
        return self.start_curriculum.sample()

    def _draw_snapshot(self, grid: int) -> Optional[str]:
        """The saved state an episode drawn at ``grid`` starts from (start library); None: insertion point."""
        if self.start_library is None or grid <= 0:
            return None
        return self.start_library.sample(grid)

    def _book_snapshot(self, saved: Optional[Mapping[str, Any]], **where: Any) -> None:
        """A state an env saved along its episode goes into the start library."""
        if not saved or self.start_library is None:
            return
        if "error" in saved:
            self._event("snapshot_failed", reason=str(saved["error"])[:300], **where)
            return
        self.start_library.add(saved)

    def _new_episode(self) -> tuple[Optional[np.ndarray], Optional[_Episode]]:
        """One reset (and, with the start curriculum, its scripted prefix). ``(None, None)`` when
        the prefix itself ended or broke the episode: it is booked, the caller resets again (so the
        step count, and with it the run's end, evaluation and checkpoints, stay in charge)."""
        seed, cutoff = self._draw_seed_and_cutoff()
        grid, ratio = self._draw_start()
        snapshot = self._draw_snapshot(grid)
        if cutoff is not None:
            self.env.set_cutoff(cutoff)
        observation, info = self.env.reset(seed=seed, **({"options": {"snapshot": snapshot}} if snapshot else {}))
        info = dict(info)
        if self.start_library is not None and self.start_library.saving_to is not None:
            from ..env.two_device.snapshot import EpisodeSnapshots
            if self._saver is None:
                self._saver = EpisodeSnapshots(bins=self.start_library.bins)
            self._saver.begin(self.start_library.saving_to, f"e0_{seed}",
                              float(info.get("progress_ratio", 0.0) or 0.0) if snapshot else 0.0)
        if ratio is not None:
            from ..env.two_device.prefix import run_prefix
            prefix = run_prefix(self.env, ratio, max_steps=self.start_curriculum.prefix_max_steps,
                                first_info=info, key=self.start_curriculum.prefix_key)
            prefix["first_observation"], prefix["first_info"] = observation, info
            if prefix["observation"] is not None:
                observation, info = prefix["observation"], dict(prefix["info"])
            info["prefix"] = prefix
        episode = self._absorb_reset(observation, info, cutoff, grid, ratio)
        if episode is None:
            self._curriculum_tick()
            return None, None
        if episode.prefix_steps:
            self._curriculum_tick()
        return observation, episode

    def _absorb_reset(self, observation: np.ndarray, info: Mapping[str, Any], cutoff: Optional[int],
                      grid: int = 0, ratio: Optional[float] = None, **where: Any) -> Optional[_Episode]:
        """The episode a reset started. With a scripted prefix (``info["prefix"]``) its steps are
        booked first (source ``prefix``, counted as env steps); the grid the episode is booked under
        is the one the prefix really reached. None: the prefix ended or broke the episode (closed
        here) and the env needs another reset."""
        self._check_cutoff(info, cutoff, **where)
        if self.start_curriculum is not None:
            self.start_curriculum.check_grids(info)
        prefix = info.get("prefix")
        if not prefix:
            started = info.get("snapshot_start")
            if started and self.start_curriculum is not None:
                # booked under the grid drawn (= the grid the state was saved in, its library folder), not
                # the grid of the progress re-measured against this episode's target: a state saved on
                # entering grid k sits on its lower edge, and re-measuring moves it by about 0.01, so half
                # of them would land in grid k - 1 (user 2026-11-13, work log 2026-11-12 21:45). The
                # start ratio is still the re-measured progress.
                ratio = float(started.get("progress") or 0.0)
                if grid <= 0:
                    saved = started.get("grid")
                    grid = int(saved) if saved else self.start_curriculum.grid_of(ratio)
            return _Episode(observation, self.label_names, info, cutoff, start_grid=grid, start_ratio=ratio)
        reached = float(prefix.get("ratio", 0.0))
        start_grid = self.start_curriculum.grid_of(reached) if self.start_curriculum is not None else grid
        episode = _Episode(prefix["first_observation"], self.label_names, prefix["first_info"], cutoff,
                           start_grid=start_grid, start_ratio=reached)
        for tr in prefix["transitions"]:
            episode.add(tr["action"], tr["reward"], tr["terminated"], tr["info"]["labels"],
                        tr["observation"], tr["info"], source=SOURCE_PREFIX)
        episode.prefix_steps = len(prefix["transitions"])
        episode.prefix_outcome = str(prefix["outcome"])
        self.step += episode.prefix_steps
        self._event("prefix", **where, outcome=prefix["outcome"], steps=episode.prefix_steps,
                    ratio=reached, ratio_asked=prefix.get("ratio_asked"), grid=start_grid,
                    script_stage=prefix.get("script_stage"), script_failure=prefix.get("script_failure"))
        outcome = prefix["outcome"]
        if outcome in ("simulation_error", "wall_breach"):
            self._drop_step(episode, outcome, **where)
            return None
        if outcome == "ended":
            # the script played the whole episode: a scripted demonstration, not the policy's episode
            # (no success of the policy's, nothing for the curriculum's statistics)
            episode.scripted = True
            last = prefix["transitions"][-1]
            self._close_episode(episode, reason="terminated" if last["terminated"] else "truncated",
                                success=bool(last["info"].get("is_success", False)))
            return None
        return episode

    def _check_cutoff(self, info: Mapping[str, Any], cutoff: Optional[int], **where: Any) -> None:
        """An env that reports its cutoff must have the one the trainer drew."""
        if cutoff is not None and info.get("cutoff") is not None and int(info["cutoff"]) != cutoff:
            self._event("cutoff_mismatch", drawn=cutoff, env_reports=int(info["cutoff"]), **where)

    def _act(self, episode: _Episode, observation: np.ndarray) -> np.ndarray:
        if self.residual is None:
            return self.agent.act(observation, first_step=episode.length == 0, eval_mode=False)
        return self.agent.act(observation, first_step=episode.length == 0, eval_mode=False,
                              bounds=self.residual.box_for(episode.last_info))

    def _drop_step(self, episode: _Episode, failure: str, **where: Any) -> None:
        """A step the env flagged (``step_failure``) or that raised: not stored; the episode so far
        is closed as ``error`` (a simulation error or an exception) or ``breach`` (``wall_breach``)."""
        if failure == "wall_breach":
            self.wall_breaches += 1
        else:
            self.env_errors += 1
        self._event("dropped_step", **where, reason=failure, episode_length=episode.length,
                    **episode.stats.where())
        self._close_episode(episode, reason="breach" if failure == "wall_breach" else "error")

    def _close_episode(self, episode: _Episode, *, reason: str, success: bool = False) -> None:
        if episode.length == 0:
            return
        stats = episode.stats
        protected = (True if episode.scripted else bool(success)) if self.self_demos else None
        self.replay.add(np.stack(episode.observations), np.stack(episode.actions),
                        np.asarray(episode.rewards), np.asarray(episode.terminated),
                        {n: np.asarray(v) for n, v in episode.labels.items()},
                        phase=np.asarray(stats.phases), materials=episode.materials_array(),
                        source=np.asarray(episode.sources, dtype=np.int8), protected=protected,
                        demo=episode.scripted, outcome=reason, expert=np.stack(episode.experts))
        self.episode_index += 1
        self.success_count += int(success and not episode.scripted)
        record = {"step": self.step, "episode": self.episode_index, "length": episode.length,
                  "return": float(np.sum(episode.rewards)), "success": bool(success),
                  "end": reason}
        if self.start_curriculum is not None or episode.prefix_steps:
            record.update(start_grid=episode.start_grid, start_ratio=episode.start_ratio,
                          prefix_steps=episode.prefix_steps, prefix_outcome=episode.prefix_outcome,
                          scripted=episode.scripted)
        materials = episode.materials_array()
        if self.reward_parts is not None and materials is not None:
            record["reward_parts"] = self.reward_parts(materials)
        for name, values in episode.labels.items():
            record[f"{name}_max"] = float(np.max(values))
            record[f"{name}_mean"] = float(np.mean(values))
        record.update(stats.fields())
        self.logger.write("episodes", record)
        self.recent.append(record)
        self.window.append(record)
        if self.mixture is not None:
            self.mixture.record(stats.cutoff, stats.max_phase, success)
        if self.start_curriculum is not None and not episode.scripted:
            self.start_curriculum.record(episode.start_grid, success)
        if self.residual is not None:
            check = self.residual.record(stats.max_phase, success)
            if check is not None:
                self._event("residual_check", **check)

    def _curriculum_tick(self) -> None:
        if self.mixture is not None:
            change = self.mixture.maybe_update(self.step)
            if change is not None:
                self._event("cutoff_weights", **change)
                self.logger.print("cutoff weights " + " ".join(
                    f"{k}:{w:.2f}" for k, w in change["weights"].items())
                    + " (reach " + " ".join(f"{k}:{r:.2f}" for k, r in change["reach"].items()) + ")")
        if self.start_curriculum is not None:
            move = self.start_curriculum.maybe_update(self.step)
            if move is not None:
                self._event("start_frontier", **move)
                self.logger.print(f"start curriculum: frontier {move['frontier_before']} -> "
                                  f"{move['frontier']} (success {move['rate']:.2f} over {move['episodes']})")

    def _event(self, kind: str, **fields: Any) -> None:
        self.logger.write("events", {"step": self.step, "event": kind, **fields})

    def _pretrain(self) -> None:
        """``training.pretrain_updates`` N > 0: N ordinary agent updates on the replay before the first
        env step of a fresh run (nothing stepped, no update yet; a resumed run skips it). The replay
        must be samplable then (demonstrations loaded), else this fails."""
        count = int(self.cfg["training"].get("pretrain_updates", 0) or 0)
        if count <= 0 or self.step > 0 or self.agent.update_count > 0:
            return
        if not self.replay.can_sample():
            raise RuntimeError(f"training.pretrain_updates={count} needs a replay to learn from before the first "
                               f"env step, but it holds {self.replay.size} transitions in {self.replay.num_episodes} "
                               "episode(s) and none is long enough to sample (training.demos loaded?)")
        self.logger.print(f"pretraining on the replay before the first env step: {count} updates "
                          f"({self.replay.size} transitions, {self.replay.num_demo_episodes} demonstrations)")
        started = time.time()
        for _ in range(count):
            self._record_update(self.agent.update(self.replay))
        self._event("pretrain", updates=count, replay_transitions=int(self.replay.size),
                    demo_episodes=int(self.replay.num_demo_episodes), minutes=(time.time() - started) / 60.0)

    def _record_update(self, metrics: Mapping[str, float]) -> None:
        self.updates.add(metrics)
        self.last_update = dict(metrics)
        if self.updates.count >= int(self.cfg["logging"]["update_interval"]):
            self.logger.write("updates", {"step": self.step,
                                          "update": self.agent.update_count,
                                          **self.updates.pop()})

    def _run_evaluation(self, seeds: List[int]) -> Dict[str, Any]:
        return evaluate(self.env, self.agent, seeds, cutoff=self.eval_cutoff,
                        residual=self.residual, components=self.reward_parts,
                        end_when_stalled=self.eval_end_when_stalled)

    def _evaluate(self) -> Dict[str, Any]:
        ev = self.cfg["evaluation"]
        seeds = [int(ev["seed"]) + i for i in range(int(ev["episodes"]))]
        started = time.time()
        result = self._run_evaluation(seeds)
        self.logger.write("eval", {"step": self.step, **result["summary"]})
        # wall-clock time goes to the events, not eval.jsonl: that file stays reproducible bit for bit
        self._event("evaluation", minutes=(time.time() - started) / 60.0)
        s = result["summary"]
        self.logger.print(f"eval @ {self.step}: success {s['success_rate']:.2f}, "
                          f"return {s['return_mean']:.3f}, length {s['length_mean']:.1f}")
        self._after_evaluation(float(s["success_rate"]))
        return result

    def _after_evaluation(self, rate: float) -> None:
        """The best checkpoint (``evaluation.keep_best``) and the schedule's moves."""
        if self.keep_best and (self.best_rate is None or rate > self.best_rate):
            self.best_rate = rate
            path = save_checkpoint(self.run_dir / "checkpoints" / "best.pt", self.checkpoint_payload())
            self._event("best_checkpoint", path=str(path), success_rate=rate)
        if self.schedule is None:
            return
        for change in self.schedule.on_evaluation(self.step, rate, self.start_curriculum):
            self._event("schedule", **change)
        self._apply_schedule()

    def _apply_schedule(self) -> None:
        """Put the schedule's knobs into the replay: the demonstration quota and the reward's
        coefficients (the hint rewards' factor, the arch ruler of the ladder's level 1, the force
        penalty's factor). Only the replay's rewards change (what the agent learns from); the env's own
        reward, the episode logs and the evaluations keep the config's coefficients."""
        s = self.schedule
        if abs(self.replay.demo_fraction - s.demo_value) > 1e-12:
            self.replay.set_demo_fraction(s.demo_value)
        if s.demo_fraction > 0:                           # behaviour cloning fades with the quota
            self.agent.bc_scale = float(s.demo_value) / float(s.demo_fraction)
        spec = self.cfg["env"].get("reward_formula")
        key = (round(s.hint_scale, 12), bool(s.arch_hint), round(s.force_scale, 12))
        if not spec or self.replay.reward_fn is None or key == self._coefficients_key:
            return
        if self._coefficients_key is None and key == (1.0, False, 1.0):
            self._coefficients_key = key                  # the config's own coefficients: nothing to do
            return
        from dataclasses import replace as _replace
        from ..env.two_device import reward_formula as rf
        base = rf.Coefficients.from_config(spec)
        coefficients = _replace(base, progress=base.progress * s.hint_scale, follow=base.follow * s.hint_scale,
                                arch_ruler=base.arch_ruler or s.arch_hint,
                                force_cap=base.force_cap * s.force_scale)
        self.replay.reward_fn = lambda materials, c=coefficients: rf.compute(materials, c)
        self.replay.recompute_rewards()
        self._coefficients_key = key

    @property
    def stopping(self) -> bool:
        """The schedule's stop rule fired (the run ends at the next chance)."""
        return self.schedule is not None and self.schedule.stop

    def _metrics_extra(self) -> Dict[str, Any]:
        """More fields for metrics.jsonl (the parallel trainers add theirs)."""
        return {}

    def _print_progress(self, started: float, start_step: int) -> None:
        now = time.time()
        rate = (self.step - start_step) / max(now - started, 1e-9)
        recent = list(self.recent)
        success = np.mean([r["success"] for r in recent]) if recent else float("nan")
        ret = np.mean([r["return"] for r in recent]) if recent else float("nan")
        loss = self.last_update.get("total_loss", float("nan"))
        self.logger.print(f"step {self.step} | episodes {self.episode_index} | "
                          f"last-{len(recent)} success {success:.2f} return {ret:.3f} | "
                          f"loss {loss:.4f} | {rate:.1f} steps/s")
        summary = window_summary(list(self.window))
        line = progress_line(summary)
        if line is not None or self.mixture is not None:
            weights = ("" if self.mixture is None else " | cutoff weights " + " ".join(
                f"{k}:{w:.2f}" for k, w in self.mixture.weight_map().items()))
            self.logger.print((line or " ") + weights)
        last_time, last_step = self._last_print
        record = {"step": self.step, "episodes": self.episode_index,
                  "updates": self.agent.update_count, "env_errors": self.env_errors,
                  "wall_breaches": self.wall_breaches, "steps_per_s": rate,
                  "steps_per_s_recent": (self.step - last_step) / max(now - last_time, 1e-9),
                  **summary}
        if self.mixture is not None:
            record["cutoff_weights"] = self.mixture.weight_map()
        if self.start_curriculum is not None:
            record["start_frontier"] = self.start_curriculum.frontier
            record["start_weights"] = self.start_curriculum.weight_map()
            record["start_rates"] = self.start_curriculum.rates()
            if self.start_library is not None:
                record["start_training_grid"] = self.start_curriculum.training_grid
                record["start_library"] = self.start_library.counts()
        if self.residual is not None:
            record["residual_delta"] = self.residual.delta
        record["replay_size"] = int(self.replay.size)
        if self.schedule is not None:
            record["schedule_stage"] = self.schedule.stage
            record["schedule_stage_index"] = self.schedule.stage_index
            record["guard_trips"] = self.schedule.guard_trips
            record["demo_fraction"] = self.replay.demo_fraction
            record["hint_scale"] = self.schedule.hint_scale
            record["force_scale"] = self.schedule.force_scale
            record["ladder_level"] = self.schedule.ladder_level
        if self.replay.num_demo_episodes:
            record["demo_episodes"] = self.replay.num_demo_episodes
        if self.replay.num_self_demo_episodes:
            record["self_demo_episodes"] = self.replay.num_self_demo_episodes
        if self.replay.stratify_by is not None:
            record["stratum_starts"] = self.replay.stratum_sizes()
        record.update(self._metrics_extra())
        self.logger.write("metrics", record)
        self._last_print = (now, self.step)

    def checkpoint_payload(self) -> Dict[str, Any]:
        payload = {
            "algorithm": "TD-MPC2", "config": dict(self.cfg),
            "agent": self.agent.state_dict(), "total_env_steps": self.step,
            "episode_index": self.episode_index, "success_count": self.success_count,
            "env_errors": self.env_errors, "wall_breaches": self.wall_breaches,
            "rng_state": capture_rng(self.rng),
            "observation_dim": int(self.env.observation_space.shape[0]),
            "observation_names": list(getattr(self.env, "observation_names", [])),
            "label_names": self.label_names,
        }
        if self.mixture is not None:
            payload["cutoff_mixture"] = self.mixture.state_dict()
        if self.start_curriculum is not None:
            payload["start_curriculum"] = self.start_curriculum.state_dict()
        if self.residual is not None:
            payload["residual"] = self.residual.state_dict()
        if self.schedule is not None:
            payload["schedule"] = self.schedule.state_dict()
        if self.best_rate is not None:
            payload["best_rate"] = self.best_rate
        if self.cfg["checkpoint"].get("save_replay", True):
            payload["replay"] = self.replay.state_dict()
        return payload

    def save(self) -> Path:
        path = save_checkpoint(self.run_dir / "checkpoints" / f"step_{self.step}.pt",
                               self.checkpoint_payload())
        self._event("checkpoint", path=str(path))
        return path

    # --------------------------------------------------------------- loop --
    def train(self) -> Dict[str, Any]:
        t = self.cfg["training"]
        total_steps = int(t["total_steps"])
        seed_steps = int(t["seed_steps"])
        initial_updates = int(t["initial_updates"])
        updates_per_step = int(t["updates_per_step"])
        checkpoint_every = int(self.cfg["checkpoint"]["interval"])
        eval_every = int(self.cfg["evaluation"]["interval"])
        print_every = int(self.cfg["logging"]["print_interval"])
        action_shape = self.env.action_space.shape

        self._pretrain()
        pretrained = self.agent.update_count > 0
        eval_due = False
        started, start_step = time.time(), self.step
        self._last_print = (started, self.step)
        self.logger.print(f"training from step {self.step} to {total_steps}; "
                          f"replay has {self.replay.size} transitions")
        observation, episode = self._new_episode()
        while self.step < total_steps and not self.stopping:
            before = self.step
            if episode is None:                   # the prefix closed the episode: reset again
                observation, episode = self._new_episode()
                if crossed(before, self.step, checkpoint_every):
                    self.save()
                if crossed(before, self.step, print_every):
                    self._print_progress(started, start_step)
                continue
            if self.step < seed_steps:
                action = self.rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
            else:
                action = self._act(episode, observation)
            try:
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                failure = step_failure(info)
            except Exception as exc:  # SOFA or label failure inside the step
                failure, info = f"env_error: {exc!r}", {}
            if failure is not None:
                self._drop_step(episode, failure)
                observation, episode = self._new_episode()
                continue

            episode.add(action, reward, terminated, info["labels"], next_observation, info)
            if self._saver is not None and self._saver.directory is not None:
                self._book_snapshot(self._saver.after_step(self.env, info))
            observation = next_observation
            self.step += 1
            self._curriculum_tick()

            if self.step >= seed_steps and self.replay.can_sample():
                count = updates_per_step if pretrained else initial_updates
                if not pretrained:
                    self.logger.print(f"pretraining on seed data: {count} updates")
                    pretrained = True
                for _ in range(count):
                    self._record_update(self.agent.update(self.replay))

            if crossed(before, self.step, eval_every):
                eval_due = True
            if terminated or truncated:
                self._close_episode(episode, reason="terminated" if terminated else "truncated",
                                    success=bool(info.get("is_success", False)))
                if eval_due:
                    self._evaluate()
                    eval_due = False
                observation, episode = self._new_episode()
            if crossed(before, self.step, checkpoint_every):
                self.save()
            if crossed(before, self.step, print_every):
                self._print_progress(started, start_step)

        final = self.save() if self.cfg["checkpoint"].get("save_final", True) else None
        summary = {"total_env_steps": self.step, "episodes": self.episode_index,
                   "success_count": self.success_count, "env_errors": self.env_errors,
                   "wall_breaches": self.wall_breaches,
                   "updates": self.agent.update_count,
                   "final_checkpoint": str(final) if final else None}
        self._event("finished", **summary)
        self.logger.print(f"finished: {summary}")
        self.logger.close()
        return summary
