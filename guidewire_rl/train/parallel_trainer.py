"""Online training loops with N envs stepping at once (``training.num_envs``).

Both loops keep ``Trainer``'s schedule and bookkeeping, counted in env steps
(all envs together): random actions for the first ``seed_steps`` steps, one
burst of ``initial_updates`` once they are in, then ``updates_per_step`` updates
per env step; an episode enters the replay when it ends; checkpoints,
evaluation and console lines at their step multiples; before a checkpoint, an
evaluation and the end every owed update is run, so the update count at step S
is the same as with one env. Evaluation runs right at its step on a separate
pool of envs, so training episodes in flight are not cut; that pool is started
for each evaluation and closed after it (kept open, its scenes held ~0.7 GB each
between evaluations). Shared by both:

* actions for several envs come from one batched planning call, each env with
  its own planner warm start; the scenes step in their own processes (``EnvPool``);
* updates run on the GPU while the scenes step, so the policy an env plans with
  lags behind the one-env schedule by the updates still owed at that moment.

``ParallelTrainer`` (``training.async_envs: false``) steps in rounds: every live
env steps, the round's owed updates run meanwhile, then all answers are
collected. A round never steps past the next print / checkpoint / evaluation /
seed / total step, so those land exactly; the lag is at most one round (N env
steps); the same configuration reproduces bit for bit. Each round waits for the
slowest scene.

``AsyncTrainer`` (``training.async_envs: true``) drops the round barrier: an env
gets its next action as soon as its step is back; updates run one at a time
between polls. Steps are counted as their answers arrive, so checkpoints,
evaluation and console lines still land on exact multiples; the actions in
flight when ``seed_steps`` is reached were chosen as random actions (up to N
random steps beyond ``seed_steps``), and the steps in flight at ``total_steps``
are dropped. The lag varies per step, from a few updates to however long that
env's step took. When updates fall behind by more than ``MAX_OWED_ROUNDS`` rounds
of steps, no new steps are dispatched until they catch up, so updates per step
stay at the configured ratio. Which env answers first depends on timing, so the
same configuration does not reproduce bit for bit.

Both pass each episode's cutoff to its worker inside the reset command
(``training.cutoff_mixture``), plan with each env's own residual box
(``planning.residual``), and write the pool's worker replacements to
``events.jsonl`` (``worker_restarted``, ``worker_recycled``) as they happen.
``metrics.jsonl`` carries the memory at each console line (``memory_mb``: this
process, the training workers' largest and sum, what the machine has left).
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .env_pool import EnvPool, Result, process_rss_mb, reset_command, system_available_mb
from .evaluate import PoolEvaluation, evaluate_pool
from .trainer import Trainer, _Episode, crossed, step_failure

MAX_RESET_FAILURES = 20        # in a row, on one env


class ParallelTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.env, EnvPool):
            raise TypeError("ParallelTrainer needs an EnvPool")
        self.num_envs = len(self.env)
        self.timing = {"planning": 0.0, "updates": 0.0, "waiting for sims": 0.0}
        self._reset_cutoff: Dict[int, Optional[int]] = {}     # cutoff sent with each env's reset
        self._reset_start: Dict[int, tuple] = {}              # (grid, ratio) sent with each env's reset

    def _env_supports_cutoff(self) -> bool:
        return bool(getattr(self.env, "supports_cutoff", False))

    def _reset(self, i: int) -> tuple:
        """The reset command for env ``i``: a new seed and, with the mixture, a cutoff; with the
        start curriculum, the prefix ratio the worker is to hand over at."""
        seed, cutoff = self._draw_seed_and_cutoff()
        grid, ratio = self._draw_start()
        snapshot = self._draw_snapshot(grid)
        if grid > 0 and ratio is None and snapshot is None:
            grid = 0                                       # the library had nothing there after all
        self._reset_cutoff[i] = cutoff
        self._reset_start[i] = (grid, ratio)
        steps = self.start_curriculum.prefix_max_steps if self.start_curriculum is not None else 600
        key = self.start_curriculum.prefix_key if self.start_curriculum is not None else "progress_ratio"
        library = self.start_library
        return reset_command(seed, cutoff, ratio, steps, key, snapshot=snapshot,
                             save_dir=None if library is None else library.saving_to,
                             save_tag=f"e{i}_{seed}", save_bins=10 if library is None else library.bins)

    def _log_pool_events(self, source: Optional[EnvPool] = None, **fields: Any) -> None:
        source = self.env if source is None else source
        for event in source.drain_events():
            kind = event.pop("event")
            self._event(kind, **fields, **event)
            self.logger.print(f"{kind}: env {event.get('env')} {event.get('reason', '')}".rstrip())

    def _run_evaluation(self, seeds: List[int]) -> Dict[str, Any]:
        pool = self.env.spawn(min(len(seeds), self.num_envs))
        try:
            result = evaluate_pool(pool, self.agent, seeds, cutoff=self.eval_cutoff,
                                   residual=self.residual, components=self.reward_parts,
                                   max_minutes=self.cfg["evaluation"].get("max_minutes") or None,
                                   end_when_stalled=self.eval_end_when_stalled)
            self._log_pool_events(pool, pool="evaluation")
            self._report_cut(result.pop("cut_seeds", []), self.step)
        finally:
            pool.close()
        return result

    def _report_cut(self, seeds: List[int], evaluated_step: int) -> None:
        """evaluation.max_minutes ran out: which episodes were ended as simulation errors."""
        if seeds:
            self._event("evaluation_cut", evaluated_step=evaluated_step, seeds=list(seeds),
                        max_minutes=float(self.cfg["evaluation"]["max_minutes"]))
            self.logger.print(f"eval @ {evaluated_step}: {len(seeds)} episodes not finished after "
                              f"{self.cfg['evaluation']['max_minutes']} min, ended as simulation errors")

    def _memory_mb(self) -> Dict[str, Any]:
        workers = [rss for rss in self.env.worker_rss_mb() if rss is not None]
        values = {"main": process_rss_mb(), "workers_max": max(workers) if workers else None,
                  "workers_total": sum(workers) if workers else None,
                  "available": system_available_mb()}
        return {key: None if value is None else round(value, 1) for key, value in values.items()}

    def _run_updates(self, count: int) -> None:
        started = time.time()
        for _ in range(count):
            self._record_update(self.agent.update(self.replay))
        self.timing["updates"] += time.time() - started

    def _metrics_extra(self) -> Dict[str, Any]:
        total = sum(self.timing.values())
        extra: Dict[str, Any] = {"env_restarts": self.env.restarts,
                                 "env_recycles": self.env.recycles, "memory_mb": self._memory_mb()}
        if total > 0:
            extra["main_process_share"] = {k: v / total for k, v in self.timing.items()}
        return extra

    def _print_progress(self, started: float, start_step: int) -> None:
        super()._print_progress(started, start_step)
        total = sum(self.timing.values())
        if total > 0:
            share = ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in self.timing.items())
            self.logger.print(f"  main process since last line: {share}")
        memory = self._memory_mb()
        if memory["workers_max"] is not None:
            gb = {k: (v or 0.0) / 1024 for k, v in memory.items()}
            self.logger.print(f"  memory: main {gb['main']:.1f} GB, sims {gb['workers_total']:.1f} GB "
                              f"(largest {gb['workers_max']:.2f}), machine has "
                              f"{gb['available']:.1f} GB left; recycled {self.env.recycles}")
        self.timing = dict.fromkeys(self.timing, 0.0)

    def _steps_this_round(self, live: int, seed_steps: int, total_steps: int,
                          intervals: List[int]) -> int:
        stops = [total_steps] + [(self.step // m + 1) * m for m in intervals]
        if self.step < seed_steps:
            stops.append(seed_steps)
        return min(live, min(stops) - self.step)

    def _plan(self, envs: List[int], episodes: List[Optional[_Episode]],
              plan_means: torch.Tensor, seed_steps: int) -> List[np.ndarray]:
        """Actions for ``envs`` (all with a live episode): random before
        ``seed_steps``, otherwise one batched planning call."""
        started = time.time()
        if self.step < seed_steps:
            shape = self.env.action_space.shape
            actions = [self.rng.uniform(-1.0, 1.0, size=shape).astype(np.float32) for _ in envs]
        elif self.residual is None:
            actions, plan_means[envs] = self.agent.act_batch(
                np.stack([episodes[i].observations[-1] for i in envs]),
                first_step=[episodes[i].length == 0 for i in envs],
                previous_means=plan_means[envs], eval_mode=False)
            actions = list(actions)
        else:
            boxes = [self.residual.box_for(episodes[i].last_info) for i in envs]
            actions, plan_means[envs] = self.agent.act_batch(
                np.stack([episodes[i].observations[-1] for i in envs]),
                first_step=[episodes[i].length == 0 for i in envs],
                previous_means=plan_means[envs], eval_mode=False, bounds=boxes)
            actions = list(actions)
        self.timing["planning"] += time.time() - started
        return actions

    def _absorb(self, i: int, kind: str, action: Any, result: Result,
                episodes: List[Optional[_Episode]], reset_failures: List[int]) -> int:
        """One worker's answer into the books; the number of env steps counted (a step: 1; a reset
        with a scripted prefix: the prefix's steps). Afterwards ``episodes[i] is None`` means the env
        needs a reset."""
        if kind == "reset":
            cutoff = self._reset_cutoff.pop(i, None)
            grid, ratio = self._reset_start.pop(i, (0, None))
            if result.ok:
                before = self.step
                episodes[i] = self._absorb_reset(result.value[0], result.value[1], cutoff, grid, ratio, env=i)
                reset_failures[i] = 0
                if self.step > before:
                    self._curriculum_tick()
                return self.step - before
            self._event("reset_failed", env=i, reason=result.error)
            reset_failures[i] += 1
            if reset_failures[i] >= MAX_RESET_FAILURES:
                raise RuntimeError(f"env {i} failed to reset {MAX_RESET_FAILURES} times "
                                   f"in a row: {result.error}")
            return 0
        failure = f"env_error: {result.error}" if not result.ok else step_failure(result.value[4])
        if failure is not None:
            self._drop_step(episodes[i], failure, env=i)
            episodes[i] = None
            return 0
        observation, reward, terminated, truncated, info = result.value
        episodes[i].add(action, reward, terminated, info["labels"], observation, info)
        self._book_snapshot(info.get("snapshot_saved"), env=i)
        self.step += 1
        self._curriculum_tick()
        if terminated or truncated:
            self._close_episode(episodes[i], reason="terminated" if terminated else "truncated",
                                success=bool(info.get("is_success", False)))
            episodes[i] = None
        return 1

    def _summary(self, final) -> Dict[str, Any]:
        summary = {"total_env_steps": self.step, "episodes": self.episode_index,
                   "success_count": self.success_count, "env_errors": self.env_errors,
                   "wall_breaches": self.wall_breaches, "env_restarts": self.env.restarts, "env_recycles": self.env.recycles,
                   "updates": self.agent.update_count,
                   "final_checkpoint": str(final) if final else None}
        self._event("finished", **summary)
        self.logger.close()
        self.logger.print(f"finished: {summary}")
        return summary

    def train(self) -> Dict[str, Any]:
        t = self.cfg["training"]
        total_steps = int(t["total_steps"])
        seed_steps = int(t["seed_steps"])
        initial_updates = int(t["initial_updates"])
        updates_per_step = int(t["updates_per_step"])
        checkpoint_every = int(self.cfg["checkpoint"]["interval"])
        eval_every = int(self.cfg["evaluation"]["interval"])
        print_every = int(self.cfg["logging"]["print_interval"])
        intervals = [checkpoint_every, eval_every, print_every]
        n = self.num_envs

        episodes: List[Optional[_Episode]] = [None] * n
        reset_failures = [0] * n
        plan_means = torch.zeros(n, self.agent.horizon, self.agent.action_dim,
                                 device=self.agent.device)
        self._pretrain()
        pretrained = self.agent.update_count > 0
        owed = 0
        started, start_step = time.time(), self.step
        self._last_print = (started, self.step)
        self.logger.print(f"training from step {self.step} to {total_steps} with {n} envs "
                          f"in lockstep; replay has {self.replay.size} transitions")
        while self.step < total_steps and not self.stopping:
            before = self.step
            live = [i for i in range(n) if episodes[i] is not None]
            stepping = live[:self._steps_this_round(len(live), seed_steps, total_steps,
                                                    intervals)]
            commands: Dict[int, tuple] = {}
            if stepping:
                actions = self._plan(stepping, episodes, plan_means, seed_steps)
                commands.update({i: ("step", actions[j]) for j, i in enumerate(stepping)})
            for i in range(n):
                if episodes[i] is None:
                    commands[i] = self._reset(i)

            self.env.send(commands)
            self._run_updates(owed)                 # on the GPU while the scenes step
            owed = 0
            wait_started = time.time()
            results = self.env.receive()
            self.timing["waiting for sims"] += time.time() - wait_started
            self._log_pool_events()

            stepped = 0
            for i in sorted(results):
                kind, action = commands[i]
                stepped += self._absorb(i, kind, action, results[i], episodes, reset_failures)

            if self.step >= seed_steps and self.replay.can_sample():
                if not pretrained:
                    self.logger.print(f"pretraining on seed data: {initial_updates} updates")
                    self._run_updates(initial_updates)
                    pretrained = True
                else:
                    owed += stepped * updates_per_step
            if stepped == 0:
                continue
            at_checkpoint = crossed(before, self.step, checkpoint_every)
            at_eval = crossed(before, self.step, eval_every)
            if at_checkpoint or at_eval:
                self._run_updates(owed)
                owed = 0
            if at_checkpoint:
                self.save()
            if at_eval:
                self._evaluate()
            if crossed(before, self.step, print_every):
                self._print_progress(started, start_step)

        self._run_updates(owed)
        final = self.save() if self.cfg["checkpoint"].get("save_final", True) else None
        return self._summary(final)


class AsyncTrainer(ParallelTrainer):
    """See the module docstring: no round barrier, one update between polls."""

    MAX_OWED_ROUNDS = 2           # stop dispatching steps when this many rounds of updates are owed
    IDLE_WAIT = 1.0               # seconds to wait for a sim when nothing else is to be done
    SUPPORTS_BACKGROUND_EVAL = True
    EXAM_POLL_SECONDS = 0.05      # how often the evaluation beside training is looked at

    # ------------------------------------------------- evaluation beside training --
    # evaluation.background: true (2026-11-15, work log). An evaluation takes as long as its slowest
    # episode (hundreds of simulator steps of a second or two each), and training used to stand still
    # for it. Here, at an evaluation's step, the agent as it is then is copied (acting_copy) and the
    # evaluation's episodes run on their own simulator pool while training goes on; the main loop
    # hands the copy's actions to those simulators between its own work. Same seeds, same episodes,
    # same numbers as the pausing evaluation of that agent (tests/test_parallel.py). What changes:
    #   * the result arrives later. eval.jsonl carries the step the agent was copied at; the schedule
    #     (stages, guard, ladder) and best.pt react when the result arrives, i.e. some ten thousand
    #     steps later than with the pausing evaluation. best.pt is the agent that was evaluated (its
    #     checkpoint is written at the copy and kept only if it is the best), not the one training
    #     has reached by then.
    #   * evaluations run one at a time; one that comes due while the previous is still running waits
    #     with its own copy, and at most one waits: a newer one replaces it (event evaluation_dropped).
    #     evaluation.max_minutes ends an evaluation that takes longer (the unfinished episodes count
    #     as simulation errors, event evaluation_cut), here and in the pausing evaluation. At the end of training the outstanding ones are finished first. A run that
    #     is interrupted (Ctrl+C, an error) drops them (event evaluation_dropped): not redone on a resume.
    #   * the evaluation's simulators share the CPU with the training's.
    def _queue_evaluation(self) -> None:
        ev = self.cfg["evaluation"]
        seeds = [int(ev["seed"]) + i for i in range(int(ev["episodes"]))]
        pending = None
        if self.keep_best:
            pending = save_checkpoint(self.run_dir / "checkpoints" / f"exam_step_{self.step}.pt",
                                      self.checkpoint_payload())
        if self._exams:                                    # at most one waits: the newer one replaces it
            stale = self._exams.pop()
            self._event("evaluation_dropped", evaluated_steps=[stale["step"]], reason="a newer one came due")
            if stale["checkpoint"] is not None:
                Path(stale["checkpoint"]).unlink(missing_ok=True)
        self._exams.append({"step": self.step, "seeds": seeds, "agent": self.agent.acting_copy(),
                            "residual": copy.deepcopy(self.residual), "cutoff": self.eval_cutoff,
                            "checkpoint": pending})
        self.logger.print(f"eval @ {self.step}: started beside training "
                          f"({len(self._exams) + (self._exam is not None)} outstanding)")

    def _poll_evaluation(self, wait: float = 0.0) -> None:
        if self._exam is None:
            if not self._exams:
                return
            job = self._exams.pop(0)
            job["pool"] = self.env.spawn(min(len(job["seeds"]), self.num_envs))
            job["run"] = PoolEvaluation(job["pool"], job["agent"], job["seeds"], cutoff=job["cutoff"],
                                        residual=job["residual"], components=self.reward_parts,
                                        end_when_stalled=self.eval_end_when_stalled)
            job["started"] = time.time()
            self._exam = job
        elif wait <= 0.0 and time.time() - self._exam_polled < self.EXAM_POLL_SECONDS:
            return
        job = self._exam
        started = time.time()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):       # the episodes' RNG streams, not training's
            done = job["run"].step(wait)
        limit = self.cfg["evaluation"].get("max_minutes")
        if not done and limit and time.time() - job["started"] > 60.0 * float(limit):
            self._report_cut(job["run"].cut(), job["step"])
            done = True
        self._exam_polled = time.time()
        self.timing["evaluation beside training"] += self._exam_polled - started
        self._log_pool_events(job["pool"], pool="evaluation")
        if done:
            self._finish_evaluation(job)

    def _finish_evaluation(self, job: Dict[str, Any]) -> None:
        result = job["run"].result()
        job["pool"].close()
        self._exam = None
        summary = result["summary"]
        self.logger.write("eval", {"step": job["step"], **summary})
        self._event("evaluation", minutes=(time.time() - job["started"]) / 60.0,
                    evaluated_step=job["step"], background=True)
        self.logger.print(f"eval @ {job['step']}: success {summary['success_rate']:.2f}, "
                          f"return {summary['return_mean']:.3f}, length {summary['length_mean']:.1f} "
                          f"(ran beside training, reported at step {self.step})")
        rate = float(summary["success_rate"])
        pending = job["checkpoint"]
        if pending is not None:
            if self.best_rate is None or rate > self.best_rate:
                self.best_rate = rate
                best = self.run_dir / "checkpoints" / "best.pt"
                os.replace(pending, best)
                self._event("best_checkpoint", path=str(best), success_rate=rate, evaluated_step=job["step"])
            else:
                Path(pending).unlink(missing_ok=True)
        if self.schedule is not None:
            for change in self.schedule.on_evaluation(self.step, rate, self.start_curriculum):
                self._event("schedule", **change)
            self._apply_schedule()

    def _settle_evaluations(self) -> None:
        """At the end of training: finish the outstanding evaluations."""
        while self._exam is not None or self._exams:
            self._poll_evaluation(wait=0.5)

    def _close_evaluations(self) -> None:
        """Whatever is still outstanding when training leaves (an interrupt, an error): dropped, said so."""
        jobs = ([self._exam] if self._exam is not None else []) + self._exams
        if jobs:
            self._event("evaluation_dropped", evaluated_steps=[job["step"] for job in jobs])
        for job in jobs:
            if job.get("pool") is not None:
                job["pool"].close()
            if job["checkpoint"] is not None:
                Path(job["checkpoint"]).unlink(missing_ok=True)
        self._exam, self._exams = None, []

    def train(self) -> Dict[str, Any]:
        self._exam: Optional[Dict[str, Any]] = None        # the evaluation running beside training
        self._exams: List[Dict[str, Any]] = []             # the ones waiting for it to end
        self._exam_polled = 0.0
        if self.background_eval:
            self.timing["evaluation beside training"] = 0.0
        try:
            return self._train()
        finally:
            self._close_evaluations()

    def _train(self) -> Dict[str, Any]:
        t = self.cfg["training"]
        total_steps = int(t["total_steps"])
        seed_steps = int(t["seed_steps"])
        initial_updates = int(t["initial_updates"])
        updates_per_step = int(t["updates_per_step"])
        checkpoint_every = int(self.cfg["checkpoint"]["interval"])
        eval_every = int(self.cfg["evaluation"]["interval"])
        print_every = int(self.cfg["logging"]["print_interval"])
        n = self.num_envs
        max_owed = self.MAX_OWED_ROUNDS * n * updates_per_step

        episodes: List[Optional[_Episode]] = [None] * n
        reset_failures = [0] * n
        idle: Set[int] = set(range(n))              # envs with no command in flight
        in_flight: Dict[int, tuple] = {}
        plan_means = torch.zeros(n, self.agent.horizon, self.agent.action_dim,
                                 device=self.agent.device)
        self._pretrain()
        pretrained = self.agent.update_count > 0
        owed = 0
        started, start_step = time.time(), self.step
        self._last_print = (started, self.step)
        self.logger.print(f"training from step {self.step} to {total_steps} with {n} envs "
                          f"not waiting for each other; replay has {self.replay.size} transitions")
        while self.step < total_steps and not self.stopping:
            commands: Dict[int, tuple] = {}
            to_step = sorted(i for i in idle if episodes[i] is not None)
            if to_step and owed <= max_owed:
                actions = self._plan(to_step, episodes, plan_means, seed_steps)
                commands.update({i: ("step", actions[j]) for j, i in enumerate(to_step)})
            for i in sorted(idle):
                if episodes[i] is None:
                    commands[i] = self._reset(i)
            if commands:
                self.env.send(commands)
                in_flight.update(commands)
                idle -= set(commands)

            if owed > 0:                        # one update, then look for answers
                self._run_updates(1)
                owed -= 1
                wait = 0.0
            else:
                wait = self.IDLE_WAIT
            if self._exam is not None or self._exams:
                self._poll_evaluation()
                wait = min(wait, self.EXAM_POLL_SECONDS)
            wait_started = time.time()
            results = self.env.receive_ready(wait)
            self.timing["waiting for sims"] += time.time() - wait_started
            self._log_pool_events()

            for i in sorted(results):
                idle.add(i)
                kind, action = in_flight.pop(i)
                before = self.step
                counted = self._absorb(i, kind, action, results[i], episodes, reset_failures)
                if not counted:
                    continue
                if self.step >= seed_steps and self.replay.can_sample():
                    if not pretrained:
                        self.logger.print(f"pretraining on seed data: {initial_updates} updates")
                        self._run_updates(initial_updates)
                        pretrained = True
                    else:
                        owed += counted * updates_per_step
                at_checkpoint = crossed(before, self.step, checkpoint_every)
                at_eval = crossed(before, self.step, eval_every)
                if at_checkpoint or at_eval:
                    self._run_updates(owed)
                    owed = 0
                if at_checkpoint:
                    self.save()
                if at_eval:
                    if self.background_eval:
                        self._queue_evaluation()
                    else:
                        self._evaluate()
                if crossed(before, self.step, print_every):
                    self._print_progress(started, start_step)
                if self.step >= total_steps:
                    break

        self._run_updates(owed)
        self._settle_evaluations()
        final = self.save() if self.cfg["checkpoint"].get("save_final", True) else None
        return self._summary(final)
