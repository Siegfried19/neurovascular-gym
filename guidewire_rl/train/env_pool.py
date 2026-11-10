"""Several copies of one env, each in its own process.

A SOFA scene steps on one thread (plus its small collision pool), so training
gets its speed from stepping N scenes at once. Every worker builds its own env
from the same factory. Each round the parent sends one command per worker
(``reset`` or ``step``), may do other work while the scenes step (the trainer
runs its updates), then collects the answers: all of them with ``receive``, or
whichever are back with ``receive_ready`` (so an env can be given its next
command while the others are still stepping).

A worker that raises inside ``reset``/``step`` reports the error and keeps its
env. A worker that dies (a SOFA segfault) or does not answer within
``timeout`` seconds (``training.step_timeout_s``, default 120) is replaced by a
fresh one and its answer is a failure; either way the run goes on. Building a
scene (at the start and after a replacement) may take up to
``max(timeout, BUILD_TIMEOUT)`` seconds. Each
replacement is also queued in ``events`` (the trainer writes them to
``events.jsonl`` as ``worker_restarted``).

SOFA does not give back all the memory a scene took when the scene is torn down
(``docs/simulator_changes.md``, "场景越跑越慢": two devices about 24 MB per 1000
steps, a straight line), so a worker grows for as long as it lives and only its
exit returns that memory. With ``max_rss_mb`` (``training.worker_max_rss_mb``;
default None = never) a worker whose resident memory is above that many MB when
it is sent a reset is closed and a fresh one is started in its place first; the
reset then goes to the fresh worker. Only at a reset, so no episode is cut: the
finished episode's steps are already in the parent. Counted in ``recycles`` (not
``restarts``) and queued in ``events`` as ``worker_recycled``.

Every worker raises its own ``oom_score_adj`` to ``WORKER_OOM_SCORE_ADJ``: when
the machine runs out of memory the kernel kills the biggest worker (whose loss
is one episode, and the pool replaces it) instead of the parent (the whole run)
or the desktop session (which sets 200-300 on its own processes).

A reset command carries either the seed or ``{"seed": s, "cutoff": k}``; with a
cutoff the worker calls ``env.set_cutoff(k)`` before ``env.reset(seed=s)``
(``training.cutoff_mixture``).

Workers are started with ``spawn`` (the parent may already hold CUDA), so the
factory and its arguments must be picklable: a module-level function or class
and plain config values, e.g. ``EnvPool(build_env, (config["env"],), num_envs=8)``.

``split_cpus`` keeps the workers off the main process's core: the main process
(planning and updates, single-thread bound) runs about 2x slower when it shares
a core or lands on an efficiency core (measured on the i9-14900K, 2026-10-28).
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import connection as mp_connection
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (AbstractSet, Any, Callable, Dict, List, Mapping, Optional, Sequence, Set,
                    Tuple)

import numpy as np

# the parts of info the trainer reads (task metrics, curriculum and residual box included)
INFO_KEYS = ("is_success", "simulation_error", "labels", "phase", "max_phase", "events",
             "shield_delta", "d", "executed_action", "nominal_action", "nominal_available",
             "sim_error", "hard_failure", "failure_reason", "cutoff", "shield_penalized",
             "wall_breached")
DEFAULT_TIMEOUT = 120.0        # seconds without an answer before a worker is replaced
BUILD_TIMEOUT = 600.0          # seconds a (new) worker may take to build its scene
RESTART_ATTEMPTS = 3           # a replacement that fails to build is tried this many times
WORKER_OOM_SCORE_ADJ = 500     # above the desktop's 200-300: out of memory, a worker goes first
CLOSE_WAIT = 10.0              # seconds a recycled worker gets to shut its scene down


def step_timeout(config: Mapping[str, Any]) -> float:
    """``training.step_timeout_s`` (seconds; default 120)."""
    return float(config.get("training", {}).get("step_timeout_s", DEFAULT_TIMEOUT))


def worker_max_rss_mb(config: Mapping[str, Any]) -> Optional[float]:
    """``training.worker_max_rss_mb`` (MB; default None = workers are never recycled)."""
    value = config.get("training", {}).get("worker_max_rss_mb")
    return None if value is None else float(value)


def process_rss_mb(pid: Optional[int] = None) -> Optional[float]:
    """Resident memory of a process in MB (``VmRSS``; this process by default); None
    when it cannot be read (no such process, not Linux)."""
    try:
        with open(f"/proc/{'self' if pid is None else pid}/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def system_available_mb() -> Optional[float]:
    """Memory the machine can still hand out in MB (``MemAvailable``); None if unknown."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def reset_command(seed: int, cutoff: Optional[int] = None) -> Tuple[str, Any]:
    """The pool's reset command; plain seed when there is no cutoff (as before)."""
    return ("reset", int(seed) if cutoff is None else {"seed": int(seed), "cutoff": int(cutoff)})


def split_cpus() -> Optional[Tuple[Set[int], Set[int]]]:
    """(main, workers): the fastest physical core, with its hyper-thread
    siblings, for the main process; every other allowed CPU for the workers.
    None when the CPU layout cannot be read (then nothing is pinned)."""
    try:
        allowed = os.sched_getaffinity(0)
        root = Path("/sys/devices/system/cpu")
        speed = {cpu: int((root / f"cpu{cpu}/cpufreq/cpuinfo_max_freq").read_text())
                 for cpu in allowed}
        fastest = min(allowed, key=lambda cpu: (-speed[cpu], cpu))
        siblings = (root / f"cpu{fastest}/topology/thread_siblings_list").read_text().strip()
    except (AttributeError, OSError, ValueError):
        return None
    main: Set[int] = set()
    for part in siblings.split(","):
        low, _, high = part.partition("-")
        main.update(range(int(low), int(high or low) + 1))
    main &= allowed
    workers = set(allowed) - main
    return (main, workers) if workers else None


def _small_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: info[key] for key in INFO_KEYS if key in info}


def _worker(conn, factory: Callable, args: Sequence[Any], kwargs: Mapping[str, Any],
            seed: Optional[int], cpus: Optional[AbstractSet[int]]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C is the parent's to handle
    try:                                             # raising it needs no privileges
        with open("/proc/self/oom_score_adj", "w") as adj:
            adj.write(str(WORKER_OOM_SCORE_ADJ))
    except OSError:
        pass
    if cpus:                                         # before SOFA starts its threads
        os.sched_setaffinity(0, cpus)
    if seed is not None:                             # anything the env draws outside reset(seed)
        random.seed(seed)
        np.random.seed(seed)
    try:
        env = factory(*args, **kwargs)
    except BaseException as exc:  # pylint: disable=broad-except
        conn.send(("error", f"building the env failed: {exc!r}"))
        conn.close()
        return
    conn.send(("ok", {
        "observation_space": env.observation_space, "action_space": env.action_space,
        "label_names": list(env.label_names), "max_episode_steps": env.max_episode_steps,
        "observation_names": list(getattr(env, "observation_names", [])),
        "supports_cutoff": callable(getattr(env, "set_cutoff", None)),
    }))
    try:
        while True:
            try:
                command, data = conn.recv()
            except EOFError:                         # parent is gone
                break
            if command == "close":
                break
            try:
                if command == "reset":
                    if isinstance(data, Mapping):
                        if data.get("cutoff") is not None:
                            env.set_cutoff(int(data["cutoff"]))
                        data = data.get("seed")
                    observation, info = env.reset(seed=data)
                    reply = ("ok", (observation, _small_info(info)))
                elif command == "step":
                    observation, reward, terminated, truncated, info = env.step(data)
                    reply = ("ok", (observation, reward, terminated, truncated, _small_info(info)))
                else:
                    reply = ("error", f"unknown command {command!r}")
            except Exception as exc:  # pylint: disable=broad-except
                reply = ("error", repr(exc))
            conn.send(reply)
    finally:
        try:
            env.close()
        except Exception:  # pylint: disable=broad-except
            pass
        conn.close()


@dataclass
class Result:
    """One worker's answer. ``value`` is ``(observation, info)`` for a reset and
    ``(observation, reward, terminated, truncated, info)`` for a step."""
    ok: bool
    value: Any = None
    error: str = ""


class EnvPool:
    def __init__(self, factory: Callable, args: Sequence[Any] = (),
                 kwargs: Optional[Mapping[str, Any]] = None, *, num_envs: int,
                 seed: Optional[int] = None, timeout: float = DEFAULT_TIMEOUT,
                 cpus: Optional[AbstractSet[int]] = None,
                 max_rss_mb: Optional[float] = None) -> None:
        """``cpus``: the CPUs the workers may run on (default: wherever the OS puts them).
        ``max_rss_mb``: recycle a worker above this resident memory at its next reset
        (module docstring; None: never)."""
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.factory, self.args, self.kwargs = factory, tuple(args), dict(kwargs or {})
        self.seed, self.timeout = seed, float(timeout)
        self.cpus = set(cpus) if cpus else None
        self.max_rss_mb = None if max_rss_mb is None else float(max_rss_mb)
        self.restarts = 0                           # replaced because they died or hung
        self.recycles = 0                           # replaced because they grew past max_rss_mb
        self.events: List[Dict[str, Any]] = []      # worker replacements not yet logged
        self._ctx = mp.get_context("spawn")
        self._procs: List[Any] = [None] * num_envs
        self._conns: List[Any] = [None] * num_envs
        self._pending: Dict[int, str] = {}
        self._sent_at: Dict[int, float] = {}
        self._dead: set = set()
        try:
            for index in range(num_envs):        # all workers build their scene at once
                self._launch(index)
            specs = [self._ready(index) for index in range(num_envs)]
        except BaseException:
            self.close()
            raise
        spec = specs[0]
        self.observation_space = spec["observation_space"]
        self.action_space = spec["action_space"]
        self.label_names: List[str] = spec["label_names"]
        self.max_episode_steps = spec["max_episode_steps"]
        self.observation_names: List[str] = spec["observation_names"]
        self.supports_cutoff = bool(spec.get("supports_cutoff", False))

    def __len__(self) -> int:
        return len(self._conns)

    def spawn(self, num_envs: int) -> "EnvPool":
        """Another pool of the same env (e.g. for evaluation; close it when done)."""
        return EnvPool(self.factory, self.args, self.kwargs, num_envs=num_envs,
                       seed=None if self.seed is None else self.seed + 10_000,
                       timeout=self.timeout, cpus=self.cpus)

    def worker_rss_mb(self) -> List[Optional[float]]:
        """Each worker's resident memory in MB (None where it cannot be read)."""
        return [None if proc is None or not proc.is_alive() else process_rss_mb(proc.pid)
                for proc in self._procs]

    # --------------------------------------------------------- workers --
    def _launch(self, index: int) -> None:
        parent, child = self._ctx.Pipe()
        replaced = self.restarts + self.recycles
        seed = None if self.seed is None else self.seed + index + 1000 * replaced
        proc = self._ctx.Process(target=_worker, name=f"env-{index}", daemon=True,
                                 args=(child, self.factory, self.args, self.kwargs, seed,
                                       self.cpus))
        proc.start()
        child.close()
        self._procs[index], self._conns[index] = proc, parent

    def _receive_one(self, index: int, deadline: float) -> Tuple[str, Any]:
        conn = self._conns[index]
        if not conn.poll(max(0.0, deadline - time.monotonic())):
            raise TimeoutError
        return conn.recv()                                     # EOFError if the worker died

    def _ready(self, index: int) -> Dict[str, Any]:
        try:
            status, payload = self._receive_one(index, time.monotonic()
                                                + max(self.timeout, BUILD_TIMEOUT))
        except (EOFError, OSError, TimeoutError) as exc:
            raise RuntimeError(f"env worker {index} did not start ({exc!r})") from exc
        if status != "ok":
            raise RuntimeError(f"env worker {index}: {payload}")
        return payload

    def _stop(self, index: int) -> None:
        proc, conn = self._procs[index], self._conns[index]
        if conn is not None:
            conn.close()
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()

    def _relaunch(self, index: int) -> None:
        """Start worker ``index`` again (the old one is stopped) and wait for its scene."""
        for attempt in range(1, RESTART_ATTEMPTS + 1):
            self._launch(index)
            try:
                self._ready(index)
                break
            except RuntimeError as exc:
                self._stop(index)
                if attempt == RESTART_ATTEMPTS:
                    raise
                self.events.append({"event": "worker_restart_failed", "env": index,
                                    "attempt": attempt, "reason": str(exc)})

    def _restart(self, index: int, reason: str = "") -> None:
        self._stop(index)
        self.restarts += 1
        self._relaunch(index)
        self.events.append({"event": "worker_restarted", "env": index, "reason": reason,
                            "restarts": self.restarts})

    def _recycle(self, index: int, rss_mb: float) -> None:
        """Close an idle worker that grew past ``max_rss_mb`` and start a fresh one."""
        proc, conn = self._procs[index], self._conns[index]
        started = time.monotonic()
        try:
            conn.send(("close", None))              # lets it shut its scene down
        except (BrokenPipeError, OSError):
            pass
        proc.join(CLOSE_WAIT)
        self._stop(index)
        self.recycles += 1
        self._relaunch(index)
        self.events.append({"event": "worker_recycled", "env": index,
                            "reason": f"{rss_mb:.0f} MB > {self.max_rss_mb:.0f} MB",
                            "rss_mb": round(rss_mb, 1), "max_rss_mb": self.max_rss_mb,
                            "seconds": round(time.monotonic() - started, 2),
                            "recycles": self.recycles})

    def drain_events(self) -> List[Dict[str, Any]]:
        """Worker replacements since the last call (for the run's events.jsonl)."""
        events, self.events = self.events, []
        return events

    # ------------------------------------------------------------ rounds --
    def send(self, commands: Mapping[int, Tuple[str, Any]]) -> None:
        """``{env index: ("reset", seed) or ("step", action)}``. Answers come from
        ``receive`` (all of them) or ``receive_ready`` (whichever are back). An env
        that still owes an answer cannot be sent another command."""
        busy = [index for index in commands if index in self._pending]
        if busy:
            raise RuntimeError(f"env(s) {busy} have not answered their last command yet")
        unknown = [command for command, _ in commands.values() if command not in ("reset", "step")]
        if unknown:
            raise ValueError(f"unknown command {unknown[0]!r}")
        if self.max_rss_mb is not None:               # between episodes: swap grown workers
            for index, (command, _) in commands.items():
                if command == "reset" and index not in self._dead:
                    proc = self._procs[index]
                    rss = None if proc is None else process_rss_mb(proc.pid)
                    if rss is not None and rss > self.max_rss_mb:
                        self._recycle(index, rss)
        now = time.monotonic()
        for index, (command, data) in commands.items():
            self._pending[index] = command
            self._sent_at[index] = now
            try:
                self._conns[index].send((command, data))
            except (BrokenPipeError, OSError):
                self._dead.add(index)

    @property
    def pending(self) -> List[int]:
        """Envs that owe an answer."""
        return sorted(self._pending)

    def receive_ready(self, timeout: float = 0.0) -> Dict[int, Result]:
        """Answers from the pending envs that have one; waits up to ``timeout``
        seconds for the first. A worker that died or has been silent for
        ``self.timeout`` seconds is replaced and answers with a failure."""
        results: Dict[int, Result] = {}
        if not self._pending:
            return results
        live = {self._conns[index]: index for index in self._pending if index not in self._dead}
        for conn in (mp_connection.wait(list(live), timeout=timeout) if live else []):
            index = live[conn]
            try:
                status, payload = conn.recv()
            except (EOFError, OSError):                        # the worker died
                self._dead.add(index)
                continue
            results[index] = (Result(True, payload) if status == "ok"
                              else Result(False, error=str(payload)))
            del self._pending[index]
        now = time.monotonic()
        for index in sorted(self._pending):
            if index in self._dead or now - self._sent_at[index] > self.timeout:
                reason = ("died" if index in self._dead
                          else f"timed out (no answer in {self.timeout:.0f} s)")
                self._restart(index, reason)
                self._dead.discard(index)
                del self._pending[index]
                results[index] = Result(False, error=f"env worker {index} {reason}; restarted")
        return results

    def receive(self) -> Dict[int, Result]:
        """Answers from every pending env (waits for all of them)."""
        results: Dict[int, Result] = {}
        while self._pending:
            deadline = min(self._sent_at[index] for index in self._pending) + self.timeout
            results.update(self.receive_ready(max(0.0, deadline - time.monotonic())))
        return results

    def close(self) -> None:
        for conn in self._conns:
            if conn is None:
                continue
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            if proc is not None:
                proc.join(10)
        for index in range(len(self._procs)):
            self._stop(index)
        self._procs = [None] * len(self._procs)
        self._conns = [None] * len(self._conns)
