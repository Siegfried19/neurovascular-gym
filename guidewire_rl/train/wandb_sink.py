"""Forward the run's jsonl records to Weights & Biases (``logging.wandb``; docs/wandb_metrics.md).

    logging:
      wandb: {enabled: true, project: neurovascular-gym, mode: offline, tags: [round2], select: core}

The jsonl files stay the record of truth (everything, every episode); wandb is the dashboard.
``select: core`` (the default; user 2026-11-11: "only the core, the rest is for the agent") sends the
``CORE_CURVES`` the core view of scripts/wandb_layout.py draws; ``select: dashboard`` sends all of
``DASHBOARD`` below (docs/wandb_metrics.md section 4). Names are ``<group>/<name>``, the group being
wandb's section (eval, learning, train, crutches, sim); a curve whose source is missing from a record
(the crutches group in a run without crutches) is simply not sent. wandb's own hardware curves are
off and the x-axis curve is hidden, so the page holds only these.
``select: all`` sends every number of every record instead: ``<stream>/<key>``, nested dicts flattened
with ``/`` (``metrics/reach/P2``), lists of numbers indexed, strings left out. Every call carries
``env_step`` (the record's ``step``), declared as the x-axis of every metric, so streams logged out of
order never trip wandb's step check. ``mode: offline`` (the default) writes to ``<run_dir>/wandb`` and
is uploaded later with ``wandb sync <run_dir>/wandb/offline-run-*``; ``online`` streams live. With
wandb not installed or ``enabled: false`` the sink is inert and the run is exactly as before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

KNOWN_KEYS = {"enabled", "project", "entity", "mode", "tags", "name", "group", "notes", "dir", "select"}
SELECTS = ("core", "dashboard", "all")


def _get(record: Mapping[str, Any], *path: str) -> Optional[float]:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value) if np.isfinite(value) else None


def _sum(*paths: tuple) -> Callable[[Mapping[str, Any]], Optional[float]]:
    def pick(record: Mapping[str, Any]) -> Optional[float]:
        parts = [_get(record, *p) for p in paths]
        return None if all(p is None for p in parts) else float(sum(p for p in parts if p is not None))
    return pick


def _scaled(path: tuple, factor: float) -> Callable[[Mapping[str, Any]], Optional[float]]:
    def pick(record: Mapping[str, Any]) -> Optional[float]:
        value = _get(record, *path)
        return None if value is None else value * factor
    return pick


def _groups(path: tuple, names: Mapping[str, str], share_of: Optional[str] = None,
            scale: float = 1.0) -> Callable[[Mapping[str, Any]], Optional[Dict[str, float]]]:
    """A dict field as several curves: ``names`` maps the record's keys (or a key's prefix ending in
    ``:``) to curve names; every named curve is sent (0 when the key is absent) so a stacked chart
    keeps its layers. ``share_of``: divide by that field (``"*"``: by the dict's own total)."""
    def pick(record: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        if not isinstance(value, Mapping):
            return None
        out = {name: 0.0 for name in dict.fromkeys(names.values())}
        matched = False
        for key, number in value.items():
            name = names.get(str(key)) or next((n for k, n in names.items()
                                                if k.endswith(":") and str(key).startswith(k)), None)
            if name is not None and isinstance(number, (int, float)) and not isinstance(number, bool):
                out[name] += float(number) * scale
                matched = True
        if not matched:
            return None                   # none of these numbers in this record: send nothing
        if share_of is not None:
            total = (sum(float(v) for v in value.values() if isinstance(v, (int, float)))
                     if share_of == "*" else _get(record, share_of))
            if not total:
                return None
            out = {k: v / total for k, v in out.items()}
        return out
    return pick


def _each(path: tuple, prefix: str = "", names: Optional[Mapping[str, str]] = None
          ) -> Callable[[Mapping[str, Any]], Optional[Dict[str, float]]]:
    """Every number of a dict field as its own curve: ``names[key]`` when given (keys it lacks are left
    out), else ``prefix`` + key. Keys without a number are not sent (unlike ``_groups``, no 0 fill: a
    grid with no episode has no success rate)."""
    def pick(record: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        if not isinstance(value, Mapping):
            return None
        out = {}
        for key, v in value.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
                continue
            if names is None:
                out[f"{prefix}{key}"] = float(v)
            elif str(key) in names:
                out[names[str(key)]] = float(v)
        return out
    return pick


def _when(match: tuple, path: tuple) -> Callable[[Mapping[str, Any]], Optional[float]]:
    """``path`` of a record whose field ``match[0]`` equals ``match[1]`` (one kind of event)."""
    def pick(record: Mapping[str, Any]) -> Optional[float]:
        return _get(record, *path) if record.get(match[0]) == match[1] else None
    return pick


END_REASONS = {"success": "success", "failure:": "hard_failure", "simulation_error": "sim_error",
               "wall_breach": "wall_breach", "stalled": "stalled", "other": "step_limit"}
PROGRESS_BINS = {**{str(k): f"bin_{(k - 1) * 10:02d}_{k * 10:02d}" for k in range(1, 11)}, "done": "done"}
# the five phases' reach rates: off the dashboard (user 2026-11-11: they miss the no-hook shortcut,
# WORK_LOG 19:25); kept for a later route-milestone version
REACH = {"P1": "cath_past_bct_30mm", "P2": "hook_set", "P3": "cath_in_bct", "P4": "wire_in_rcca",
         "success": "success"}
PHASES = {"0": "arch", "1": "hook", "2": "select", "3": "wire_out", "4": "follow"}
GRIDS = {str(k): f"grid_{k:02d}" for k in range(11)}
TIMES = {"planning": "planning", "updates": "updates", "waiting for sims": "waiting_sims"}
REWARD = {k: k for k in ("progress", "follow", "tier1", "tier2", "force_cath", "force_wire", "step",
                         "wrong_vessel", "failure", "smooth")}
GB = 1.0 / 1024.0

# (wandb name "<panel>/<curve>", stream, source). The source is a key path, or a function giving a
# number or a dict of numbers (each sent as "<name>/<key>"). The panels follow docs/wandb_metrics.md
# section 4 (user 2026-11-11: English names, no numbers, everything the document lists): eval = did it
# learn the task (the 50 test episodes), learning = is learning healthy, train = what the training
# episodes do, crutches = start curriculum, demonstrations, schedule (sent only when the run has them),
# sim = simulator and machine (not "system": wandb's own hardware curves are system.*).
# scripts/wandb_layout.py lays them out.
DASHBOARD = (
    # eval: one point per evaluation, the 50 fixed-seed test episodes from the insertion point
    ("eval/tier2_success", "eval", ("success_rate",)),
    ("eval/tier2_ci95_low", "eval", ("success_ci95_low",)),
    ("eval/tier2_ci95_high", "eval", ("success_ci95_high",)),
    ("eval/tier1_rate", "eval", ("tier1_rate",)),
    ("eval/tier1_ci95_low", "eval", ("tier1_ci95_low",)),
    ("eval/tier1_ci95_high", "eval", ("tier1_ci95_high",)),
    ("eval/end", "eval", _groups(("end_reasons",), END_REASONS, share_of="episodes")),
    ("eval/progress_end_mean", "eval", ("progress_end_mean",)),
    ("eval/progress_end", "eval", _groups(("progress_end_hist",), PROGRESS_BINS, share_of="*")),
    ("eval/success_steps_mean", "eval", ("success_steps_mean",)),
    ("eval/success_steps_median", "eval", ("success_steps_median",)),
    ("eval/cath_force_peak_p90", "eval", ("catheter_tip_force_peak_p90",)),
    ("eval/cath_force_peak_max", "eval", ("catheter_tip_force_peak_max",)),
    ("eval/wire_force_peak_p90", "eval", ("distal_force_peak_p90",)),
    ("eval/wire_force_peak_max", "eval", ("distal_force_peak_max",)),
    ("eval/lead_end_mean", "eval", ("lead_end_mean",)),
    ("eval/lead_end_below_30mm", "eval", ("lead_end_below_30_rate",)),
    ("eval/wrong_vessel_per_ep", "eval", ("events_per_episode", "wrong_vessel")),
    ("eval/dislodged_per_ep", "eval", ("events_per_episode", "dislodged")),
    ("eval/return_mean", "eval", ("return_mean",)),
    ("eval/length_mean", "eval", ("length_mean",)),
    ("eval/reward", "eval", _groups(("reward_parts",), REWARD)),
    ("eval/shield_active_frac", "eval", ("shield_active_frac",)),
    ("eval/cath_moving_frac", "eval", ("catheter_moving_frac",)),
    ("eval/lead_p50", "eval", ("d_p50",)),
    # learning: every logging.update_interval updates (averaged), Q against the return per evaluation
    ("learning/consistency_loss", "updates", ("consistency_loss",)),
    ("learning/reward_loss", "updates", ("reward_loss",)),
    ("learning/value_loss", "updates", ("value_loss",)),
    ("learning/termination_loss", "updates", ("termination_loss",)),
    ("learning/termination_recall", "updates", ("termination_recall",)),
    ("learning/termination_false_positive", "updates", ("termination_false_positive",)),
    ("learning/model_grad_norm", "updates", ("model_grad_norm",)),
    ("learning/policy_grad_norm", "updates", ("policy_grad_norm",)),
    ("learning/policy_entropy", "updates", ("policy_entropy",)),
    ("learning/policy_scale", "updates", ("policy_scale",)),
    ("learning/aux", "updates", _groups((), {"aux_wire_force_loss": "wire_force",
                                                "aux_cath_force_loss": "cath_force",
                                                "aux_wire_clearance_loss": "wire_clearance",
                                                "aux_cath_clearance_loss": "cath_clearance"})),
    ("learning/policy_loss", "updates", ("policy_loss",)),
    ("learning/total_loss", "updates", ("total_loss",)),
    ("learning/q_mean", "eval", ("q_mean",)),
    ("learning/g_mean", "eval", ("g_mean",)),
    ("learning/q_minus_g", "eval", ("q_minus_g_mean",)),
    ("learning/q0", "eval", ("q0",)),
    ("learning/g0", "eval", ("g0",)),
    ("learning/q_vs_return_episodes", "eval", ("q_versus_return_episodes",)),
    # train: the last 100 training episodes, every logging.print_interval steps
    ("train/train_tier2_rate", "metrics", ("success_rate",)),
    ("train/train_tier1_rate", "metrics", ("tier1_rate",)),
    ("train/progress_end_mean", "metrics", ("progress_end_mean",)),
    ("train/progress_end", "metrics", _groups(("progress_end_hist",), PROGRESS_BINS, share_of="*")),
    ("train/return_mean", "metrics", ("return_mean",)),
    ("train/length_mean", "metrics", ("length_mean",)),
    ("train/reward", "metrics", _groups(("reward_parts",), REWARD)),
    # the same in three kinds (the core view): hints, penalties, the finish (tiers minus the hard failure)
    ("train/reward_hint", "metrics", _sum(("reward_parts", "progress"), ("reward_parts", "follow"))),
    ("train/reward_penalty", "metrics", _sum(("reward_parts", "force_cath"), ("reward_parts", "force_wire"),
                                              ("reward_parts", "step"), ("reward_parts", "wrong_vessel"),
                                              ("reward_parts", "smooth"))),
    ("train/reward_finish", "metrics", _sum(("reward_parts", "tier1"), ("reward_parts", "tier2"),
                                             ("reward_parts", "failure"))),
    ("train/wrong_vessel_per_ep", "metrics", ("events_per_episode", "wrong_vessel")),
    ("train/dislodged_per_ep", "metrics", ("events_per_episode", "dislodged")),
    ("train/shield_active_frac", "metrics", ("shield_active_frac",)),
    ("train/shield_delta_mean", "metrics", ("shield_delta_mean",)),
    ("train/lead_p10", "metrics", ("d_p10",)),
    ("train/lead_p50", "metrics", ("d_p50",)),
    ("train/lead_p90", "metrics", ("d_p90",)),
    ("train/lead_end_p50", "metrics", ("lead_end_p50",)),
    ("train/cath_moving_frac", "metrics", ("catheter_moving_frac",)),
    # crutches: only in runs that have them (the record carries nothing otherwise)
    ("crutches/start_frontier", "metrics", ("start_frontier",)),
    ("crutches/start_weight", "metrics", _groups(("start_weights",), GRIDS)),
    ("crutches/start_share", "metrics", _groups(("start_grid_share",), GRIDS)),
    ("crutches/success_by_start", "metrics", _each(("success_by_start_grid",), names=GRIDS)),
    ("crutches/prefix_step_share", "metrics", ("prefix_step_share",)),
    ("crutches/prefix_gave_up_rate", "metrics", ("prefix_gave_up_rate",)),
    ("crutches/demo_fraction", "metrics", ("demo_fraction",)),
    ("crutches/demo_episodes", "metrics", ("demo_episodes",)),
    ("crutches/self_demo_episodes", "metrics", ("self_demo_episodes",)),
    ("crutches/schedule_stage", "metrics", ("schedule_stage_index",)),
    ("crutches/hint_scale", "metrics", ("hint_scale",)),
    ("crutches/force_scale", "metrics", ("force_scale",)),
    ("crutches/ladder_level", "metrics", ("ladder_level",)),
    ("crutches/guard_trips", "metrics", ("guard_trips",)),
    ("crutches/residual_delta", "metrics", ("residual_delta",)),
    # sim: the simulator and the machine
    ("sim/steps_per_s", "metrics", ("steps_per_s",)),
    ("sim/steps_per_s_recent", "metrics", ("steps_per_s_recent",)),
    ("sim/sim_errors_total", "metrics", ("env_errors",)),
    ("sim/wall_breaches_total", "metrics", ("wall_breaches",)),
    ("sim/sim_error_rate", "metrics", ("error_rate",)),
    ("sim/breach_rate", "metrics", ("breach_rate",)),
    ("sim/sim_error_at", "metrics", _groups(("error_phase_hist",), PHASES)),
    ("sim/env_restarts", "metrics", ("env_restarts",)),
    ("sim/env_recycles", "metrics", ("env_recycles",)),
    ("sim/mem_available_gb", "metrics", _scaled(("memory_mb", "available"), GB)),
    ("sim/mem_main_gb", "metrics", _scaled(("memory_mb", "main"), GB)),
    ("sim/mem_worker_max_gb", "metrics", _scaled(("memory_mb", "workers_max"), GB)),
    ("sim/mem_workers_total_gb", "metrics", _scaled(("memory_mb", "workers_total"), GB)),
    ("sim/time", "metrics", _groups(("main_process_share",), TIMES)),
    ("sim/episodes_total", "metrics", ("episodes",)),
    ("sim/updates_total", "metrics", ("updates",)),
    ("sim/replay_size", "metrics", ("replay_size",)),
    ("sim/eval_minutes", "events", _when(("event", "evaluation"), ("minutes",))),
)


def dashboard(stream: str, record: Mapping[str, Any]) -> Dict[str, float]:
    """The ``DASHBOARD`` curves this record carries."""
    out: Dict[str, float] = {}
    for name, source_stream, source in DASHBOARD:
        if source_stream != stream:
            continue
        value = source(record) if callable(source) else _get(record, *source)
        if isinstance(value, Mapping):
            out.update({f"{name}/{key}": float(v) for key, v in value.items()})
        elif value is not None:
            out[name] = value
    return out


# What goes to wandb by default (``select: core``; user 2026-11-11: "only the core, the rest is for the
# agent", read from the jsonl files): the curves of the core view in scripts/wandb_layout.py (``CORE``),
# 51 curves: 34 plus the section "curriculum and schedule" (user 2026-11-13: frontier, schedule stage,
# demonstration quota, the force penalty's factor, guard trips and ladder level, success by start grid;
# sent only by runs that have them). ``select: dashboard`` sends all of ``DASHBOARD``, ``select: all`` every number.
CORE_CRUTCHES = (
    "crutches/start_frontier", "crutches/schedule_stage", "crutches/demo_fraction",
    "crutches/force_scale", "crutches/guard_trips", "crutches/ladder_level",
    *(f"crutches/success_by_start/{name}" for name in GRIDS.values()),
)
CORE_CURVES = (
    "eval/tier2_success", "eval/tier2_ci95_low", "eval/tier2_ci95_high",
    "eval/tier1_rate", "eval/tier1_ci95_low", "eval/tier1_ci95_high",
    *(f"eval/end/{k}" for k in END_REASONS.values()),
    "eval/progress_end_mean", "train/progress_end_mean",
    "eval/cath_force_peak_p90", "eval/wire_force_peak_p90",
    "train/return_mean", "train/reward_hint", "train/reward_penalty", "train/reward_finish",
    "train/lead_p10", "train/lead_p50", "train/lead_p90", "train/shield_active_frac",
    "learning/consistency_loss", "learning/reward_loss", "learning/q_mean", "learning/g_mean",
    "learning/model_grad_norm", "learning/policy_grad_norm", "learning/policy_entropy",
    "sim/steps_per_s_recent", "sim/steps_per_s", "sim/sim_errors_total", "sim/wall_breaches_total",
    *CORE_CRUTCHES,
)
_CORE = frozenset(CORE_CURVES)


def core(stream: str, record: Mapping[str, Any]) -> Dict[str, float]:
    """The ``CORE_CURVES`` this record carries."""
    return {k: v for k, v in dashboard(stream, record).items() if k in _CORE}


def flatten(record: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Numbers of a (nested) record as ``prefix/key`` -> float; text and None are left out."""
    out: Dict[str, float] = {}
    for key, value in record.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                out[name] = float(value)
        elif isinstance(value, Mapping):
            out.update(flatten(value, name))
        elif isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value)
            if array.dtype.kind in "biuf" and array.ndim == 1:
                for i, x in enumerate(array.tolist()):
                    if np.isfinite(x):
                        out[f"{name}/{i}"] = float(x)
    return out


class WandbSink:
    """One run's forwarder; ``enabled`` says whether anything is sent."""

    def __init__(self, run=None, select: str = "core") -> None:
        if select not in SELECTS:
            raise ValueError(f"logging.wandb.select must be one of {SELECTS}, got {select!r}")
        self.run = run
        self.enabled = run is not None
        self.select = select
        self.sent = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any], run_dir: str | Path, resume: bool = False) -> "WandbSink":
        """``resume``: the run goes on from a checkpoint; the wandb run this run folder used last
        (``<run_dir>/wandb/latest-run``) is continued instead of a new one (one run, one set of curves)."""
        spec = (config.get("logging") or {}).get("wandb") or {}
        if not spec or not spec.get("enabled", True):
            return cls(None)
        unknown = set(spec) - KNOWN_KEYS
        if unknown:
            raise KeyError(f"logging.wandb: unknown key(s) {sorted(unknown)}; known: {sorted(KNOWN_KEYS)}")
        select = str(spec.get("select", "core"))
        if select not in SELECTS:
            raise ValueError(f"logging.wandb.select must be one of {SELECTS}, got {select!r}")
        try:
            import wandb  # noqa: WPS433 (optional dependency)
        except ImportError:
            print("logging.wandb is on but wandb is not installed; logging to the jsonl files only", flush=True)
            return cls(None)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        mode = str(spec.get("mode", "offline"))
        os.environ.setdefault("WANDB_SILENT", "true")
        previous = previous_run_id(run_dir) if resume else None
        more = {"id": previous, "resume": "allow"} if previous else {}
        run = wandb.init(project=spec.get("project", "guidewire_rl"), entity=spec.get("entity"), **more,
                         mode=mode, dir=str(spec.get("dir", run_dir)), name=spec.get("name", run_dir.name),
                         group=spec.get("group"), notes=spec.get("notes"), tags=list(spec.get("tags", []) or []),
                         config=_plain_config(config), reinit=True,
                         # wandb's own hardware curves (system.cpu, system.gpu.*, ...) off: they crowd
                         # the page and the machine numbers we want are in sim/ (user 2026-11-11)
                         settings=wandb.Settings(x_disable_stats=True))
        run.define_metric("env_step", hidden=True)    # the x-axis, not a chart of its own
        run.define_metric("*", step_metric="env_step")
        return cls(run, select)

    def log(self, stream: str, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if self.select == "all":
            data = flatten(record, stream)
            data.pop(f"{stream}/step", None)          # the step is the x-axis, not a metric
        elif self.select == "dashboard":
            data = dashboard(stream, record)
        else:
            data = core(stream, record)
        if not data:
            return                                    # nothing of this record is on the dashboard
        step = record.get("step")
        if step is not None:
            data["env_step"] = float(step)
        self.run.log(data)
        self.sent += 1

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()
            self.enabled = False


def previous_run_id(run_dir: str | Path) -> Optional[str]:
    """The wandb run id this run folder used last: ``<run_dir>/wandb/latest-run`` points at
    ``run-<date>_<time>-<id>`` (or ``offline-run-...``); None when there is none."""
    latest = Path(run_dir) / "wandb" / "latest-run"
    try:
        name = latest.resolve().name if latest.exists() else None
    except OSError:
        return None
    if not name or "-" not in name:
        return None
    return name.rsplit("-", 1)[1] or None


def _plain_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    def plain(value):
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value
    return plain(config)
