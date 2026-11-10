"""Record scripted episodes of the two-device task as demonstrations for the trainer.

Runs ``StagedController`` (``guidewire_rl/env/two_device/script.py``) from the insertion point in the
env of a config (default ``configs/two_devices_round1.yaml``, training half of the targets) and
writes one ``episode_<seed>.npz`` per seed, plus ``summary.json`` and ``README.md`` (format, success
rate, time per stage, how the failed ones failed) into the output directory.

    conda run --no-capture-output -n guidewire_rl python scripts/record_demos.py \\
        --seeds 1000-1019 --workers 10 --out runs/demos_round1

Every episode is kept, failed ones too (``success`` says which); the transitions are the script's
own, so the early phases of a failed episode are still good data. The action stored is the one
passed to ``env.step`` (the policy action); what the shell made of it is ``executed_action``.
A step on which the simulation went wrong (``simulation_error``: NaN; ``wall_breached``: a device's
centerline outside the vessel mesh) is not recorded and ends the episode, as the trainers drop it;
the episode also ends once the script gives up (``StagedController.failed``: it would only send
zeros from there on).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "two_devices_round1.yaml"
LABELS = ("distal_force", "catheter_tip_force", "wire_tip_clearance", "catheter_tip_clearance",
          "substep_motion", "wall_penetration")      # the last two NaN for a config without the wall monitors
EVENTS = ("wrong_vessel", "dislodged", "bare_advance", "wire_lead_arch", "wire_past_fork")
SHIELD_CAUSES = ("speed", "residual", "hard_rule", "force", "device_limit", "protector_cross",
                 "protector_clamp")
PHASE_NAMES = ("arch", "hook", "select", "wire_out", "follow")


def parse_seeds(text: str) -> List[int]:
    """``1000-1019`` or ``3,5,8`` or a mix (non-negative seeds)."""
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def stage_times(log: Sequence[Sequence[Any]], steps: int) -> Dict[str, int]:
    """Steps spent per top-level stage (``wire:fork`` counts as ``wire``) from the controller's log."""
    out: Dict[str, int] = {}
    entries = [(str(name).split(":")[0], int(step)) for name, step in log] + [("end", steps)]
    for (name, start), (_, stop) in zip(entries[:-1], entries[1:]):
        out[name] = out.get(name, 0) + max(stop - start, 0)
    return out


def failure_mode(success: bool, terminated: bool, truncated: bool, info: Dict[str, Any],
                 ctl_stage: str, tries: int, log: Sequence[Sequence[Any]]) -> str:
    if success:
        return "success"
    if info.get("hard_failure"):
        return f"hard failure ({info.get('failure_reason', '')})"
    if info.get("force_blowup_N", 0) > 0:
        return f"force blow-up ({info['force_blowup_N']:.0f} N; the contact solve blew up, that step is not recorded)"
    if info.get("simulation_error") or info.get("sim_error"):
        return "simulation error"
    if info.get("wall_breached"):
        return "wall breach (a centerline outside the vessel mesh; that step is not recorded)"
    gave_up = [str(name).split(":", 1)[1] for name, _ in log if str(name).startswith("failed:")]
    if gave_up:
        return f"the script gave up ({gave_up[0]}) in {ctl_stage}"
    rewires = sum(1 for name, _ in log if str(name) == "place:rewire")
    if rewires:
        return f"catheter came out of the trunk {rewires}x while the wire worked; out of steps in {ctl_stage}"
    if tries:
        return f"{tries} hook placement tries; out of steps in {ctl_stage}"
    return f"out of steps in {ctl_stage}"


def run_episode(env, seed: int, max_steps: int, out_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    from guidewire_rl.env.two_device.script import STAGES, ScriptParams, StagedController

    ctl = StagedController(ScriptParams(**params))
    obs, info = env.reset(seed=seed)
    ctl.reset(env)
    obs_list = [np.asarray(obs, np.float32)]
    phase_list = [int(info.get("phase", 0))]
    actions, executed, rewards, terminated, truncated = [], [], [], [], []
    labels = {k: [] for k in LABELS}
    events, causes, stages, success_flags = [], [], [], []
    t0 = time.time()
    te = tr = False
    for _ in range(max_steps):
        action = ctl.act(env)
        stage_now = ctl.stage
        obs, reward, te, tr, info = env.step(action)
        if info.get("simulation_error") or info.get("wall_breached"):   # the physics is wrong on this step
            break                                                      # (not recorded, as the trainers drop it)
        obs_list.append(np.asarray(obs, np.float32))
        phase_list.append(int(info.get("phase", phase_list[-1])))
        actions.append(np.asarray(action, np.float32))
        executed.append(np.asarray(info.get("executed_action", np.full(4, np.nan)), np.float32))
        rewards.append(float(reward))
        terminated.append(bool(te))
        truncated.append(bool(tr))
        step_labels = info.get("labels", {}) or {}
        for k in LABELS:
            value = step_labels.get(k, np.nan)
            labels[k].append(np.nan if value is None else float(value))
        ev = info.get("events", {}) or {}
        events.append([int(ev.get(k, 0)) for k in EVENTS])
        sc = info.get("shield_causes", {}) or {}
        causes.append([int(sc.get(k, 0)) for k in SHIELD_CAUSES])
        stages.append(STAGES.index(stage_now) if stage_now in STAGES else -1)
        success_flags.append(bool(info.get("is_success", False)))
        if te or tr or getattr(ctl, "failed", False):
            break
    steps = len(actions)
    success = bool(success_flags[-1]) if success_flags else False
    record = dict(
        obs=np.stack(obs_list).astype(np.float32),
        action=np.stack(actions).astype(np.float32),
        reward=np.asarray(rewards, np.float32),
        phase=np.asarray(phase_list, np.int64),
        terminated=np.asarray(terminated, bool),
        truncated=np.asarray(truncated, bool),
        executed_action=np.stack(executed).astype(np.float32),
        events=np.asarray(events, np.int16),
        event_names=np.asarray(EVENTS),
        shield_causes=np.asarray(causes, np.int8),
        shield_cause_names=np.asarray(SHIELD_CAUSES),
        stage=np.asarray(stages, np.int8),
        stage_names=np.asarray(STAGES),
        is_success=np.asarray(success_flags, bool),
        success=np.asarray(success),
        seed=np.asarray(seed, np.int64),
        target_depth_mm=np.asarray(float(ctl.readout.target_depth) if ctl.readout is not None else np.nan),
        stage_log=np.asarray(json.dumps(ctl.log)),
    )
    for k in LABELS:
        record[f"labels/{k}"] = np.asarray(labels[k], np.float32)
    path = out_dir / f"episode_{seed}.npz"
    np.savez_compressed(path, **record)
    phases = np.asarray(phase_list)
    first_step = {PHASE_NAMES[k]: (int(np.argmax(phases == k)) if np.any(phases == k) else None)
                  for k in range(len(PHASE_NAMES))}
    return dict(seed=seed, success=success, steps=steps, terminated=bool(te), truncated=bool(tr),
                max_phase=int(phases.max()), first_step_of_phase=first_step,
                stage_steps=stage_times(ctl.log, steps), tries=int(ctl.tries),
                failure=failure_mode(success, bool(te), bool(tr), info, ctl.stage, ctl.tries, ctl.log),
                events={k: int(np.asarray(events)[:, i].sum()) for i, k in enumerate(EVENTS)} if events else {},
                shield_steps={k: int(np.asarray(causes)[:, i].sum()) for i, k in enumerate(SHIELD_CAUSES)}
                if causes else {},
                return_=float(np.sum(rewards)), wall_s=round(time.time() - t0, 1),
                target_depth_mm=float(record["target_depth_mm"]), file=str(path))


def apply_experiment(cfg: Dict[str, Any], dt: float | None, mesh: str | None) -> None:
    """Validation knobs that --set cannot reach (docs/simulation_speed.md): a physics step (the labels'
    expected substep counts follow it) and another wall mesh for the phantom."""
    if dt:
        cfg["env"]["scene"]["simulation"]["dt_simulation"] = float(dt)
        n = int((1.0 / 7.5) / float(dt))
        for label in cfg["env"].get("labels", []):
            if "expected_substeps" in label:
                label["expected_substeps"] = n
    if mesh:
        cfg["env"]["scene"]["vessel_tree"]["mesh"] = str(mesh)


def worker(args) -> List[Dict[str, Any]]:
    config_path, overrides, seeds, max_steps, out_dir, params, dt, mesh = args
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env

    cfg = load_config(config_path, overrides)
    apply_experiment(cfg, dt, mesh)
    env = build_env(cfg["env"])
    rows = []
    try:
        for seed in seeds:
            try:
                row = run_episode(env, seed, max_steps, Path(out_dir), params)
            except Exception as exc:          # noqa: BLE001 - one bad episode must not stop the rest
                row = dict(seed=seed, success=False, steps=0, failure=f"error: {exc!r}")
            print(f"seed {seed}: {'success' if row['success'] else 'fail'} in {row['steps']} steps "
                  f"({row['failure']})", flush=True)
            rows_dir = Path(out_dir) / "rows"
            rows_dir.mkdir(parents=True, exist_ok=True)
            (rows_dir / f"seed_{seed}.json").write_text(json.dumps(row))
            rows.append(row)
    finally:
        env.close()
    return rows


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(float(centre - half), 3), round(float(centre + half), 3))


def write_readme(out_dir: Path, rows: List[Dict[str, Any]], config: str, overrides: Sequence[str],
                 obs_dim: int) -> None:
    ok = [r for r in rows if r.get("success")]
    n, k = len(rows), len(ok)
    lo, hi = wilson(k, n)
    from guidewire_rl.env.two_device.script import STAGES
    stage_names = tuple(s for s in STAGES if s != "done")
    lines = [
        "# Scripted demonstrations, two devices (round 1)",
        "",
        f"Written by `scripts/record_demos.py` ({time.strftime('%Y-%m-%d %H:%M')}): `StagedController`",
        f"(`guidewire_rl/env/two_device/script.py`) from the insertion point, config `{config}`"
        + (f" with `{' '.join(overrides)}`" if overrides else "") + ".",
        "",
        f"**{k}/{n} reached success** (Wilson 95 %: {lo}-{hi}); successful episodes took "
        + (f"{min(r['steps'] for r in ok)}-{max(r['steps'] for r in ok)} steps." if ok else "-"),
        "",
        "## Files",
        "",
        "One `episode_<seed>.npz` per seed (`np.load(path)`; T = steps of the episode):",
        "",
        "| key | shape, type | what |",
        "| --- | --- | --- |",
        f"| `obs` | (T+1, {obs_dim}) float32 | observation, row 0 = after `reset` |",
        "| `action` | (T, 4) float32 | normalized action passed to `env.step` (the script's = the policy action) |",
        "| `reward` | (T,) float32 | env reward |",
        "| `phase` | (T+1,) int64 | referee phase from `info`, row 0 = after `reset` |",
        "| `terminated`, `truncated` | (T,) bool | as returned by `env.step` |",
        "| `labels/distal_force`, `labels/catheter_tip_force`, `labels/wire_tip_clearance`, "
        "`labels/catheter_tip_clearance`, `labels/substep_motion`, `labels/wall_penetration` | (T,) float32 | "
        "`info['labels']` (NaN where the config has no such label) |",
        "| `executed_action` | (T, 4) float32 | what the safety shell executed (`info['executed_action']`) |",
        "| `shield_causes` | (T, 7) int8 | shell stage that changed the command (names: `shield_cause_names`) |",
        "| `events` | (T, 5) int16 | referee events that step (names: `event_names`) |",
        "| `stage` | (T,) int8 | the script's stage when it chose the action (names: `stage_names`) |",
        "| `is_success` | (T,) bool | `info['is_success']` |",
        "| `success`, `seed`, `target_depth_mm`, `stage_log` | scalars | outcome, seed, target depth in "
        "the carotid, the script's stage changes (JSON) |",
        "",
        "Failed episodes are kept (`success` False): their early phases are good script data.",
        "`summary.json`: one row per episode (steps per stage, first step of each phase, events, shell "
        "interventions, return, failure mode).",
        "",
        "## Steps per stage (successful episodes: median, min-max)",
        "",
        "| stage | median | min | max |",
        "| --- | --- | --- | --- |",
    ]
    for name in stage_names:
        values = [r["stage_steps"].get(name, 0) for r in ok]
        if values:
            lines.append(f"| {name} | {int(np.median(values))} | {min(values)} | {max(values)} |")
    lines += ["", "## Episodes", "", "| seed | outcome | steps | max phase | target mm | failure |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: r["seed"]):
        lines.append(f"| {r['seed']} | {'success' if r.get('success') else 'fail'} | {r.get('steps')} | "
                     f"{r.get('max_phase', '')} | {r.get('target_depth_mm', float('nan')):.1f} | "
                     f"{'' if r.get('success') else r.get('failure', '')} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def run_children(a, seeds: List[int], n: int, timeout_per_episode: float) -> None:
    """One child process per chunk of seeds (``--child``); a child that dies (SOFA can segfault or
    hang after a numerical blow-up) or overruns its time is killed, and its unfinished seeds are
    tried once more in a fresh child."""
    import subprocess

    out_dir = Path(a.out)
    base = [sys.executable, str(Path(__file__).resolve()), "--child", "--config", a.config,
            "--out", str(out_dir), "--max-steps", str(a.max_steps)]
    for item in a.set:
        base += ["--set", item]
    for item in a.param:
        base += ["--param", item]
    if a.dt:
        base += ["--dt", str(a.dt)]
    if a.mesh:
        base += ["--mesh", a.mesh]

    def missing(chunk):
        return [s for s in chunk if not (out_dir / "rows" / f"seed_{s}.json").exists()]

    for attempt in range(2):
        todo = missing(seeds)
        if not todo:
            return
        chunks = [todo[i::n] for i in range(min(n, len(todo)))]
        procs = []
        for chunk in chunks:
            cmd = base + ["--seeds", ",".join(str(s) for s in chunk)]
            procs.append((subprocess.Popen(cmd), chunk, time.time() + timeout_per_episode * len(chunk) + 120.0))
        while procs:
            time.sleep(2.0)
            still = []
            for proc, chunk, deadline in procs:
                if proc.poll() is not None:
                    continue
                if time.time() > deadline:
                    proc.kill()
                    proc.wait()
                    print(f"worker for seeds {chunk} overran its time: killed", flush=True)
                    continue
                still.append((proc, chunk, deadline))
            procs = still
        lost = missing(todo)
        if lost:
            print(f"no result for seeds {lost} (worker died or was killed)"
                  + ("; trying them once more" if attempt == 0 else ""), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "runs" / "demos_round1"))
    ap.add_argument("--seeds", default="1000-1019", help="e.g. 1000-1019 or 3,5,8")
    ap.add_argument("--workers", type=int, default=1, help="child processes, each with its own env")
    ap.add_argument("--max-steps", type=int, default=2000, help="safety stop; the env's MaxSteps ends it first")
    ap.add_argument("--episode-timeout", type=float, default=600.0, help="wall seconds per episode before a "
                    "worker is killed")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                    help="ScriptParams override (Python literal)")
    ap.add_argument("--dt", type=float, default=None, help="physics step in s (labels' substep counts follow)")
    ap.add_argument("--mesh", default=None, help="another wall mesh for the phantom (BiroPhantomArch mesh=)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import ast
    params = {k: ast.literal_eval(v) for k, v in (item.split("=", 1) for item in a.param)}
    if a.child:                                   # one worker's share, rows written per episode
        worker((a.config, list(a.set), seeds, a.max_steps, str(out_dir), params, a.dt, a.mesh))
        return 0
    rows_dir = out_dir / "rows"
    for s in seeds:                               # a fresh recording of these seeds
        (rows_dir / f"seed_{s}.json").unlink(missing_ok=True)
    t0 = time.time()
    run_children(a, seeds, max(1, min(a.workers, len(seeds))), a.episode_timeout)
    rows = []
    for s in seeds:
        path = rows_dir / f"seed_{s}.json"
        rows.append(json.loads(path.read_text()) if path.exists() else
                    dict(seed=s, success=False, steps=0, failure="no result (worker died or was killed)"))
    obs_dim = 0
    for r in rows:
        if r.get("file"):
            with np.load(r["file"]) as z:
                obs_dim = int(z["obs"].shape[1])
            break
    summary = dict(config=a.config, overrides=a.set, dt=a.dt, mesh=a.mesh, params=params, seeds=seeds,
                   success=int(sum(bool(r.get("success")) for r in rows)), episodes=len(rows),
                   wall_s=round(time.time() - t0, 1), rows=rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_readme(out_dir, rows, a.config, a.set, obs_dim)
    print(f"{summary['success']}/{summary['episodes']} reached success; {out_dir}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
