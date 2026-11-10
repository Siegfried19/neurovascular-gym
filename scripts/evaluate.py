"""Evaluate a checkpoint (ours or the reference G13) on fixed seeds.

    python scripts/evaluate.py --config configs/g13.yaml \
        --checkpoint reference/checkpoints_g13/step_260000.pt --episodes 10

    # two-device task: 100 episodes from the insertion point on held-out targets,
    # 8 processes, every episode saved for the workbench (visual/workbench --replay)
    python scripts/evaluate.py --config configs/two_devices_round1.yaml \
        --checkpoint runs/<run>/checkpoints/step_N.pt --episodes 100 --workers 8 \
        --save-episodes runs/<run>/eval_step_N --out runs/<run>/eval_step_N.json
    # ablation: the catheter's two actions forced to 0
    ... --freeze-catheter

What it does:

* Targets: when the config's target has a ``split`` (train / eval halves of the
  candidate points), the eval half is used unless ``--target-split`` says
  otherwise; ``--set env.scene.target.split=...`` still wins.
* Episodes start at the insertion point (the config's ``start``) and are never
  cut early: an env with ``set_cutoff`` gets the cutoff that only stops at
  success, failure or the step limit.
* Planner mean action; each episode seeded from its seed, so the numbers do not
  depend on ``--workers``. The residual box (``planning.residual``) is applied as
  in training, with the box width saved in the checkpoint.
* Report: success rate with its 95 % Wilson interval, highest-phase histogram,
  how episodes ended (success / the env's failure reason / simulation error /
  other), events per episode, episode time (steps and simulated seconds; wall
  time), peak force of every ``*force*`` label (the two tips), shield
  intervention, catheter-moving share.
* ``--save-episodes DIR``: one ``.npz`` per episode in the workbench's
  ``EpisodeLog`` format (``visual/workbench/parts/frame.py``); per-frame extras
  carry phase, events, shield, d, nominal/executed action and both tips (mm,
  vessel frame).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config
from guidewire_rl.env import build_env
from guidewire_rl.model import build_agent
from guidewire_rl.train.checkpoint import (is_reference_g13, load_checkpoint,
                                           reference_agent_state)
from guidewire_rl.train.curriculum import SUCCESS_PHASE, ResidualSchedule
from guidewire_rl.train.evaluate import report, run_episode

WORKBENCH = WORKSPACE_ROOT / "visual" / "workbench"
CATHETER_ACTIONS = (2, 3)        # catheter push and turn in a two-instrument action
FRAME_INFO_KEYS = ("phase", "max_phase", "shield_delta", "d", "nominal_action",
                   "executed_action", "policy_action")


# ------------------------------------------------------------------ options --
def target_split_overrides(config_path: str, split: str) -> List[str]:
    """``env.scene.target.split=<split>`` when the config's target has a split."""
    if split == "keep":
        return []
    target = load_config(config_path)["env"]["scene"].get("target") or {}
    return [f"env.scene.target.split={split}"] if isinstance(target, Mapping) and "split" in target else []


def freeze_catheter(action: np.ndarray) -> np.ndarray:
    action = np.array(action, dtype=np.float32, copy=True)
    if action.size > max(CATHETER_ACTIONS):
        action[list(CATHETER_ACTIONS)] = 0.0
    return action


def full_episode_cutoff(env, config: Mapping[str, Any]) -> Optional[int]:
    """The cutoff that never cuts, when the env takes cutoffs at all (an env
    without a cutoff part raises AttributeError from ``set_cutoff``)."""
    setter = getattr(env, "set_cutoff", None)
    if not callable(setter):
        return None
    mixture = config["training"].get("cutoff_mixture") or {}
    cutoff = int(max(mixture.get("phases", [SUCCESS_PHASE])))
    try:
        setter(cutoff)
    except AttributeError:
        return None
    return cutoff


# ---------------------------------------------------------------- recording --
def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def instrument_tips_mm(env) -> Optional[List[List[float]]]:
    """Each instrument's tip in the vessel frame (mm), when the fluoroscopy tracks them."""
    try:
        fluoroscopy = env.intervention.fluoroscopy
        trackings = fluoroscopy.instrument_trackings3d
        from eve.util.coordtransform import tracking3d_to_vessel_cs
        return [tracking3d_to_vessel_cs(np.asarray(t[:1], dtype=np.float64), fluoroscopy.image_rot_zx,
                                        fluoroscopy.image_center)[0].tolist() for t in trackings]
    except Exception:  # pylint: disable=broad-except  # not every env/stEVE has per-instrument tracking
        return None


def frame_extras(env, info: Mapping[str, Any]) -> Dict[str, Any]:
    extras = {key: _plain(info[key]) for key in FRAME_INFO_KEYS if info.get(key) is not None}
    events = {k: int(v) for k, v in (info.get("events") or {}).items() if int(v)}
    if events:
        extras["events"] = events
    tips = instrument_tips_mm(env)
    if tips is not None:
        extras["instrument_tips_mm"] = tips
    return extras


class EpisodeRecorder:
    """``run_episode`` hooks that build workbench frames and save one ``.npz`` per episode."""

    def __init__(self, out_dir: Path, description: str = "") -> None:
        if str(WORKBENCH) not in sys.path:
            sys.path.insert(0, str(WORKBENCH))
        from parts import frame as frame_mod      # noqa: E402  (the workbench's own package)
        from parts import sources as sources_mod  # noqa: E402
        self.EpisodeLog = frame_mod.EpisodeLog
        self.sources = sources_mod
        self.out_dir = Path(out_dir)
        self.description = description
        self.info = None
        self.log = None
        self.episode, self.step, self.total, self.seed = -1, 0, 0.0, None

    def on_reset(self, env, observation, info, seed) -> None:
        key = str(env.intervention.vessel_tree.mesh_path)
        if self.info is None or self.info.vessel_key != key:
            self.info = self.sources.scene_info_from_env(env, self.description)
        self.log = self.EpisodeLog(self.info)
        self.episode += 1
        self.step, self.total, self.seed = 0, 0.0, seed
        self.log.append(self.sources.frame_from_env(env, episode=self.episode, step=0, obs=observation,
                                                    seed=seed, extras=frame_extras(env, info)))

    def on_step(self, env, action, observation, reward, terminated, truncated, info) -> None:
        self.step += 1
        self.total += float(reward)
        self.log.append(self.sources.frame_from_env(
            env, episode=self.episode, step=self.step, obs=observation,
            action=np.asarray(action, dtype=np.float32), raw_action=info.get("raw_action"),
            reward=float(reward), episode_return=self.total, terminated=bool(terminated),
            truncated=bool(truncated), success=bool(info.get("is_success", False)),
            labels={k: float(v) for k, v in (info.get("labels") or {}).items()},
            seed=self.seed, extras=frame_extras(env, info)))

    def on_error(self, env, action, error) -> None:
        self.step += 1
        last = self.log.frames[-1]
        self.log.append(self.sources.frame_from_env(
            env, episode=self.episode, step=self.step, obs=last.obs,
            action=np.asarray(action, dtype=np.float32), episode_return=self.total, truncated=True,
            seed=self.seed, extras={"error": repr(error)}))

    def save(self, order: int, record: Mapping[str, Any]) -> str:
        outcome = "success" if record["success"] else (
            "error" if record["simulation_error"] else "breach" if record.get("wall_breach")
            else f"P{record.get('max_phase', 0)}")
        path = self.out_dir / f"episode_{order:03d}_seed{record['seed']}_{outcome}.npz"
        return str(self.log.save(path))


# --------------------------------------------------------------- evaluation --
def load_agent(config: Mapping[str, Any], env, checkpoint_path: str, device: Optional[str]):
    agent = build_agent(config, observation_dim=env.observation_space.shape[0],
                        action_dim=env.action_space.shape[0],
                        episode_length=env.max_episode_steps, device=device)
    checkpoint = load_checkpoint(checkpoint_path)
    if is_reference_g13(checkpoint):
        state = reference_agent_state(checkpoint, config["aux_heads"][0]["name"])
    else:
        state = checkpoint["agent"]
    agent.load_state_dict(state, load_optimizers=False)
    residual = ResidualSchedule.from_config((config.get("planning") or {}).get("residual"))
    if residual is not None and checkpoint.get("residual"):
        residual.load_state_dict(checkpoint["residual"])
    return agent, residual


def evaluate_seeds(config: Mapping[str, Any], checkpoint_path: str, jobs: Sequence[tuple],
                   options: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Episodes ``jobs`` = [(order, seed), ...] in this process (one env)."""
    if options.get("torch_threads"):
        torch.set_num_threads(int(options["torch_threads"]))
    env = build_env(config["env"])
    try:
        agent, residual = load_agent(config, env, checkpoint_path, options.get("device"))
        cutoff = full_episode_cutoff(env, config)
        hook = freeze_catheter if options.get("freeze_catheter") else None
        recorder = (EpisodeRecorder(Path(options["save_dir"]), options.get("description", ""))
                    if options.get("save_dir") else None)
        records = []
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            for order, seed in jobs:
                torch.manual_seed(int(seed))
                started = time.time()
                record = run_episode(env, agent, seed, cutoff=cutoff, residual=residual,
                                     action_hook=hook, hooks=recorder)
                record["wall_s"] = time.time() - started
                record["order"] = int(order)
                if recorder is not None:
                    record["file"] = recorder.save(order, record)
                records.append(record)
                print(json.dumps(record), flush=True)
        return records
    finally:
        env.close()


def evaluate_parallel(config, checkpoint_path, jobs, options, workers: int) -> List[Dict[str, Any]]:
    chunks = [jobs[k::workers] for k in range(workers) if jobs[k::workers]]
    sub = dict(options, torch_threads=options.get("torch_threads") or 2)
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_seeds, config, checkpoint_path, chunk, sub) for chunk in chunks]
        records = [r for f in futures for r in f.result()]
    return sorted(records, key=lambda r: r["order"])


def table(summary: Mapping[str, Any]) -> str:
    lines = []
    low, high = summary["success_ci95"]
    lines.append(f"episodes {summary['episodes']} | success {summary['successes']}/{summary['episodes']}"
                 f" = {summary['success_rate']:.2f} (95% Wilson {low:.2f}-{high:.2f}) | "
                 f"simulation errors {summary['simulation_errors']}"
                 + (f" | wall breaches {summary['wall_breaches']}" if summary.get("wall_breaches") else ""))
    if "max_phase_hist" in summary:
        hist = summary["max_phase_hist"]
        lines.append("highest phase: " + " | ".join(
            f"{'success' if k == str(SUCCESS_PHASE) else 'P' + k} {v}" for k, v in hist.items()))
    if summary.get("end_reasons"):
        lines.append("how episodes ended: " + " | ".join(
            f"{k} {v}" for k, v in summary["end_reasons"].items()))
    if summary.get("events_per_episode"):
        lines.append("events per episode: " + " | ".join(
            f"{k} {v:.2f}" for k, v in summary["events_per_episode"].items()))
    t = summary.get("time", {})
    text = f"time: {t.get('steps_mean', 0):.0f} steps mean"
    if "seconds_mean" in t:
        text += f" ({t['seconds_mean']:.1f} s simulated)"
    if "success_steps_mean" in t:
        text += f", successes {t['success_steps_mean']:.0f} steps"
        if "success_seconds_mean" in t:
            text += f" ({t['success_seconds_mean']:.1f} s)"
    if "wall_s_per_episode" in t:
        text += f" | wall {t['wall_s_per_episode']:.1f} s per episode"
    lines.append(text)
    for name, f in (summary.get("peak_force") or {}).items():
        lines.append(f"peak {name}: max {f['max']:.3f}, mean of episode peaks {f['mean_of_episode_max']:.3f},"
                     f" p90 {f['p90_of_episode_max']:.3f}")
    extras = []
    if "shield_delta_mean" in summary:
        extras.append(f"shield mean change {summary['shield_delta_mean']:.3f}")
    if "shield_active_frac" in summary:
        extras.append(f"shield active {100 * summary['shield_active_frac']:.0f}% of steps")
    if "catheter_moving_frac" in summary:
        extras.append(f"catheter moving {100 * summary['catheter_moving_frac']:.0f}% of steps")
    if extras:
        lines.append(" | ".join(extras))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="first seed")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--target-split", default="eval",
                        help="train | eval | keep (default eval; only for targets with a split)")
    parser.add_argument("--freeze-catheter", action="store_true",
                        help="send 0 for the catheter's push and turn (actions 2 and 3)")
    parser.add_argument("--save-episodes", metavar="DIR", help="one workbench .npz per episode")
    parser.add_argument("--workers", type=int, default=1, help="processes, one env each")
    parser.add_argument("--device", help="override training.device")
    parser.add_argument("--out", help="write per-episode results and the summary as JSON")
    args = parser.parse_args()

    overrides = target_split_overrides(args.config, args.target_split) + list(args.set)
    config = load_config(args.config, overrides)
    first = args.seed if args.seed is not None else int(config["evaluation"]["seed"])
    count = args.episodes or int(config["evaluation"]["episodes"])
    jobs = [(k, first + k) for k in range(count)]
    options = {"device": args.device, "freeze_catheter": args.freeze_catheter,
               "save_dir": args.save_episodes,
               "description": f"{Path(args.checkpoint).name} · {Path(args.config).name}"}
    split = next((o.split("=", 1)[1] for o in reversed(overrides)
                  if o.startswith("env.scene.target.split=")), None)
    print(f"evaluating {count} episodes (seeds {first}-{first + count - 1})"
          + (f", target split {split}" if split else "")
          + (", catheter frozen" if args.freeze_catheter else "") + f", {args.workers} process(es)",
          flush=True)
    if args.workers > 1:
        records = evaluate_parallel(config, args.checkpoint, jobs, options, args.workers)
    else:
        records = evaluate_seeds(config, args.checkpoint, jobs, options)

    frequency = float(config["env"]["scene"]["fluoroscopy"].get("image_frequency", 7.5))
    summary = report(records, step_seconds=1.0 / frequency)
    summary.update(checkpoint=str(args.checkpoint), config=str(args.config), seeds=[first, first + count - 1],
                   target_split=split, freeze_catheter=bool(args.freeze_catheter))
    print("summary:", json.dumps(summary, indent=1))
    print(table(summary))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"summary": summary, "episodes": records}, handle, indent=1)


if __name__ == "__main__":
    main()
