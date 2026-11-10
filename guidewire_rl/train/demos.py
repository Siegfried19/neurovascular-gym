"""Scripted demonstration episodes into the replay (``training.demos``).

    training:
      demos: {dir: runs/demos_round1, fraction: 0.08}

At the start of a run (and on a resume whose replay has no demonstrations yet)
every ``episode_*.npz`` in ``dir`` (relative paths are taken from the workspace
root) goes into the replay as an ordinary episode, flagged as a demonstration.
``fraction`` is the **minimum share of every training batch** drawn from them
(``ceil(fraction * batch_size)`` slices; see ``replay.py``); they are never
dropped when the replay is full. No directory, or no files in it: the run
starts without demonstrations and says so.

File layout (one episode of ``T`` steps; recorded by ``scripts/record_demos.py``
with the same env config as the training, so observations and rewards match):

| key | shape | |
| --- | --- | --- |
| ``obs`` (or ``observations``) | (T+1, obs_dim) float | observation after reset, then after every step |
| ``action`` (or ``actions``) | (T, act_dim) float | the normalized [-1, 1] action sent to the env |
| ``reward`` (or ``rewards``) | (T,) float | |
| ``phase`` (or ``phases``) | (T+1,) int | phase at each observation (optional: 0) |
| ``terminated`` | (T,) bool | |
| ``truncated`` | (T,) bool | optional, not needed by the replay |
| ``labels/<name>`` (or ``label_<name>``) | (T,) float | every label the env defines (optional: 0, with a warning) |
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from .. import WORKSPACE_ROOT


def _first(data, *names: str):
    for name in names:
        if name in data.files:
            return data[name]
    return None


def resolve_dir(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def read_demo(path: Path, label_names) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        obs = _first(data, "obs", "observations")
        action = _first(data, "action", "actions")
        reward = _first(data, "reward", "rewards")
        terminated = _first(data, "terminated", "terminals")
        if obs is None or action is None or reward is None or terminated is None:
            raise ValueError(f"{path}: needs obs, action, reward and terminated "
                             f"(has {sorted(data.files)})")
        phase = _first(data, "phase", "phases")
        labels, missing = {}, []
        for name in label_names:
            value = _first(data, f"labels/{name}", f"label_{name}")
            if value is None:
                missing.append(name)
                value = np.zeros(len(action), dtype=np.float32)
            labels[name] = value
        return {"observations": obs, "actions": action, "rewards": reward,
                "terminated": terminated, "phase": phase, "labels": labels, "missing": missing}


def load_demos(replay, spec: Optional[Mapping[str, Any]], *,
               say: Callable[[str], None] = print) -> Dict[str, Any]:
    """Put the demonstration files of ``spec`` into ``replay``; returns what happened."""
    if not spec or not spec.get("dir"):
        return {"loaded": 0, "reason": "not configured"}
    folder = resolve_dir(spec["dir"])
    fraction = float(spec.get("fraction", 0.0))
    if replay.num_demo_episodes:
        return {"loaded": 0, "reason": "replay already holds demonstrations",
                "episodes": replay.num_demo_episodes}
    files = sorted(folder.glob("episode_*.npz")) if folder.is_dir() else []
    if not files:
        why = "no such directory" if not folder.is_dir() else "no episode_*.npz in it"
        say(f"demos: {folder}: {why}; training without demonstrations")
        return {"loaded": 0, "reason": why, "dir": str(folder)}
    transitions, short, missing_labels = 0, 0, set()
    for path in files:
        demo = read_demo(path, replay.label_names)
        obs = np.asarray(demo["observations"])
        if obs.ndim != 2 or obs.shape[1] != replay.observation_dim:
            raise ValueError(f"{path}: observations are {obs.shape[1:]} but the env gives "
                             f"({replay.observation_dim},); re-record the demos with this config")
        if np.asarray(demo["actions"]).shape[1:] != (replay.action_dim,):
            raise ValueError(f"{path}: actions are {np.asarray(demo['actions']).shape[1:]} but the "
                             f"env takes ({replay.action_dim},)")
        replay.add(obs, demo["actions"], demo["rewards"], demo["terminated"], demo["labels"],
                   phase=demo["phase"], demo=True)
        transitions += len(demo["actions"])
        short += int(len(demo["actions"]) < replay.horizon)
        missing_labels.update(demo["missing"])
    per_batch = int(np.ceil(fraction * replay.batch_size - 1e-9))
    say(f"demos: {len(files)} episodes, {transitions} transitions from {folder}; "
        f"each batch draws at least {per_batch} of {replay.batch_size} slices from them")
    if missing_labels:
        say(f"demos: WARNING the files lack label(s) {sorted(missing_labels)}; filled with 0")
    if short:
        say(f"demos: {short} episode(s) shorter than the horizon ({replay.horizon}) are never sampled")
    return {"loaded": len(files), "transitions": transitions, "dir": str(folder),
            "fraction": fraction, "per_batch": per_batch,
            "missing_labels": sorted(missing_labels)}
