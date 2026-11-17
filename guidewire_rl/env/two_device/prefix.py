"""The scripted prefix of a start-curriculum episode (work log 2026-11-09 22:22, 2026-11-10 14:56).

"现跑交接": after an ordinary reset the hand-written procedure (``script.StagedController``) drives
the just-reset env, step by ordinary step (shell, referee, labels, reward materials all in; the
steps go into the replay as demonstration data, source ``prefix``), until the referee's
``progress_ratio`` (0 at the insertion point, 1 with both devices at their goals) reaches the ratio
the curriculum drew (or, with the landmark grid, until the referee's ``landmark_grid`` reaches the
drawn grid: ``key``); then the policy takes over. The script giving up hands over where it stopped
(user decision 2026-11-09 14:08); a step the trainers would drop (simulation error, wall breach)
ends the prefix as it would end any episode; the episode ending inside the prefix is reported too.

``run_prefix`` is called inside the env's process (``train/env_pool.py`` worker, or the single-env
trainer): the script reads the simulator's 3D truth through ``env``. The other supply, restoring a
saved state from a snapshot library (``snapshot.py``), is the speed line's; both give the trainer the
same thing: an episode under way, with the ratio it is at.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

HANDOVER, GAVE_UP, ENDED, SIM_ERROR, WALL_BREACH, OUT_OF_STEPS = (
    "handover", "gave_up", "ended", "simulation_error", "wall_breach", "out_of_steps")


def run_prefix(env, ratio: float, *, max_steps: int = 600, controller=None,
               small_info: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None,
               first_info: Optional[Mapping[str, Any]] = None,
               key: str = "progress_ratio") -> Dict[str, Any]:
    """Drive ``env`` (just reset) with the script until ``info[key] >= ratio`` (``key``:
    ``progress_ratio``, or ``landmark_grid`` with ``ratio`` the grid to reach).

    Returns ``transitions`` (one dict per kept step: ``action``, ``reward``, ``terminated``,
    ``truncated``, ``observation`` (after the step), ``info`` (small)), ``observation`` and ``info``
    of the state handed over (the last kept step's), ``outcome`` (``handover`` / ``gave_up`` /
    ``ended`` / ``out_of_steps`` / ``simulation_error`` / ``wall_breach``: the last two mean the
    offending step was not kept and the episode is to be closed as the trainers close such steps),
    ``ratio`` reached and ``steps`` taken."""
    small = small_info or (lambda info: dict(info))
    ctl = controller if controller is not None else getattr(env, "prefix_controller", None)
    if ctl is None:
        from .script import StagedController
        ctl = StagedController()
    ctl.reset(env)
    transitions: List[Dict[str, Any]] = []
    info: Dict[str, Any] = dict(first_info or {})
    observation = None
    reached = float(info.get(key, 0.0))
    outcome = OUT_OF_STEPS
    for _ in range(int(max_steps)):
        if reached >= ratio:
            outcome = HANDOVER
            break
        if getattr(ctl, "failed", False):
            outcome = GAVE_UP
            break
        action = np.asarray(ctl.act(env), dtype=np.float32)
        observation, reward, terminated, truncated, full = env.step(action)
        if full.get("simulation_error"):
            outcome = SIM_ERROR
            break
        if full.get("wall_breached"):
            outcome = WALL_BREACH
            break
        info = small(full)
        transitions.append({"action": action, "reward": float(reward), "terminated": bool(terminated),
                            "truncated": bool(truncated), "observation": observation, "info": info})
        reached = float(full.get(key, reached))
        if terminated or truncated:
            outcome = ENDED
            break
    else:
        outcome = HANDOVER if reached >= ratio else OUT_OF_STEPS
    return {"transitions": transitions, "observation": observation, "info": info, "outcome": outcome,
            "ratio": reached, "steps": len(transitions), "ratio_asked": float(ratio), "key": key,
            "script_stage": getattr(ctl, "stage", None),
            "script_failure": getattr(ctl, "failure_reason", "") if getattr(ctl, "failed", False) else ""}
