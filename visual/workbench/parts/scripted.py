"""Scripted controllers as workbench policies (two-device environments only).

Two versions of ``StagedController``, each under its own menu name:

- "手写推法·11-06（真实质量）": ``guidewire_rl.env.two_device.script`` (the 2026-11-06 retune for the
  real-mass setting, on to the right internal carotid when the target lies past the carotid's split:
  ``configs/two_devices_ica.yaml``);
- "手写推法·第一轮（重质量）": ``guidewire_rl.env.two_device.script_round1``, the first training
  round's version (commit e908bee, tuned on ``configs/two_devices_round1.yaml`` with stEVE's heavy
  default masses).

A scripted controller reads the simulator's 3D truth itself (``act(env)``), so the policy holds the
live environment; it is listed for two-instrument environments only, and only where its module
imports.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

import numpy as np

from .policies import _Base

LABEL = "手写推法·11-06（真实质量）"
LABEL_ROUND1 = "手写推法·第一轮（重质量）"
MODULES = {LABEL: "guidewire_rl.env.two_device.script",
           LABEL_ROUND1: "guidewire_rl.env.two_device.script_round1"}


class ScriptedPolicy(_Base):
    def __init__(self, env, controller, label: str = LABEL) -> None:
        self.env, self.controller, self.label = env, controller, label

    def reset(self) -> None:
        self.controller.reset(self.env)

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        return np.asarray(self.controller.act(self.env), dtype=np.float32)

    def debug(self) -> Dict[str, Any]:
        c = self.controller
        out: Dict[str, Any] = {"stage": f"{c.stage}:{c.substage}" if c.substage else str(c.stage),
                               "tries": int(getattr(c, "tries", 0))}
        extra = getattr(c, "last_extra", None) or {}
        out.update({k: v for k, v in extra.items() if v is None or isinstance(v, (int, float, str, bool))})
        return out


def staged_factories(source) -> Dict[str, Callable[[], ScriptedPolicy]]:
    """Menu name -> factory of each scripted controller's policy for this live source; empty where
    none can run."""
    if int(getattr(source, "action_dim", 0)) != 4:
        return {}
    out: Dict[str, Callable[[], ScriptedPolicy]] = {}
    for label, module in MODULES.items():
        try:
            controller = importlib.import_module(module).StagedController
        except Exception:
            continue
        out[label] = lambda controller=controller, label=label: ScriptedPolicy(source.env, controller(), label)
    return out
