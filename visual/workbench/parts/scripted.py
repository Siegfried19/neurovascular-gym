"""Scripted controllers as workbench policies (two-device environments only).

"手写推法": ``guidewire_rl.env.two_device.script.StagedController`` with its own settings (the
2026-11-06 retune for the real-mass setting, on to the right internal carotid when the target lies
past the carotid's split: ``configs/two_devices_ica.yaml``). A scripted controller reads the
simulator's 3D truth itself (``act(env)``), so the policy holds the live environment; it is listed
for two-instrument environments only, and only where that module imports.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .policies import _Base

LABEL = "手写推法"


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


def staged_factory(source) -> Optional[Callable[[], ScriptedPolicy]]:
    """Factory of the scripted controller's policy for this live source, or None where it cannot run."""
    if int(getattr(source, "action_dim", 0)) != 4:
        return None
    try:
        from guidewire_rl.env.two_device.script import StagedController
    except Exception:
        return None
    return lambda: ScriptedPolicy(source.env, StagedController())
