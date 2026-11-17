"""The hand-written procedure in shadow: what it would do, as each step's imitation target (DAgger-style
labels; lead 2026-11-13).

``env.shadow_expert: {type: StagedController, ...}`` (absent, null or false: off; the other keys go
to the controller, e.g. ``ScriptParams`` overrides). While the POLICY drives an episode, the
controller runs alongside it inside the env (the env's process: an ``EnvPool`` worker or the
single-env trainer). It is reset at every ``env.reset`` (after the env's own reset, so also after a
snapshot start, where it then begins in ``approach`` in the middle of the procedure: its labels there
are that stage's) and, at every ``env.step(action)``, BEFORE the action is applied, asked what it would
do from the true simulator state. Its action is NOT executed: the env reports it in
``info["expert_action"]``, normalized exactly like every action the replay stores (what
``StagedController.act`` returns, the numbers a scripted prefix step stores and sends to ``env.step``),
and the trainers keep it as that step's imitation target (``training.bc_target: expert``,
``model/agent.py``).

The controller only reads the env (polylines, inserted lengths, rotations, the target, the referee's
state, the last executed action); its own state (stage machine, counters) moves on as if its action
had been executed, which it was not. How well the labels hold up off the script's path is measured,
not assumed. Measured 2026-11-13 on the 160k-step policy (runs/round2_twocrutch_v2/shadow_160k_pi,
8 seeds): the stage machine advances only while the policy does what the script would (1 episode of
8 all the way); otherwise it stays in a stage (``approach``; ``place:retract`` waiting for the wire
37 mm behind the catheter tip; ``turn`` waiting for the same) and keeps labelling that stage's
action from the current state, e.g. "wire back at full speed" on 98-100 % of the steps where the
referee's phase is already past the stage (41-81 % of the steps of four episodes). With ``sync``
(runs/round2_twocrutch_v2/shadow_sync_160k_pi, same 8 seeds): not one such label on the 876 P4 steps
short of the target in the four episodes that got there; labels on 93 % of all steps (one episode's
script gave up after 300 steps of backing the wire out of the external carotid; 2 safety-net steps).

No label (a NaN vector) on a step where the controller has given up (``failed``), has finished
(stage ``done``), or raised; ``info["expert_off"]`` says which (``gave_up:<reason>``, ``done``,
``error:<exception>``; "" with a label). A step that raised is skipped, the next step asks again.

During a scripted prefix (``prefix.py``) this controller follows the prefix's own controller step by
step: both are reset with the episode and see the same states and executed actions, so they choose
the same actions (tests/test_shadow_expert.py checks it in SOFA), and it is at the right stage when
the policy takes over. It is NOT reset at the handover: a fresh controller starts in ``approach``,
which is wrong in the middle of the procedure. The replay keeps the executed action as the target of
every scripted step anyway (``train/replay.py``).

``sync: true`` moves a lagging script forward to the stage the referee's phase implies before each
label, and drops labels chosen in a stage behind the phase (``ShadowExpert``; lead 2026-11-13).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

# short names for ``env.shadow_expert.type``; anything else is a dotted path to a class
CONTROLLERS = {"StagedController": "guidewire_rl.env.two_device.script.StagedController"}

# Syncing (``sync: true``): the script's stages in the order of the procedure (the climb up the carotid,
# ``wire:up``, after the wire stage's other substages), the stage each referee phase lets the procedure
# have reached at least (``LEAST``: the safety net), and the stage a phase moves a lagging script to
# (``ShadowExpert._target``).
ORDER = ("approach", "place", "turn", "pull", "seat", "wire", "wire:up", "ica", "follow", "trim", "done")
LEAST = {0: "approach", 1: "place", 2: "turn", 3: "seat", 4: "wire"}


def position(stage: str, substage: str = "") -> int:
    """Where a stage (and substage) is in the procedure (``ORDER``); -1 for an unknown name."""
    name = "wire:up" if (stage == "wire" and substage == "up") else stage
    return ORDER.index(name) if name in ORDER else -1


def referee_state(env):
    """The env's referee state (``env.monitors["phase_detector"].state``); None without one."""
    monitors = getattr(env, "monitors", None) or {}
    detector = monitors.get("phase_detector") if hasattr(monitors, "get") else None
    return None if detector is None else getattr(detector, "state", None)


class ShadowExpert:
    """A controller with ``reset(env)``, ``act(env)``, ``stage``, ``failed`` (``StagedController``)
    asked for its action before every step (module docstring).

    ``sync`` (``env.shadow_expert.sync``, default false; lead 2026-11-13): the script's stage moves on
    only on its own triggers (the catheter 33 mm past the trunk's mouth, the wire 37 mm behind the hook,
    ...), which a policy that does the task its own way may never set off; its labels then belong to a
    stage the procedure has left (measured: "wire back at full speed" with the wire in the carotid). With
    ``sync`` the referee's phase is compared with the stage before every label; a script that lags is
    moved FORWARD (never back) to the stage the phase says the procedure has reached, entered as the
    script's own transitions enter it (``StagedController._go``: substage, counters, the stage's budget
    from now):

    =====  =========================================================================================
    phase  stage the script is moved to when it is behind it
    =====  =========================================================================================
    P1     ``place:retract`` (catheter >= 30 mm past the trunk's mouth along the aorta; the script's
           own approach ends at 33): wire back behind the hook, then the hook placed and turned up
    P2     ``seat`` if the script reads the catheter tip >= ``engage_mm`` (2) into the trunk, else
           ``turn`` (wire behind the hook and hook up; the turn stage hands on to pull itself)
    P3     ``seat`` if the wire is still inside the catheter (d < 0) and the script reads the catheter
           in the trunk (the script pulls on to 20 mm before the wire goes out), else ``wire:out``
    P4     ``ica`` (an ICA target; ``wire:up`` for a carotid target): the script's own ica -> follow
           rule takes it on
    =====  =========================================================================================

    The geometry is the script's own readout (``script.read``), so it enters a stage only when its
    own tests agree. Safety net: a label chosen in a stage behind ``LEAST[phase]`` (the script fell
    back by its own rules, or could not be moved) is no label (``expert_off`` ``behind:<stage>@P<k>``).
    A sync is logged in the controller's log as ``sync:<from>><to>`` (so in ``expert_events``)."""

    def __init__(self, controller, sync: bool = False, readout=None) -> None:
        self.controller = controller
        self.sync = bool(sync)
        self._readout = readout          # env -> the script's Readout; default script.read

    @classmethod
    def from_config(cls, spec: Any) -> Optional["ShadowExpert"]:
        """``env.shadow_expert``: None / false / {} = off; ``sync`` is the expert's, every other key
        goes to the controller."""
        if not spec:
            return None
        spec = dict({"type": spec} if isinstance(spec, str) else spec)
        name = str(spec.pop("type", "StagedController"))
        sync = bool(spec.pop("sync", False))
        path = CONTROLLERS.get(name, name)
        if "." not in path:
            raise KeyError(f"env.shadow_expert: unknown type {name!r} (known: {sorted(CONTROLLERS)}, "
                           "or a dotted path to a class)")
        module, attr = path.rsplit(".", 1)
        return cls(getattr(importlib.import_module(module), attr)(**spec), sync=sync)

    def reset(self, env) -> None:
        self.controller.reset(env)

    def _off(self) -> str:
        ctl = self.controller
        if getattr(ctl, "failed", False):
            return f"gave_up:{getattr(ctl, 'failure_reason', '')}"
        if getattr(ctl, "stage", "") == "done":
            return "done"
        return ""

    def _read(self, env):
        if self._readout is not None:
            return self._readout(env)
        from .script import read
        ctl = self.controller
        return read(env, ctl.anatomy, ctl.hook_len)

    def _target(self, phase: int, r) -> Tuple[str, str]:
        """The stage (and substage) a lagging script is moved to in referee phase ``phase`` (class
        docstring), from the script's readout ``r``."""
        ctl = self.controller
        if phase == 1:
            return "place", "retract"
        if phase == 2:
            seated = r.cath_in_bct and r.depth_bct_cath >= ctl.p.engage_mm
            return ("seat", "") if seated else ("turn", "")
        if phase == 3:
            return ("seat", "") if (r.d < 0.0 and r.cath_in_bct) else ("wire", "out")
        return ("ica", "") if ctl._ica_task(r) else ("wire", "up")   # pylint: disable=protected-access

    def _sync(self, env, phase: int) -> None:
        """Move a lagging script forward to the stage referee phase ``phase`` implies (never back)."""
        ctl = self.controller
        if phase <= 0 or self._off():
            return
        r = self._read(env)
        stage, substage = self._target(phase, r)
        if position(stage, substage) <= position(ctl.stage, ctl.substage):
            return
        before = f"{ctl.stage}:{ctl.substage}" if ctl.substage else ctl.stage
        ctl._go(stage, substage, float(r.L[1]))       # pylint: disable=protected-access  # as on entry
        ctl.log.append((f"sync:{before}>{stage}:{substage}" if substage else f"sync:{before}>{stage}",
                        ctl.steps))

    def label(self, env, action_dim: int) -> Dict[str, Any]:
        """This step's target, asked before the env applies the policy's action: ``expert_action``
        (action_dim float32, NaN without a label), ``expert_stage`` / ``expert_substage`` (where the
        controller is after choosing), ``expert_off`` (why there is no label, "" with one) and, when
        the controller logged something this step, ``expert_events`` (its log entries: stage changes,
        syncs, ``fork:stall``, ``fork:blocked``, ...)."""
        ctl = self.controller
        logged = len(getattr(ctl, "log", ()))
        state = referee_state(env) if self.sync else None
        phase = None if state is None else int(getattr(state, "phase", 0))
        off, expert = self._off(), None
        if not off:
            try:
                if phase is not None:
                    self._sync(env, phase)
                expert = np.asarray(ctl.act(env), dtype=np.float32).reshape(-1)
            except Exception as exc:  # pylint: disable=broad-except  # no label this step, the episode goes on
                off = f"error:{exc!r}"[:300]
            else:
                off = self._off()          # gave up or finished inside this call: its zeros are no target
                if not off and (expert.shape != (action_dim,) or not np.all(np.isfinite(expert))):
                    off = f"error:bad action {expert.tolist()}"
                if not off and phase is not None and \
                        position(ctl.stage, ctl.substage) < position(LEAST.get(phase, "approach")):
                    off = f"behind:{ctl.stage}@P{phase}"      # the safety net
        out: Dict[str, Any] = {
            "expert_action": np.full(action_dim, np.nan, np.float32) if off else expert,
            "expert_stage": str(getattr(ctl, "stage", "")),
            "expert_substage": str(getattr(ctl, "substage", "")),
            "expert_off": off,
        }
        events = [str(entry[0]) for entry in list(getattr(ctl, "log", ()))[logged:]]
        if events:
            out["expert_events"] = events
        return out
