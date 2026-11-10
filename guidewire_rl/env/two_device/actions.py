"""The safety shell between the policy and the two devices (todo_list/two_devices_training.md,
section 2), and the optional script suggestion.

``TwoDeviceGuarded`` is the env's action map (``env.action``): the policy gives four numbers in
[-1, 1] (wire push, wire turn, catheter push, catheter turn), scaled by the instruments' velocity
limits to mm/s and rad/s, and the shell passes them through these stages in this order:

1. ``speed``: wire 40 mm/s and pi rad/s, catheter 25 mm/s and pi/2 rad/s (``speed_limits``);
2. ``residual``: off by default (``residual_phases: []``; the planner keeps the policy near the
   script's suggestion itself, ``planning.residual``). When on, each number is kept within
   ``residual_delta`` of the suggestion in the listed phases;
3. ``hard_rule``: no fast bare catheter advance. With the catheter tip in a branch (anywhere but the
   aorta) and more than ``bare_push_d_mm`` (10 mm) ahead of the wire (d < -10), a catheter push
   above ``bare_push_max`` (0.15 of its limit, 3.75 mm/s) is cut to 0.15: slow feeding over the
   wire (the lab script's 3 mm/s) passes, a fast push with a bare tip does not. Counted as the
   ``bare_advance`` event when it cuts;
4. ``force``: a device whose tip contact force last step was above its limit (``force_limits_N``:
   wire, catheter) may not push; in P1 the catheter's limit is ``force_limit_hook_factor`` (2) times
   larger. The wire's limit only applies while it leads (inside the catheter its "tip force" is the
   catheter's). Forces come from the env's labels (``force_labels``), measured last step;
5. ``device_limit``: stEVE's own limits, applied here so stage 6 predicts right (``MonoPlaneStatic``
   drops a push or pull that would end at or below 0 mm or at or past the device's length);
6. same-length guard (``protector_mm`` = 2.5 mm): the two tips never end a step within 2.5 mm of the
   same insertion length (predicted with the insertion lengths + push x 1/7.5 s). A step that would
   cross over is completed to 2.5 mm on the other side by speeding up, within its own limits, the
   device(s) already moving that way (``protector_cross``, not penalized); the other device is never
   moved. If that is not enough the step stops at the band's near edge, by slowing down what moves
   towards it (``protector_clamp``, penalized). Inside the band (at the start both are at 0) a step
   may only leave it. The band is 5 mm wide, less than the wire's 5.3 mm per step, so the wire can
   cross on its own; the catheter (3.3 mm per step) cannot.

The command that comes out is what the simulator executes; stEVE's ``MonoPlaneStatic`` stores it as
``intervention.last_action``, which the ``LastAction`` observation reads. ``ShieldInfo`` puts in
``info``, in normalized units (each axis / its velocity limit): ``policy_action``,
``executed_action`` (from ``last_action``), ``shield_delta`` = executed - policy,
``nominal_action`` / ``nominal_available`` (the script's suggestion for the *next* step, zeros
without one), ``shield_penalized`` (what ``ShieldPenalty`` charges: L1 size of the changes made by
the penalized stages ``speed``, ``hard_rule``, ``force``, ``protector_clamp``), ``shield_stages``
(the stages that changed the command) and ``shield_causes`` (every stage name -> 1 if it changed
the command this step, else 0; fixed keys for counting).
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from eve.info import Info

from ..registry import register
from .phases import CATH, WIRE, PhaseDetector

STAGES = ("speed", "residual", "hard_rule", "force", "device_limit", "protector_cross",
          "protector_clamp")
DEFAULT_FORCE_LIMITS_N = (1.0, 1.0)   # placeholders; the config sets measured ones
PENALIZED_STAGES = ("speed", "hard_rule", "force", "protector_clamp")
OUT_OF_BRANCH = ("aorta", "aorta_inlet")       # where the catheter tip is not "in a branch"


@dataclass
class ShieldRecord:
    """One step of the shell, normalized units (each axis / its velocity limit)."""

    policy: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    nominal: Optional[np.ndarray] = None
    commanded: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    penalized: float = 0.0
    stages: List[str] = field(default_factory=list)


@register("action")
class TwoDeviceGuarded:
    """Policy action -> executed command through the safety shell (module docstring)."""

    def __init__(self, intervention, phase_detector: PhaseDetector, labels: Sequence[Any] = (),
                 nominal_advisor=None,
                 speed_limits: Sequence[Sequence[float]] = ((40.0, np.pi), (25.0, np.pi / 2)),
                 residual_delta: float = 0.5, residual_phases: Sequence[int] = (),
                 bare_push_d_mm: float = -10.0, bare_push_max: float = 0.15,
                 force_limits_N: Sequence[float] = DEFAULT_FORCE_LIMITS_N,
                 force_limit_hook_factor: float = 2.0, hook_phase: int = 1,
                 force_labels: Optional[Sequence[str]] = ("distal_force", "catheter_tip_force"),
                 wire_force_only_when_leading: bool = True,
                 protector_mm: float = 2.5, control_hz: Optional[float] = None,
                 branch_push_max: Optional[float] = None, wire_branch_push_max: Optional[float] = None,
                 penalized_stages: Sequence[str] = PENALIZED_STAGES) -> None:
        self.intervention = intervention
        self.phase_detector = phase_detector
        self.labels = list(labels)
        self.nominal_advisor = nominal_advisor
        self.speed_limits = np.asarray(speed_limits, dtype=np.float64).reshape(-1, 2)
        self.residual_delta = float(residual_delta)
        self.residual_phases = tuple(int(p) for p in residual_phases)
        self.bare_push_d_mm = float(bare_push_d_mm)
        self.bare_push_max = float(bare_push_max)
        limits = np.broadcast_to(np.asarray(force_limits_N, dtype=np.float64), (2,))
        self.force_limits_N = tuple(float(x) for x in limits)
        self.force_limit_hook_factor = float(force_limit_hook_factor)
        self.hook_phase = int(hook_phase)
        self.force_labels = tuple(force_labels) if force_labels else ()
        self.wire_force_only_when_leading = bool(wire_force_only_when_leading)
        self.protector_mm = float(protector_mm)
        # In a branch, pushes are capped (fraction of the device's limit): a catheter pushed at
        # full speed against the trunk's end stores energy in the arch and lurches 15-35 mm into
        # the carotid, through its wall (run A2, 2026-10-31: 41% of P4 episodes ended with the
        # catheter outside the vessel). None = no cap.
        self.branch_push_max = None if branch_push_max is None else float(branch_push_max)
        self.wire_branch_push_max = None if wire_branch_push_max is None else float(wire_branch_push_max)
        self.control_hz = control_hz
        unknown = [s for s in penalized_stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; stages are {STAGES}")
        self.penalized_stages = tuple(penalized_stages)
        self.last = ShieldRecord()
        self._labels_by_name: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- action map ----
    def space(self, raw_low: np.ndarray, raw_high: np.ndarray) -> gym.spaces.Box:
        shape = np.asarray(raw_low).reshape(-1).shape
        return gym.spaces.Box(low=-np.ones(shape, dtype=np.float32),
                              high=np.ones(shape, dtype=np.float32), dtype=np.float32)

    def to_raw(self, action: np.ndarray, raw_low: np.ndarray, raw_high: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float64).reshape(-1)
        if normalized.shape != np.asarray(raw_low).reshape(-1).shape:
            raise ValueError(f"Expected action shape {np.asarray(raw_low).shape}, got {normalized.shape}.")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Action contains NaN or infinity")
        low = np.asarray(raw_low, dtype=np.float64).reshape(-1)
        high = np.asarray(raw_high, dtype=np.float64).reshape(-1)
        policy = np.clip(normalized, -1.0, 1.0)
        raw = low + 0.5 * (policy + 1.0) * (high - low)
        executed, record = self.guard(raw, scale=high, policy=policy)
        self.last = record
        return executed.astype(np.float32)

    # --------------------------------------------------------------------- stages ----
    def guard(self, raw: np.ndarray, scale: np.ndarray, policy: np.ndarray):
        """The five stages on a raw command [v_w, w_w, v_c, w_c] (mm/s, rad/s). ``scale`` is the
        velocity limit per axis (normalization). Returns (executed raw, ``ShieldRecord``)."""
        state = self.phase_detector.state
        scale = np.where(np.abs(scale) > 0, np.abs(scale), 1.0)
        command = np.asarray(raw, dtype=np.float64).copy()
        penalized = 0.0
        stages: List[str] = []

        def apply(stage: str, new: np.ndarray) -> None:
            nonlocal command, penalized
            if np.any(np.abs(new - command) > 1e-6 * scale):     # round-off is not an intervention
                stages.append(stage)
                if stage in self.penalized_stages:
                    penalized += float(np.sum(np.abs(new - command) / scale))
            command = new

        # 1. speed limit; in a branch the pushes are capped (``branch_push_max`` of the catheter's
        # limit, ``wire_branch_push_max`` of the wire's)
        limits = np.tile(self.speed_limits, (len(command) // 2 // len(self.speed_limits) + 1, 1))
        limits = limits[: len(command) // 2].reshape(-1)
        upper = limits.copy()
        in_branch_c = bool(state.branch_c) and state.branch_c not in OUT_OF_BRANCH
        in_branch_w = bool(state.branch_w) and state.branch_w not in OUT_OF_BRANCH
        if self.branch_push_max is not None and in_branch_c:
            upper[2 * CATH] = min(upper[2 * CATH], self.branch_push_max * scale[2 * CATH])
        if self.wire_branch_push_max is not None and in_branch_w:
            upper[2 * WIRE] = min(upper[2 * WIRE], self.wire_branch_push_max * scale[2 * WIRE])
        apply("speed", np.clip(command, -limits, upper))

        # 2. residual box around the script's suggestion
        nominal = None
        if self.nominal_advisor is not None:
            nominal = self.nominal_advisor.action
        if nominal is not None:
            nominal = np.clip(np.asarray(nominal, dtype=np.float64).reshape(-1), -1.0, 1.0)
            if state.phase in self.residual_phases:
                boxed = np.clip(command / scale, nominal - self.residual_delta,
                                nominal + self.residual_delta)
                apply("residual", np.clip(boxed, -1.0, 1.0) * scale)

        # 3. hard rule: no fast bare catheter advance in a branch
        cath_v = 2 * CATH
        in_branch = bool(state.branch_c) and state.branch_c not in OUT_OF_BRANCH
        no_push = [False, False]
        if in_branch and state.d < self.bare_push_d_mm:
            cap = self.bare_push_max * scale[cath_v]
            if command[cath_v] > cap + 1e-6 * scale[cath_v]:
                new = command.copy()
                new[cath_v] = cap
                apply("hard_rule", new)
                self.phase_detector.report("bare_advance")

        # 4. force limit from last step's tip forces
        new = command.copy()
        for device, force in self._last_forces().items():
            limit = self.force_limits_N[device]
            if device == CATH and state.phase == self.hook_phase:
                limit *= self.force_limit_hook_factor
            if device == WIRE and self.wire_force_only_when_leading and state.d <= 0:
                continue
            if force is not None and force > limit:
                no_push[device] = True
                new[2 * device] = min(new[2 * device], 0.0)
        apply("force", new)

        # 5. stEVE's own limits (MonoPlaneStatic drops a push or pull that would end at or below
        # 0 mm or at or past the device's length); applied here so the guard predicts right
        T = self._step_seconds()
        maximum = getattr(self.intervention, "instrument_lengths_maximum", None)
        new = command.copy()
        for device, inserted in ((WIRE, state.L_wire), (CATH, state.L_cath)):
            end = inserted + new[2 * device] * T
            if end <= 0.0 or (maximum is not None and end >= float(maximum[device])):
                new[2 * device] = 0.0
        apply("device_limit", new)

        # 6. same-length guard. Bounds for speeding a device up: its speed limit, no push the
        # stages above forbid, no faster catheter push in a branch, no pull below 0 mm, no push
        # past the device's length
        bounds = []
        for device, inserted in ((WIRE, state.L_wire), (CATH, state.L_cath)):
            lim = float(limits[2 * device])
            high = 0.0 if no_push[device] else float(upper[2 * device])
            if device == CATH and in_branch:
                high = min(high, max(float(command[cath_v]), 0.0))
            if maximum is not None:
                high = min(high, max(float(maximum[device]) - inserted, 0.0) / T * (1.0 - 1e-6))
            low = -min(lim, max(inserted, 0.0) / T * (1.0 - 1e-6))
            bounds.append((low, high))
        crossed, clamped = self.protect(command, state.L_wire, state.L_cath, self.protector_mm, T, bounds)
        if crossed is not None:
            apply("protector_cross", crossed)
        if clamped is not None:
            apply("protector_clamp", clamped)

        record = ShieldRecord(policy=policy.astype(np.float32),
                              nominal=None if nominal is None else nominal.astype(np.float32),
                              commanded=(command / scale).astype(np.float32),
                              penalized=penalized, stages=stages)
        return command, record

    def _last_forces(self) -> Dict[int, Optional[float]]:
        if not self.force_labels:
            return {}
        if self._labels_by_name is None:
            self._labels_by_name = {getattr(l, "name", None): l for l in self.labels}
        out: Dict[int, Optional[float]] = {}
        for device, name in enumerate(self.force_labels[:2]):
            label = self._labels_by_name.get(name)
            if label is None:
                raise KeyError(f"TwoDeviceGuarded: force label {name!r} is not in env.labels "
                               f"({sorted(n for n in self._labels_by_name if n)}); set force_labels")
            out[device] = getattr(label, "last_value", None)
        return out

    def _step_seconds(self) -> float:
        hz = self.control_hz or getattr(self.intervention.fluoroscopy, "image_frequency", 7.5)
        return 1.0 / float(hz)

    @classmethod
    def protect(cls, command: np.ndarray, L_wire: float, L_cath: float, delta: float, T: float,
                bounds):
        """Stage 6 on a raw command: (crossing completed, clamped), each None if not needed.
        ``bounds``: ((low, high) of the wire push, (low, high) of the catheter push), mm/s."""
        vw, vc = float(command[2 * WIRE]), float(command[2 * CATH])
        d = L_wire - L_cath
        d_next = d + (vw - vc) * T
        if abs(d_next) >= delta and (abs(d) >= delta or abs(d_next) >= abs(d)):
            return None, None                 # ends outside the band (or leaves it)
        side = np.sign(d) if d != 0 else np.sign(d_next)
        if side == 0:
            return None, None                 # both standing at the same length (the start)
        crossing = np.sign(d_next) != side
        if not crossing and abs(d) < delta and abs(d_next) >= abs(d):
            return None, None                 # inside the band, on its way out
        if crossing:
            done = cls.speed_up(command, d, -side * delta, T, bounds)
            if done is not None:
                return done, None
        edge = side * delta if abs(d) >= delta else d     # the band's near edge, or stay
        return None, cls.slow_down(command, d, edge, T)

    @staticmethod
    def speed_up(command: np.ndarray, d: float, d_target: float, T: float,
                 bounds) -> Optional[np.ndarray]:
        """The command with the device(s) already moving the needed way (d = wire minus catheter
        must change towards ``d_target``) sped up in their own direction, within ``bounds``, so
        that d ends at ``d_target``; a device standing still or moving the other way is not
        touched. Shared in proportion to how fast each moves that way. None if they cannot."""
        (low_w, high_w), (low_c, high_c) = bounds
        vw, vc = float(command[2 * WIRE]), float(command[2 * CATH])
        need = (d_target - d) / T - (vw - vc)        # extra relative velocity
        if abs(need) < 1e-12:
            return np.asarray(command, dtype=np.float64).copy()
        s = 1.0 if need > 0 else -1.0
        with_w = max(s * vw, 0.0)                    # the wire moving the needed way
        with_c = max(-s * vc, 0.0)                   # the catheter moving the needed way
        room_w = (max(high_w - vw, 0.0) if vw > 0 else max(vw - low_w, 0.0)) if with_w > 0 else 0.0
        room_c = (max(high_c - vc, 0.0) if vc > 0 else max(vc - low_c, 0.0)) if with_c > 0 else 0.0
        amount = abs(need)
        if room_w + room_c < amount - 1e-9:
            return None
        share_w = with_w / (with_w + with_c)
        give_w = min(room_w, amount * share_w)
        give_c = min(room_c, amount - give_w)
        give_w = min(room_w, amount - give_c)
        new = np.asarray(command, dtype=np.float64).copy()
        new[2 * WIRE] = vw + s * give_w
        new[2 * CATH] = vc - s * give_c
        return new

    @staticmethod
    def slow_down(command: np.ndarray, d: float, d_target: float, T: float) -> np.ndarray:
        """The command with what moves d away from ``d_target`` slowed down (in proportion to its
        share of that motion, never past standing still), so that d ends at ``d_target``; a device
        moving the other way is not touched. Assumes ``d_target`` lies between d and where the
        command would take it (then slowing down is always enough)."""
        vw, vc = float(command[2 * WIRE]), float(command[2 * CATH])
        need = (d_target - d) / T - (vw - vc)
        new = np.asarray(command, dtype=np.float64).copy()
        if abs(need) < 1e-12:
            return new
        s = 1.0 if need > 0 else -1.0
        against_w = max(-s * vw, 0.0)
        against_c = max(s * vc, 0.0)
        total = against_w + against_c
        if total <= 1e-12:
            return new
        cut = min(abs(need), total)
        new[2 * WIRE] = vw + s * cut * against_w / total
        new[2 * CATH] = vc - s * cut * against_c / total
        return new


@register("monitor")
class NominalAdvisor:
    """The script's suggestion for the next step (normalized, 4 numbers), or None.

    ``controller``: ``{type: "package.module.Class", ...params}`` built with the scene objects it
    asks for by name (``intervention``, ``phase_detector``); each step after the referee, its
    ``method`` (``nominal_for_phase``) is called with the referee's ``PhaseState`` and should return
    4 numbers in [-1, 1] or None. Without a controller the suggestion is None (the residual box is
    skipped, ``ScriptSuggestion`` shows zeros)."""

    def __init__(self, intervention, phase_detector: PhaseDetector,
                 controller: Optional[Mapping[str, Any]] = None, method: str = "nominal_for_phase",
                 name: str = "nominal_advisor") -> None:
        self.name = name
        self.intervention = intervention
        self.phase_detector = phase_detector
        self.method = method
        self.controller = None
        if controller:
            spec = dict(controller)
            module_name, class_name = spec.pop("type").rsplit(".", 1)
            cls = getattr(importlib.import_module(module_name), class_name)
            params = inspect.signature(cls.__init__).parameters
            for dep, obj in (("intervention", intervention), ("phase_detector", phase_detector)):
                if dep in params and dep not in spec:
                    spec[dep] = obj
            self.controller = cls(**spec)
        self.action: Optional[np.ndarray] = None

    def _ask(self) -> Optional[np.ndarray]:
        if self.controller is None:
            return None
        out = getattr(self.controller, self.method)(self.phase_detector.state)
        if out is None:
            return None
        return np.clip(np.asarray(out, dtype=np.float64).reshape(-1), -1.0, 1.0)

    def reset(self, episode_nr: int = 0) -> None:
        if self.controller is not None and hasattr(self.controller, "reset"):
            try:
                self.controller.reset(episode_nr)
            except TypeError:
                self.controller.reset()
        self.action = self._ask()

    def step(self) -> None:
        self.action = self._ask()


@register("info")
class ShieldInfo(Info):
    """The shell's step in ``info`` (module docstring). ``nominal_action`` is the suggestion for the
    *next* step (``nominal_advisor``'s current one, what the planner needs)."""

    def __init__(self, intervention, action_map: TwoDeviceGuarded, nominal_advisor=None,
                 name: str = "shield") -> None:
        super().__init__(name)
        self.intervention = intervention
        self.action_map = action_map
        self.nominal_advisor = nominal_advisor

    @property
    def info(self) -> Dict[str, Any]:
        record = self.action_map.last
        limits = np.abs(np.asarray(self.intervention.velocity_limits, dtype=np.float64)).reshape(-1)
        limits = np.where(limits > 0, limits, 1.0)
        executed = np.asarray(self.intervention.last_action, dtype=np.float64).reshape(-1) / limits
        policy = np.asarray(record.policy, dtype=np.float64)
        advisor = self.nominal_advisor if self.nominal_advisor is not None else \
            self.action_map.nominal_advisor
        nominal = None if advisor is None else advisor.action
        return {
            "policy_action": policy.astype(np.float32),
            "executed_action": executed.astype(np.float32),
            "shield_delta": (executed - policy).astype(np.float32),
            "nominal_action": (np.zeros_like(policy) if nominal is None else
                               np.asarray(nominal, dtype=np.float64).reshape(-1)).astype(np.float32),
            "nominal_available": nominal is not None,
            "shield_penalized": float(record.penalized),
            "shield_stages": list(record.stages),
            "shield_causes": {stage: int(stage in record.stages) for stage in STAGES},
        }

    def reset(self, episode_nr: int = 0) -> None:
        self.action_map.last = ShieldRecord()
