"""器械仿真: the chosen guidewire and catheter pushed into empty space, live (no vessel).

Opened as the phantom "无血管：器械仿真" (``environments.PHANTOMS["bench"]``). The instruments are the
setting's, or the ones picked in the guidewire / catheter menus (``environments.with_devices``, from
``guidewire_rl.env.instruments.catalog``); the physics (step, alarm distance, masses ...) are the
setting's. ``BenchSource`` steps ``guidewire_rl.env.instruments.free_space.FreeSpace`` with a policy,
as ``LiveEnvSource`` steps an environment: the observation is each device's inserted length and
turn, the action each device's push and turn speed in [-1, 1] of its limits.

``ThroughDemo`` is the "wire through the catheter" run: the catheter goes out with the wire kept
behind its head, then the wire is pushed through and out of the tip, pulled back, and again.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .frame import Frame, SceneInfo, inserted_wire
from .instrument import instrument_info
from .policies import Policy

BENCH_KEY = "bench:free_space"       # SceneInfo.vessel_key of the bench (the grid layer draws on it)
AXIS_MM = 250.0                       # length of the drawn insertion axis
# at turn 0 the shaped ends bend in the x-y plane, towards +y (measured): the "fluoroscope" looks down
# the -z axis at it, +x to the right and +y up on the screen
BENCH_VIEW_DEG = (0.0, -90.0)
DEMO_LABEL = "演示：导丝穿过导管"


class BenchSource:
    """Instruments in free space, stepped with ``policy`` (the ``FrameSource`` methods)."""

    can_run = True

    def __init__(self, config: Mapping[str, Any], policy: Optional[Policy] = None, description: str = "") -> None:
        from guidewire_rl.env import registry
        from guidewire_rl.env.instruments.free_space import FreeSpace

        scene = config["env"]["scene"]
        self.instruments = [registry.instantiate("instrument", spec, {}) for spec in scene["instruments"]]
        physics = {k: v for k, v in dict(scene.get("simulation") or {}).items() if k != "type"}
        self.bench = FreeSpace(self.instruments, physics)
        self.limits = np.array([[float(v) for v in inst.velocity_limit] for inst in self.instruments])
        self.policy = policy
        self.description = description
        self.episode = -1
        self._step = 0
        self._seed: Optional[int] = None
        self._info: Optional[SceneInfo] = None

    @property
    def n(self) -> int:
        return len(self.instruments)

    @property
    def observation_dim(self) -> int:
        return 2 * self.n

    @property
    def action_dim(self) -> int:
        return 2 * self.n

    @property
    def step_seconds(self) -> float:
        return self.bench.step_seconds

    def head_length_mm(self, index: int) -> float:
        """Length of an instrument's shaped end (its tip section), mm."""
        return float(self.instruments[index].tip_section.length)

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy

    def scene_info(self) -> SceneInfo:
        if self._info is None:
            self._info = bench_scene_info(self.instruments, self.limits, self.step_seconds, self.description)
        return self._info

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.bench.inserted_lengths, self.bench.rotations]).astype(np.float32)

    def _frame(self, **fields) -> Frame:
        return Frame(episode=self.episode, step=self._step, obs=self._obs(),
                     wire_mm=inserted_wire(self.bench.dof_positions),
                     inserted_mm=self.bench.inserted_lengths.astype(np.float32),
                     rotation_rad=self.bench.rotations.astype(np.float32), seed=self._seed, **fields)

    def reset(self, seed: Optional[int] = None) -> Frame:
        self.bench.reset(seed=seed)
        if self.policy is not None:
            self.policy.reset()
        self.episode += 1
        self._step, self._seed = 0, seed
        return self._frame()

    def step(self) -> Frame:
        if self.policy is None:
            raise RuntimeError("Set a policy and call reset() first")
        obs = self._obs()
        action = np.clip(np.asarray(self.policy.act(obs, first_step=self._step == 0), dtype=np.float32), -1, 1)
        raw = action.reshape(self.n, 2) * self.limits
        extras: Dict[str, Any] = dict(self.policy.debug())
        truncated = False
        try:
            self.bench.step(raw)
        except Exception as exc:  # SOFA failure inside the step: end the run, keep the view
            extras["error"], truncated = repr(exc), True
        if self.bench.simulation_error:
            extras["error"], truncated = "simulation_error", True
        self._step += 1
        return self._frame(action=action, raw_action=raw.reshape(-1).astype(np.float32),
                           truncated=truncated, extras=extras)

    def close(self) -> None:
        self.bench.close()


def bench_scene_info(instruments: Sequence, limits: np.ndarray, step_seconds: float,
                     description: str = "") -> SceneInfo:
    """No vessel (one point-sized triangle); the insertion axis as the one 'centerline'."""
    n = len(instruments)
    infos = []
    for instrument in instruments:
        try:
            infos.append(instrument_info(instrument))
        except Exception:  # a type the instrument window cannot read: leave it out
            pass
    high = np.asarray(limits, dtype=np.float32).reshape(-1)
    names = [f"inserted[{i}]" for i in range(n)] + [f"rotation[{i}]" for i in range(n)]
    return SceneInfo(
        vessel_vertices=np.zeros((3, 3), np.float32), vessel_faces=np.array([[0, 1, 2]], np.int32),
        centerlines={"insertion_axis": np.array([[0.0, 0.0, 0.0], [AXIS_MM, 0.0, 0.0]], np.float32)},
        insertion_position=np.zeros(3, np.float32), insertion_direction=np.array([1.0, 0.0, 0.0], np.float32),
        target_threshold_mm=0.0, image_rot_zx_deg=BENCH_VIEW_DEG,
        observation_names=names, observation_layout=[("inserted", n), ("rotation", n)],
        action_low=-high, action_high=high, max_steps=None, step_seconds=float(step_seconds),
        vessel_key=BENCH_KEY, description=description, task_branches=("insertion_axis",), instruments=infos)


class ThroughDemo:
    """Catheter out (wire ``behind_mm`` behind its head), wire through and ``out_mm`` past the tip,
    hold, wire back, hold, through again ... Closed loop on the inserted lengths (obs[:2])."""

    label = DEMO_LABEL
    PHASES = ("out", "through", "hold", "back", "rest")

    def __init__(self, limits: np.ndarray, head_mm: float, step_seconds: float, *, catheter_out_mm: float = 80.0,
                 behind_mm: float = 20.0, out_mm: float = 60.0, wire_speed: float = 15.0,
                 catheter_speed: float = 20.0, hold_steps: int = 8) -> None:
        self.limits = np.asarray(limits, dtype=np.float64).reshape(-1, 2)
        if len(self.limits) < 2:
            raise ValueError("the demo needs a guidewire and a catheter")
        self.catheter_at = float(head_mm) + catheter_out_mm
        self.wire_home = max(self.catheter_at - float(head_mm) - behind_mm, 10.0)
        self.wire_out = self.catheter_at + out_mm
        self.speeds = (float(wire_speed), float(catheter_speed))
        self.dt, self.hold_steps = float(step_seconds), int(hold_steps)
        self.reset()

    def reset(self) -> None:
        self.phase, self._held = "out", 0

    def _targets(self) -> Sequence[float]:
        wire = {"out": self.wire_home, "through": self.wire_out, "back": self.wire_home}.get(self.phase)
        return wire, self.catheter_at

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        lengths = np.asarray(obs, dtype=np.float64)[:2]
        wire, catheter = self._targets()
        if self.phase in ("hold", "rest"):
            self._held += 1
            if self._held >= self.hold_steps:
                self.phase, self._held = ("back" if self.phase == "hold" else "through"), 0
            wire = lengths[0]
        elif max(abs(wire - lengths[0]), abs(catheter - lengths[1])) < 0.5:
            self.phase = {"out": "through", "through": "hold", "back": "rest"}[self.phase]
        action = np.zeros(2 * len(self.limits), np.float32)
        for i, (target, speed) in enumerate(zip((wire, catheter), self.speeds)):
            velocity = float(np.clip((target - lengths[i]) / self.dt, -speed, speed))
            action[2 * i] = velocity / self.limits[i, 0]
        return action

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        return {}

    def debug(self) -> Dict[str, Any]:
        return {"demo_phase": self.phase}


def bench_menu(source: BenchSource, extra: Optional[Mapping] = None) -> Dict:
    """Policy menu of the bench: the demo first (with a catheter), then ``extra`` (the keyboard),
    then the built-in ones."""
    from .policies import policy_menu

    menu: Dict = {}
    if source.n >= 2:
        menu[DEMO_LABEL] = lambda: ThroughDemo(source.limits, source.head_length_mm(1), source.step_seconds)
    menu.update(extra or {})
    for label, make in policy_menu(source.action_dim, [], lambda p: None).items():
        menu.setdefault(label, make)
    return menu
