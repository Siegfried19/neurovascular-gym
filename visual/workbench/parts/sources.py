"""Where frames come from.

A ``FrameSource`` hands the viewer one ``Frame`` per step:

    scene_info() -> SceneInfo     fixed things (vessel mesh, centerlines, obs names ...)
    reset(seed) -> Frame          start an episode, frame at step 0
    step() -> Frame               advance one control step
    close()

``LiveEnvSource`` runs the YAML-configured simulator (``guidewire_rl.env``) with a
policy; ``ReplaySource`` plays an episode saved by the workbench, without SOFA;
``ShapeSource`` only shows a vessel that is not in the simulator yet (one frame, no
wire). ``can_run`` says whether stepping means anything. A future real-robot or
physical-phantom source only needs the same methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import numpy as np

from .frame import EpisodeLog, Frame, SceneInfo, inserted_wire
from .instrument import instrument_info
from .policies import Policy


class FrameSource(Protocol):
    def scene_info(self) -> SceneInfo: ...
    def reset(self, seed: Optional[int] = None) -> Frame: ...
    def step(self) -> Frame: ...
    def close(self) -> None: ...


# ----------------------------------------------------------- read the env --
def _first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def scene_info_from_env(env, description: str = "") -> SceneInfo:
    """Fixed scene data of a ``guidewire_rl`` env (upstream stEVE or Wilson's eve)."""
    import trimesh

    intervention = env.intervention
    tree = intervention.vessel_tree
    mesh_path = str(tree.mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    fluoroscopy = intervention.fluoroscopy
    frequency = float(_first_attr(fluoroscopy, "image_frequency", default=7.5))
    low, high = env.raw_action_limits
    instruments = _first_attr(intervention, "instruments", "devices", default=[]) or []
    infos = []
    for instrument in instruments:
        try:
            info = instrument_info(instrument)
        except Exception:  # an instrument type the window cannot read: leave it out
            info = None
        if info is not None:
            infos.append(info)
    return SceneInfo(
        vessel_vertices=np.asarray(mesh.vertices, dtype=np.float32),
        vessel_faces=np.asarray(mesh.faces, dtype=np.int32),
        centerlines={b.name: np.asarray(b.coordinates, dtype=np.float32) for b in tree.branches},
        insertion_position=np.asarray(tree.insertion.position, dtype=np.float32),
        insertion_direction=np.asarray(tree.insertion.direction, dtype=np.float32),
        target_threshold_mm=float(intervention.target.threshold),
        image_rot_zx_deg=tuple(float(a) for a in fluoroscopy.image_rot_zx),
        observation_names=list(env.observation_names),
        observation_layout=list(env.observation_layout),
        action_low=low, action_high=high,
        max_steps=env.max_episode_steps, step_seconds=1.0 / frequency,
        vessel_key=mesh_path,
        description=description or f"{type(tree).__name__} · {type(intervention.simulation).__name__}",
        task_branches=tuple(getattr(tree, "task_branch_names", None) or ()),
        centerline_radii={b.name: np.asarray(b.radii, dtype=np.float32) for b in tree.branches
                          if getattr(b, "radii", None) is not None},
        instruments=infos)


def target_in_vessel_frame(intervention) -> Optional[np.ndarray]:
    """stEVE keeps the target in the fluoroscope's frame (``Target.coordinates3d``: "tracking
    coordinate space", rotated by ``image_rot_zx`` about ``image_center``); the workbench draws in
    the vessel frame, as the wire."""
    target = getattr(intervention.target, "coordinates3d", None)
    if target is None:
        return None
    fluoroscopy = intervention.fluoroscopy
    try:
        from eve.util.coordtransform import tracking3d_to_vessel_cs
    except ImportError:
        return np.asarray(target, dtype=np.float64)
    return tracking3d_to_vessel_cs(np.asarray(target, dtype=np.float64), fluoroscopy.image_rot_zx,
                                   fluoroscopy.image_center)


def frame_from_env(env, *, episode: int, step: int, obs: np.ndarray, **fields) -> Frame:
    intervention = env.intervention
    target = target_in_vessel_frame(intervention)
    inserted = _first_attr(intervention, "instrument_lengths_inserted", "device_lengths_inserted")
    rotations = _first_attr(intervention, "instrument_rotations", "device_rotations")
    return Frame(
        episode=episode, step=step, obs=np.asarray(obs, dtype=np.float32),
        wire_mm=inserted_wire(intervention.simulation.dof_positions),
        target_mm=None if target is None else np.asarray(target, dtype=np.float32),
        inserted_mm=None if inserted is None else np.asarray(inserted, dtype=np.float32),
        rotation_rad=None if rotations is None else np.asarray(rotations, dtype=np.float32),
        **fields)


# --------------------------------------------------------------- sources --
class LiveEnvSource:
    """The simulator from a YAML config, stepped with ``policy``."""

    can_run = True

    def __init__(self, config: Mapping[str, Any], policy: Optional[Policy] = None,
                 description: str = "") -> None:
        from guidewire_rl.env import build_env

        self.config = config
        self.env = build_env(config["env"])
        self.policy = policy
        self.description = description
        self.episode = -1
        self._info: Optional[SceneInfo] = None
        self._obs: Optional[np.ndarray] = None
        self._step = 0
        self._return = 0.0
        self._seed: Optional[int] = None

    @property
    def observation_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.shape[0])

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy

    def scene_info(self) -> SceneInfo:
        if self._info is None:
            raise RuntimeError("Call reset() first")
        return self._info

    def reset(self, seed: Optional[int] = None) -> Frame:
        obs, _ = self.env.reset(seed=seed)
        if self.policy is not None:
            self.policy.reset()
        self.episode += 1
        self._obs, self._step, self._return, self._seed = obs, 0, 0.0, seed
        key = str(self.env.intervention.vessel_tree.mesh_path)
        if self._info is None or self._info.vessel_key != key:
            self._info = scene_info_from_env(self.env, self.description)
        return frame_from_env(self.env, episode=self.episode, step=0, obs=obs, seed=seed)

    def step(self) -> Frame:
        if self.policy is None or self._obs is None:
            raise RuntimeError("Set a policy and call reset() first")
        obs = self._obs
        action = np.asarray(self.policy.act(obs, first_step=self._step == 0), dtype=np.float32)
        predictions = self.policy.predict(obs, action)
        extras = dict(self.policy.debug())
        try:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as exc:  # SOFA failure inside the step: end the episode, keep the view
            self._step += 1
            extras["error"] = repr(exc)
            return frame_from_env(self.env, episode=self.episode, step=self._step, obs=obs,
                                  action=action, episode_return=self._return, truncated=True,
                                  predictions=predictions, seed=self._seed, extras=extras)
        if info.get("simulation_error"):
            extras["error"] = "simulation_error"
            truncated = True
        self._step += 1
        self._return += float(reward)
        self._obs = next_obs
        return frame_from_env(
            self.env, episode=self.episode, step=self._step, obs=next_obs, action=action,
            raw_action=info.get("raw_action"), reward=float(reward), episode_return=self._return,
            terminated=bool(terminated), truncated=bool(truncated),
            success=bool(info.get("is_success", False)),
            labels={k: float(v) for k, v in (info.get("labels") or {}).items()},
            predictions=predictions, seed=self._seed, extras=extras)

    def close(self) -> None:
        self.env.close()


class ReplaySource:
    """Plays back an episode file written by ``EpisodeLog.save`` (no SOFA needed)."""

    can_run = True

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.log = EpisodeLog.load(self.path)
        self._i = 0

    @property
    def action_dim(self) -> int:
        return len(self.log.info.action_low)

    def scene_info(self) -> SceneInfo:
        return self.log.info

    def reset(self, seed: Optional[int] = None) -> Frame:
        self._i = 0
        return self.log.frames[0]

    def step(self) -> Frame:
        self._i = min(self._i + 1, len(self.log) - 1)
        frame = self.log.frames[self._i]
        if self._i == len(self.log) - 1 and not frame.done:
            frame.truncated = True  # recording ended here
        return frame

    def close(self) -> None:
        pass


class ShapeSource:
    """A vessel shape that no config uses yet: mesh + centerline JSON, drawn without a simulator.

    The JSON has the layout of ``guidewire_rl/env/vessels/biro_phantom/*.json``: ``branches``
    (name -> coordinates, radii), ``insertion`` (position, direction), optionally
    ``task_branches``. One frame, no wire; ``step`` returns it again.
    """

    can_run = False
    action_dim = 0

    def __init__(self, mesh_path: str | Path, centerlines_path: str | Path,
                 image_rot_zx_deg=(20.0, 5.0), description: str = "") -> None:
        import json
        import trimesh

        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        spec = json.loads(Path(centerlines_path).read_text())
        branches = spec["branches"]
        self._info = SceneInfo(
            vessel_vertices=np.asarray(mesh.vertices, dtype=np.float32),
            vessel_faces=np.asarray(mesh.faces, dtype=np.int32),
            centerlines={n: np.asarray(b["coordinates"], dtype=np.float32) for n, b in branches.items()},
            insertion_position=np.asarray(spec["insertion"]["position"], dtype=np.float32),
            insertion_direction=np.asarray(spec["insertion"]["direction"], dtype=np.float32),
            target_threshold_mm=5.0, image_rot_zx_deg=tuple(float(a) for a in image_rot_zx_deg),
            observation_names=[], observation_layout=[],
            action_low=np.zeros(0, np.float32), action_high=np.zeros(0, np.float32),
            max_steps=None, step_seconds=1.0, vessel_key=str(mesh_path), description=description,
            task_branches=tuple(spec.get("task_branches", ())),
            centerline_radii={n: np.asarray(b["radii"], dtype=np.float32) for n, b in branches.items()
                              if "radii" in b})
        self._frame = Frame(episode=0, step=0, obs=np.zeros(0, np.float32),
                            wire_mm=self._info.insertion_position[None].astype(np.float32))

    def scene_info(self) -> SceneInfo:
        return self._info

    def set_policy(self, policy) -> None:
        pass

    def reset(self, seed: Optional[int] = None) -> Frame:
        return self._frame

    def step(self) -> Frame:
        return self._frame

    def close(self) -> None:
        pass
