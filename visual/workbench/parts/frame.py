"""What the viewer shows: one step of a deployment (``Frame``), the things that
stay fixed during an episode (``SceneInfo``), and an episode of frames that can
be saved and replayed (``EpisodeLog``).

Pure numpy: no SOFA, no viser, no torch. Lengths are mm in the vessel frame
(the frame stEVE builds the SOFA scene in), angles are radians unless named
``*_deg``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .instrument import InstrumentInfo

FORMAT = "workbench.episode/2"
OLD_FORMATS = ("deploy_viewer.episode/1",)   # still loadable; lacks the fields added in /2


@dataclass(eq=False)
class SceneInfo:
    vessel_vertices: np.ndarray                 # (V, 3) mm
    vessel_faces: np.ndarray                    # (F, 3) int
    centerlines: Dict[str, np.ndarray]          # branch name -> (N, 3) mm
    insertion_position: np.ndarray              # (3,)
    insertion_direction: np.ndarray             # (3,) unit
    target_threshold_mm: float
    image_rot_zx_deg: Tuple[float, float]       # fluoroscope angles of the 2D projection
    observation_names: List[str]
    observation_layout: List[Tuple[str, int]]   # (term name, size) in observation order
    action_low: np.ndarray                      # device units per action dim (mm/s, rad/s)
    action_high: np.ndarray
    max_steps: Optional[int]
    step_seconds: float                         # simulated time per control step
    vessel_key: str = ""                        # changes when the vessel changes (re-draw then)
    description: str = ""                       # free text shown in the status panel
    task_branches: Tuple[str, ...] = ()         # the training's branches; empty = all of them
    centerline_radii: Dict[str, np.ndarray] = field(default_factory=dict)   # branch -> (N,) mm
    instruments: List[InstrumentInfo] = field(default_factory=list)

    def is_task_branch(self, name: str) -> bool:
        return not self.task_branches or name in self.task_branches


@dataclass(eq=False)
class Frame:
    episode: int
    step: int                                   # 0 = right after reset
    obs: np.ndarray                             # what the policy sees at this step
    wire_mm: np.ndarray                         # (N, 3) inserted part of the wire, tip first
    target_mm: Optional[np.ndarray] = None      # (3,)
    action: Optional[np.ndarray] = None         # normalized [-1, 1] action that led to this frame
    raw_action: Optional[np.ndarray] = None     # the same action in device units
    reward: float = 0.0
    episode_return: float = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    labels: Dict[str, float] = field(default_factory=dict)       # measured during the step
    predictions: Dict[str, float] = field(default_factory=dict)  # policy's prediction for this step
    inserted_mm: Optional[np.ndarray] = None    # per instrument
    rotation_rad: Optional[np.ndarray] = None   # per instrument
    seed: Optional[int] = None
    # Free slot for modules that do not exist yet (planner candidates, safety filter, force
    # vectors, ...). Keys the viewer already reads: "nominal_action", "filter", "force_vector_N",
    # "error". Values must be JSON-serializable or numpy arrays of numbers.
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def tip_mm(self) -> Optional[np.ndarray]:
        return self.wire_mm[0] if len(self.wire_mm) else None


def inserted_wire(dof_positions: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Tip-first DOF positions without the not-yet-inserted tail.

    stEVE parks every beam node that is still outside the vessel on the insertion
    point, so the array ends in repeats of one point; keep one copy of it.
    """
    points = np.asarray(dof_positions, dtype=np.float32).reshape(-1, 3)
    if len(points) < 2:
        return points
    moving = np.linalg.norm(np.diff(points, axis=0), axis=1) > eps
    if not moving.any():
        return points[:1]
    last = int(np.nonzero(moving)[0][-1])
    return points[: last + 2]


def instrument_polylines(wire_mm: np.ndarray, inserted_mm: Optional[Sequence[float]]) -> List[np.ndarray]:
    """Each instrument's inserted part, tip first, out of the combined beam ``wire_mm`` (tip first).

    Coaxial instruments (guidewire in a catheter) share one beam in BeamAdapter, so a frame holds a
    single polyline. One that is x mm shorter than the leading one ends x mm down the beam, counted
    back from the leading tip (the two-device env's rule, ``guidewire_rl.env.two_device.phases.
    instrument_polylines``; a test keeps the two equal). Fewer than two lengths: the beam itself.
    """
    points = np.asarray(wire_mm, dtype=np.float64).reshape(-1, 3)
    if inserted_mm is None or len(inserted_mm) < 2 or len(points) < 2:
        return [points]
    cum = arc_length(points)
    lead = float(max(inserted_mm))
    out = []
    for length in inserted_mm:
        back = min(max(lead - float(length), 0.0), float(cum[-1]))
        tip = np.array([np.interp(back, cum, points[:, k]) for k in range(3)])
        out.append(np.vstack([tip[None], points[cum > back + 1e-9]]))
    return out


def polyline_head(points: np.ndarray, length_mm: float) -> np.ndarray:
    """The first ``length_mm`` of a polyline (tip first), end point interpolated."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2 or length_mm <= 0:
        return points[:1] if length_mm > 0 else points[:0]
    cum = arc_length(points)
    length = min(float(length_mm), float(cum[-1]))
    end = np.array([np.interp(length, cum, points[:, k]) for k in range(3)])
    return np.vstack([points[cum < length - 1e-9], end[None]])


def arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative length along a polyline, 0 at the first point."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def fluoro_rotation(image_rot_zx_deg: Sequence[float]) -> np.ndarray:
    """stEVE's vessel -> tracking rotation (``eve.util.coordtransform._get_rot_matrix``)."""
    rz, rx = (-float(a) * np.pi / 180.0 for a in image_rot_zx_deg)
    mz = np.array([[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]])
    mx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]])
    return mz @ mx


def project_2d(points_mm: np.ndarray, image_rot_zx_deg: Sequence[float]) -> np.ndarray:
    """Fluoroscope projection of vessel-frame points: rotate, then drop the depth axis (y)."""
    rotated = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3) @ fluoro_rotation(image_rot_zx_deg).T
    return rotated[:, [0, 2]]


def tracked_points_3d(wire_mm: np.ndarray, image_rot_zx_deg: Sequence[float], spacing_mm: float = 1.0,
                      count: int = 5) -> np.ndarray:
    """The 3D wire points behind the tracked image points: ``count`` points ``spacing_mm`` apart
    along the wire as the image shows it (the projected length), from the tip."""
    wire = np.asarray(wire_mm, dtype=np.float64).reshape(-1, 3)
    if len(wire) < 2:
        return wire[:count]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(project_2d(wire, image_rot_zx_deg), axis=0),
                                                        axis=1))])
    wanted = np.minimum(np.arange(count) * spacing_mm, s[-1])
    return np.stack([np.interp(wanted, s, wire[:, i]) for i in range(3)], axis=1)


def resample_polyline(points: np.ndarray, spacing: float, count: int) -> np.ndarray:
    """``count`` points ``spacing`` apart along a polyline, from its first point.

    Same idea as stEVE's ``Tracking2D`` sampling (repeats the end when the line is short).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, points.shape[-1] if points.ndim == 2 else 3))
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    wanted = np.minimum(np.arange(count) * spacing, s[-1])
    return np.stack([np.interp(wanted, s, points[:, i]) for i in range(points.shape[1])], axis=1)


# ------------------------------------------------------------------ episode log --
class EpisodeLog:
    """Frames of one episode plus its ``SceneInfo``; saves to / loads from one ``.npz``."""

    def __init__(self, info: SceneInfo) -> None:
        self.info = info
        self.frames: List[Frame] = []

    def __len__(self) -> int:
        return len(self.frames)

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    def series(self, getter) -> np.ndarray:
        return np.array([getter(f) for f in self.frames], dtype=np.float64)

    def tip_trail(self) -> np.ndarray:
        tips = [f.tip_mm for f in self.frames if f.tip_mm is not None]
        return np.array(tips, dtype=np.float32).reshape(-1, 3)

    # -- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info, frames = self.info, self.frames
        if not frames:
            raise ValueError("Nothing to save: the episode has no frames")
        a_dim = len(info.action_low)
        wires = [f.wire_mm.astype(np.float32) for f in frames]
        arrays = {
            "vessel_vertices": info.vessel_vertices.astype(np.float32),
            "vessel_faces": info.vessel_faces.astype(np.int32),
            "insertion_position": np.asarray(info.insertion_position, np.float32),
            "insertion_direction": np.asarray(info.insertion_direction, np.float32),
            "action_low": np.asarray(info.action_low, np.float32),
            "action_high": np.asarray(info.action_high, np.float32),
            "obs": np.stack([f.obs for f in frames]).astype(np.float32),
            "wire": np.concatenate(wires).astype(np.float32),
            "wire_offsets": np.cumsum([0] + [len(w) for w in wires]).astype(np.int64),
            "target": np.stack([_or_nan(f.target_mm, 3) for f in frames]),
            "action": np.stack([_or_nan(f.action, a_dim) for f in frames]),
            "raw_action": np.stack([_or_nan(f.raw_action, a_dim) for f in frames]),
        }
        for name, line in info.centerlines.items():
            arrays[f"centerline/{name}"] = np.asarray(line, np.float32)
        for name, radii in info.centerline_radii.items():
            arrays[f"radii/{name}"] = np.asarray(radii, np.float32)
        n_inst = max((len(f.inserted_mm) for f in frames if f.inserted_mm is not None), default=0)
        if n_inst:
            arrays["inserted_mm"] = np.stack([_or_nan(f.inserted_mm, n_inst) for f in frames])
            arrays["rotation_rad"] = np.stack([_or_nan(f.rotation_rad, n_inst) for f in frames])
        meta = {
            "format": FORMAT,
            "info": {
                "target_threshold_mm": info.target_threshold_mm,
                "image_rot_zx_deg": list(info.image_rot_zx_deg),
                "observation_names": list(info.observation_names),
                "observation_layout": [list(x) for x in info.observation_layout],
                "max_steps": info.max_steps, "step_seconds": info.step_seconds,
                "vessel_key": info.vessel_key, "description": info.description,
                "centerlines": list(info.centerlines),
                "radii": list(info.centerline_radii),
                "task_branches": list(info.task_branches),
                "instruments": [i.to_json() for i in info.instruments],
            },
            "frames": [{
                "episode": f.episode, "step": f.step, "reward": f.reward,
                "episode_return": f.episode_return, "terminated": f.terminated,
                "truncated": f.truncated, "success": f.success, "seed": f.seed,
                "labels": f.labels, "predictions": f.predictions, "extras": _jsonable(f.extras),
            } for f in frames],
        }
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeLog":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") not in (FORMAT, *OLD_FORMATS):
                raise ValueError(f"{path}: not a {FORMAT} file")
            m = meta["info"]
            info = SceneInfo(
                vessel_vertices=data["vessel_vertices"], vessel_faces=data["vessel_faces"],
                centerlines={n: data[f"centerline/{n}"] for n in m["centerlines"]},
                insertion_position=data["insertion_position"],
                insertion_direction=data["insertion_direction"],
                target_threshold_mm=float(m["target_threshold_mm"]),
                image_rot_zx_deg=tuple(m["image_rot_zx_deg"]),
                observation_names=list(m["observation_names"]),
                observation_layout=[(str(n), int(k)) for n, k in m["observation_layout"]],
                action_low=data["action_low"], action_high=data["action_high"],
                max_steps=m["max_steps"], step_seconds=float(m["step_seconds"]),
                vessel_key=m["vessel_key"], description=m["description"],
                task_branches=tuple(m.get("task_branches", ())),
                centerline_radii={n: data[f"radii/{n}"] for n in m.get("radii", [])},
                instruments=[InstrumentInfo.from_json(i) for i in m.get("instruments", [])])
            log = cls(info)
            offsets = data["wire_offsets"]
            has_inst = "inserted_mm" in data
            for i, fm in enumerate(meta["frames"]):
                log.append(Frame(
                    episode=fm["episode"], step=fm["step"], obs=data["obs"][i],
                    wire_mm=data["wire"][offsets[i]:offsets[i + 1]],
                    target_mm=_nan_to_none(data["target"][i]),
                    action=_nan_to_none(data["action"][i]),
                    raw_action=_nan_to_none(data["raw_action"][i]),
                    reward=fm["reward"], episode_return=fm["episode_return"],
                    terminated=fm["terminated"], truncated=fm["truncated"], success=fm["success"],
                    labels=fm["labels"], predictions=fm["predictions"], seed=fm["seed"],
                    inserted_mm=_nan_to_none(data["inserted_mm"][i]) if has_inst else None,
                    rotation_rad=_nan_to_none(data["rotation_rad"][i]) if has_inst else None,
                    extras=fm["extras"]))
        return log


def _or_nan(value: Optional[np.ndarray], size: int) -> np.ndarray:
    if value is None:
        return np.full(size, np.nan, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(size)


def _nan_to_none(value: np.ndarray) -> Optional[np.ndarray]:
    return None if np.all(np.isnan(value)) else value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value
