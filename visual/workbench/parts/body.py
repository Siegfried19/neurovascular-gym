"""Body layer: a see-through human trunk and head around the BIRO phantom's vessels.

Only there to see roughly where in a body the vessels lie. The body is the skin of MakeHuman's
base mesh (``assets/makehuman_base.obj``, CC0; its ``body`` group only: no eyes, teeth or
skeleton), arms and legs cut off (the cuts closed flat), placed in the phantom's training frame
from the vessels of ``rebuilt_cut.json`` (every BIRO phantom version shares that frame):

- axes: left-right along the two common carotids in the neck (mean direction at five
  heights), up = z, front perpendicular to both with the ascending aorta in front of the
  descending one;
- left-right position: the body's midline through the head's vessels (the two internal
  carotids' ends and the top of the basilar artery);
- height: the top of the aortic arch 25 mm below the sternal notch (MakeHuman's clavicle joints);
- front-back: the most anterior chest vessel (a coronary artery) 20 mm behind the skin;
- then the whole body, undeformed, leans forward 20 deg about the left-right axis through the
  top of the aortic arch and moves 40 mm down and 60 mm forward (``LEAN_DEG``, ``LEAN_MOVE_MM``):
  the phantom's head vessels lie further forward than on an upright body. Picked by a grid search
  (lean 0-45 deg, scale 0.85-1.05, moves of 20 mm) for every vessel point inside with the most
  skin over them.

On ``rebuilt_cut`` every centerline point lies inside the skin, at least 17.6 mm under it for
90 % of them (distance to the nearest skin vertex). Upright (not leaning) the head held 11 % of its vessels at best; bending or
stretching the neck to make them fit looked wrong, so the body keeps its own shape.

Drawn see-through, only the faces that look into the body: from outside these are the far
wall, which lies behind every vessel and instrument, so the body never hides them whatever
order the browser draws the see-through meshes in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .environments import VESSELS
from .frame import SceneInfo
from .scene import MM, Color, outside_side

ASSET = Path(__file__).resolve().parents[1] / "assets" / "makehuman_base.obj"
REFERENCE = VESSELS / "rebuilt_cut.json"
BODY_COLOR: Color = (205, 196, 188)

CAROTID_HEIGHTS_MM = (150.0, 170.0, 190.0, 210.0, 250.0)
ARCH_BELOW_NOTCH_MM = 25.0
CHEST_TOP_MM = 120.0              # chest vessels = centerline points below this height
CHEST_VESSEL_DEPTH_MM = 20.0
LEAN_DEG = 20.0
LEAN_MOVE_MM = (0.0, -40.0, 60.0)  # after leaning: (left, up, front)
HIP_CUT_ABOVE_CROTCH_MM = 20.0
ARM_CUT_MM = 60.0                 # along the arm (shoulder -> elbow joint) from the shoulder joint
ARM_CUT_REACH_MM = (45.0, 75.0)   # faces cut there: within this distance along / off the arm's axis
SAME_AORTA_MM = 5.0               # median distance of a scene's aorta to the reference one


@dataclass(frozen=True)
class Skin:
    vertices: np.ndarray          # (N, 3) mm; MakeHuman's frame: x left, y up, z front
    faces: np.ndarray             # (F, 3) int, triangles
    joints: Dict[str, np.ndarray]  # joint name -> (3,) mm


def load_skin(path: Path = ASSET) -> Skin:
    """The ``body`` group of a MakeHuman OBJ (quads split in two) and its joint centres."""
    points, groups, group = [], {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                points.append(line.split()[1:4])
            elif line.startswith("g "):
                group = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                groups.setdefault(group, []).append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    points = np.asarray(points, dtype=np.float64) * 100.0        # dm -> mm
    joints = {name[len("joint-"):]: points[np.unique(np.concatenate(faces))].mean(axis=0)
              for name, faces in groups.items() if name.startswith("joint-")}
    quads = np.asarray(groups["body"])
    used = np.unique(quads)
    index = np.full(len(points), -1)
    index[used] = np.arange(len(used))
    q = index[quads]
    return Skin(points[used], np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), joints)


# ------------------------------------------------------------------ trunk --
def _cut(vertices, faces, origin, normal, only: Optional[np.ndarray] = None):
    """Keep what lies on the side the normal points to; only the faces in ``only`` are cut."""
    import trimesh
    from trimesh import intersections
    only = np.ones(len(faces), bool) if only is None else only
    v, f = intersections.slice_faces_plane(vertices, faces[only], plane_normal=normal, plane_origin=origin)[:2]
    mesh = trimesh.Trimesh(np.vstack([vertices, v]), np.vstack([faces[~only], f + len(vertices)]), process=True)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _largest_piece(vertices, faces):
    import trimesh
    pieces = trimesh.Trimesh(vertices, faces, process=False).split(only_watertight=False)
    big = max(pieces, key=lambda m: len(m.faces))
    return np.asarray(big.vertices), np.asarray(big.faces)


def _close_cuts(vertices, faces) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Close every open rim with a fan around its centre (each rim is one flat cut)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, count = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    following = {int(a): int(b) for a, b in edges[count[inverse.ravel()] == 1]}
    vertices, faces, rims, seen = list(vertices), list(faces), [], set()
    for start in following:
        if start in seen:
            continue
        rim, b = [start], following[start]
        seen.add(start)
        while b != start:
            rim.append(b)
            seen.add(b)
            b = following[b]
        centre = len(vertices)
        vertices.append(np.mean([vertices[i] for i in rim], axis=0))
        faces += [[rim[(i + 1) % len(rim)], rim[i], centre] for i in range(len(rim))]   # against the rim's own edge
        rims.append(len(rim))
    return np.asarray(vertices), np.asarray(faces), rims


def trunk(skin: Skin) -> Tuple[np.ndarray, np.ndarray]:
    """The skin without arms and legs, closed: cut just above the crotch and across each upper arm."""
    v, f = skin.vertices, skin.faces
    midline = v[(np.abs(v[:, 0]) < 8) & (v[:, 1] > -300)]
    v, f = _cut(v, f, [0.0, midline[:, 1].min() + HIP_CUT_ABOVE_CROTCH_MM, 0.0], [0.0, 1.0, 0.0])
    for side in ("l", "r"):
        shoulder, elbow = skin.joints[f"{side}-shoulder"], skin.joints[f"{side}-elbow"]
        axis = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
        along = (v - shoulder) @ axis
        off = np.linalg.norm(v - shoulder - along[:, None] * axis, axis=1)
        near = (np.abs(along - ARM_CUT_MM) < ARM_CUT_REACH_MM[0]) & (off < ARM_CUT_REACH_MM[1])
        v, f = _cut(v, f, shoulder + ARM_CUT_MM * axis, -axis, near[f].all(axis=1))
    v, f = _largest_piece(v, f)            # the arms beyond the cuts are loose now
    v, f, _ = _close_cuts(v, f)
    return v, f


# -------------------------------------------------------------- placement --
def _nearest_in_height(line: np.ndarray, z: float) -> np.ndarray:
    return line[np.argmin(np.abs(line[:, 2] - z))]


def body_axes(branches: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(left, up, front) unit vectors of the body in the phantom's training frame."""
    right = np.concatenate([branches["bct"], branches["rcca"], branches["rcca_rest"]])
    left = np.concatenate([branches["lcca"], branches["lcca_rest"]])
    across = [(_nearest_in_height(left, z) - _nearest_in_height(right, z))[:2] for z in CAROTID_HEIGHTS_MM]
    lateral = np.mean([a / np.linalg.norm(a) for a in across], axis=0)
    lateral /= np.linalg.norm(lateral)
    to_left, up = np.array([lateral[0], lateral[1], 0.0]), np.array([0.0, 0.0, 1.0])
    front = np.cross(up, to_left)
    aorta = branches["aorta"]                       # from the descending part to the root
    if front @ (aorta[-1] - aorta[0]) < 0:
        front = -front
    return to_left, up, front


def place(skin: Skin, branches: Mapping[str, np.ndarray],
          vertices: Optional[np.ndarray] = None) -> np.ndarray:
    """``vertices`` (default: the whole skin; MakeHuman's frame) in the phantom's training frame (mm)."""
    axes = body_axes(branches)

    def body(p):            # training frame -> MakeHuman's (left, up, front)
        return np.stack([np.asarray(p) @ a for a in axes], axis=-1)

    head = body(np.array([branches["rcca_rest"][-1], branches["lcca_rest"][-1],
                          branches["v4"][-1]])).mean(axis=0)     # v4 ends at the basilar artery's top
    aorta = branches["aorta"]
    arch_top = body(aorta[np.argmax(aorta[:, 2])])
    points = body(np.concatenate(list(branches.values())))
    chest = points[points[:, 1] < CHEST_TOP_MM]
    shift = np.zeros(3)
    shift[0] = head[0]
    shift[1] = arch_top[1] + ARCH_BELOW_NOTCH_MM - skin.joints["l-clavicle"][1]
    lx, ly, lz = chest[np.argmax(chest[:, 2])]
    near = (np.abs(skin.vertices[:, 0] - (lx - shift[0])) < 12) & (np.abs(skin.vertices[:, 1] - (ly - shift[1])) < 12)
    shift[2] = lz + CHEST_VESSEL_DEPTH_MM - skin.vertices[near, 2].max()
    v = (skin.vertices if vertices is None else vertices) + shift - arch_top
    a = np.radians(LEAN_DEG)
    lean = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    v = arch_top + v @ lean.T + np.asarray(LEAN_MOVE_MM)
    return v[:, :1] * axes[0] + v[:, 1:2] * axes[1] + v[:, 2:] * axes[2]


def _branches(path: Path) -> Dict[str, np.ndarray]:
    spec = json.loads(Path(path).read_text())
    return {name: np.asarray(b["coordinates"], dtype=np.float64) for name, b in spec["branches"].items()}


@lru_cache(maxsize=1)
def body_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """(vertices mm in the training frame, faces) of the placed trunk and head; made once."""
    skin = load_skin()
    vertices, faces = trunk(skin)
    return place(skin, _branches(REFERENCE), vertices).astype(np.float32), faces.astype(np.int32)


@lru_cache(maxsize=1)
def _reference_aorta() -> np.ndarray:
    return _branches(REFERENCE)["aorta"]


def fits(info: SceneInfo) -> bool:
    """Whether the body was placed around this scene's vessels: its aorta is the BIRO phantom's
    (every version's lies within 1 mm of the reference one, median)."""
    aorta = info.centerlines.get("aorta")
    if aorta is None or len(aorta) == 0:
        return False
    ref = _reference_aorta()
    gap = np.min(np.linalg.norm(np.asarray(aorta, dtype=np.float64)[:, None] - ref[None], axis=-1), axis=1)
    return float(np.median(gap)) < SAME_AORTA_MM


class BodyLayer:
    """The trunk and head around the vessels; nothing when the scene is not the BIRO phantom."""

    def __init__(self, server, path: str = "/body", opacity: float = 0.15) -> None:
        self._server, self._path, self._opacity = server, path, opacity
        self._handle = None
        self._visible = True

    def set_scene(self, info: SceneInfo) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if not fits(info):
            return
        vertices, faces = body_mesh()
        inward = {"front": "back", "back": "front"}.get(outside_side(vertices, faces), "double")
        self._handle = self._server.scene.add_mesh_simple(
            self._path, vertices=vertices * MM, faces=faces, color=BODY_COLOR, opacity=self._opacity,
            side=inward, cast_shadow=False, receive_shadow=False, visible=self._visible)

    @property
    def shown(self) -> bool:
        return self._handle is not None

    def set_opacity(self, opacity: float) -> None:
        self._opacity = float(opacity)
        if self._handle is not None:
            self._handle.opacity = self._opacity

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible
