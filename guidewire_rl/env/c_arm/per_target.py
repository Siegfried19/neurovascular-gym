"""The C-arm angle per target branch, in training (README.md, "In training").

A config switches it on with one line::

    env: {scene: {fluoroscopy: {image_rot_zx: per_target}}}

The builder (``guidewire_rl/env/builder.py``) then

1. finds the angle table of the scene's phantom: the table in ``data/`` whose ``centerline.file`` is the
   vessel tree's centerline file (``BiroPhantomArch.spec_path``) and whose ``centerline.sha256`` is that
   file's; no table, or a table made from another version of the file, is an error that names the
   command writing one (``choose_angle.py --write-table``; training never runs the search);
2. builds the fluoroscopy at the table's all-5 angle (it holds until the first reset);
3. after every ``intervention.reset`` (scene rebuilt, target drawn; before start, pathfinder, monitors,
   observations, rewards): finds the target's branch, maps a continuation such as ``rcca_rest`` to the
   target branch it continues (``rcca``; the table's ``continuations``), sets
   ``fluoroscopy.image_rot_zx`` to that branch's angle and, when the angle changed, recomputes the
   target's ``coordinates3d`` (stEVE keeps targets in the image frame, so the target was drawn under
   the previous angle) from its exact vessel-frame point. A target on any other branch is an error.

The lookup goes by the target's branch; the angle in that branch's row was chosen for the branch's whole
route from the aorta (``choose_angle.route``: the right carotid's is aorta + trunk + carotid), so the
same row serves every target on the branch.

The target's branch: ``target.branch`` when the target names it (``RouteWindowTarget``, stEVE's
``BranchIndex``); otherwise the branch with the centerline point nearest to the target (stEVE's
``find_nearest_branch_to_point``, as its pathfinder decides the target branch; a point two branches
share, such as a fork, goes to the first one in the vessel tree's order).

Everything that projects to the image reads ``fluoroscopy.image_rot_zx`` when it is used or caches it
by the angle (``two_device.observations._TaskBox``, ``VesselGeometry``), so it follows the new angle;
``NormalizeTracking2DEpisode`` recomputes its range at every reset, after this.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

PER_TARGET = "per_target"
TABLE_DIR = Path(__file__).with_name("data")
SNAP_MM = 0.01      # a target found again this close to one of the points it draws from is that point
CANDIDATES = ("_potential_targets", "targets_vessel_cs")   # stEVE's CenterlineRandom, Manual


def is_per_target(spec: Any) -> bool:
    """Whether a fluoroscopy spec asks for the per-target angle."""
    if not isinstance(spec, Mapping):
        return False
    angle = spec.get("image_rot_zx")
    if isinstance(angle, str):
        if angle != PER_TARGET:
            raise ValueError(f"env.scene.fluoroscopy.image_rot_zx must be [rot_z, rot_x] or "
                             f"{PER_TARGET!r}, got {angle!r}")
        return True
    return False


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _no_table(spec: Path) -> str:
    from . import choose_angle
    head = f"no C-arm angle table for the centerlines {spec.name} in {TABLE_DIR}"
    inputs = [name for name, path in choose_angle.DATA.items() if path.name == spec.name]
    if inputs:
        return f"{head}; write one with: {choose_angle.command(inputs[0])}"
    return (f"{head}, and the angle chooser has no input for that file: add it to DATA in "
            f"guidewire_rl/env/c_arm/choose_angle.py, then run it with --write-table")


def table_for(vessel_tree) -> Dict[str, Any]:
    """The vendored angle table of this vessel tree's centerline file (module docstring, 1)."""
    spec = getattr(vessel_tree, "spec_path", None)
    if spec is None:
        raise ValueError(f"image_rot_zx: {PER_TARGET} needs a phantom vessel tree that has an angle table "
                         f"(BiroPhantomArch); {type(vessel_tree).__name__} names no centerline file")
    spec = Path(spec)
    for path in sorted(TABLE_DIR.glob("angles_*.json")):
        table = json.loads(path.read_text())
        if table["centerline"]["file"] != spec.name:
            continue
        actual = _sha256(spec)
        if actual != table["centerline"]["sha256"]:
            raise ValueError(f"C-arm angle table {path.name} was made from {spec.name} with sha256 "
                             f"{table['centerline']['sha256']}, the file now has {actual}; "
                             f"write it again: {table['command']}")
        return table
    raise FileNotFoundError(_no_table(spec))


def initial_angle(table: Mapping[str, Any]) -> List[float]:
    """The angle before the first reset: all 5 targets together."""
    return [float(a) for a in table["angles"]["all"]["image_rot_zx"]]


def table_key(table: Mapping[str, Any], branch: str) -> str:
    """The table row for a target on ``branch``: the branch itself, or the target it continues."""
    targets = [k for k in table["angles"] if k != "all"]
    if branch in targets:
        return branch
    parent = table["continuations"].get(branch)
    if parent is not None:
        return parent
    raise ValueError(f"the episode's target is on {branch!r}; the C-arm angle table has angles for "
                     f"{targets} and their continuations {sorted(table['continuations'])} only")


def target_point_vessel(target, fluoroscopy) -> np.ndarray:
    """The target in vessel coordinates, exactly the point it was drawn as: ``coordinates_vessel`` when
    the target keeps it, else ``coordinates3d`` turned back by the current angle and put on the
    candidate point it came from (within ``SNAP_MM``) when the target keeps its candidates."""
    exact = getattr(target, "coordinates_vessel", None)
    if exact is not None:
        return np.asarray(exact, dtype=np.float64).reshape(3)
    from eve.util.coordtransform import tracking3d_to_vessel_cs
    point = np.asarray(tracking3d_to_vessel_cs(np.asarray(target.coordinates3d, dtype=np.float64).reshape(3),
                                               fluoroscopy.image_rot_zx, fluoroscopy.image_center),
                       dtype=np.float64).reshape(3)
    for attr in CANDIDATES:
        candidates = getattr(target, attr, None)
        if candidates is None or len(candidates) == 0:
            continue
        candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
        distance = np.linalg.norm(candidates - point, axis=1)
        i = int(np.argmin(distance))
        if distance[i] < SNAP_MM:
            return candidates[i].copy()
    return point


def target_branch(target, vessel_tree, point: np.ndarray) -> str:
    """The branch the target lies on (module docstring)."""
    name = getattr(target, "branch", None)
    if isinstance(name, str) and name:
        return name
    from eve.intervention.vesseltree.vesseltree import find_nearest_branch_to_point
    return find_nearest_branch_to_point(point, vessel_tree).name


class PerTargetCArm:
    """Sets the fluoroscopy's angle from the episode's target (module docstring, 3)."""

    def __init__(self, table: Mapping[str, Any], vessel_tree, fluoroscopy, target) -> None:
        self.table = table
        self.vessel_tree = vessel_tree
        self.fluoroscopy = fluoroscopy
        self.target = target
        self.branch = None        # this episode's target branch
        self.key = None           # its row in the table

    def angle(self, key: str) -> List[float]:
        return [float(a) for a in self.table["angles"][key]["image_rot_zx"]]

    def reset(self) -> None:
        point = target_point_vessel(self.target, self.fluoroscopy)   # under the angle it was drawn with
        self.branch = target_branch(self.target, self.vessel_tree, point)
        self.key = table_key(self.table, self.branch)
        angle = self.angle(self.key)
        fluoroscopy = self.fluoroscopy
        if [float(a) for a in np.ravel(fluoroscopy.image_rot_zx)] == angle:
            return
        from eve.util.coordtransform import vessel_cs_to_tracking3d
        fluoroscopy.image_rot_zx = angle
        self.target.coordinates3d = vessel_cs_to_tracking3d(point, angle, fluoroscopy.image_center,
                                                            fluoroscopy.field_of_view)
