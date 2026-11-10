"""Centerlines for the phantom model's own wall (``phantom_lumen.obj``): the same lines as
``phantom_arch_all.json``, every radius measured to that wall.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_phantom_lumen_centerlines

reads ``biro_phantom/phantom_arch.json``, ``vessel_tree.npz`` and ``phantom_lumen.obj``, writes
``biro_phantom/phantom_lumen_all.json`` (about a minute; ``guidewire_rl`` env only).

Why: a centerline point's radius is its distance to the wall (the largest ball centred there
that fits). The lab's radii are distances to the lab's wall, which is built from balls along
these lines; the model's own wall is wider (a median 0.35-0.9 mm on the training branches,
several mm at the aortic root and junctions), so with it the old radii would leave a ring of
lumen outside every branch tube again.

How:
  1. The aortic root is re-routed to the bottom of the new wall's root (``biro_phantom_root.extend``
     with the new wall; its root is wider and shaped differently).
  2. Every other lab vessel is added as in ``biro_phantom_all.build``, if it lies inside the new wall.
  3. Every point's radius becomes its exact distance to the new wall. The points stay where they
     are: checked 2026-10-27, they sit a median 0.25-0.35 mm from the point of their
     cross-section farthest from the new wall (the check's grid is 0.25 mm), and moving them
     would add about 0.1 mm of radius. A point nearer the wall than ``R_FLOOR_MM`` (or outside
     it) gets ``R_FLOOR_MM``; the record counts them.
``task_branches`` are the six, as in ``phantom_arch_all.json``. This file goes with
``phantom_lumen.obj``, not with the lab's ``phantom_arch.obj``; it is not wired into
``BiroPhantomArch`` yet.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .biro_phantom import DATA_DIR, SPEC_PATH
from .biro_phantom_all import TREE_PATH, build
from .biro_phantom_lumen import OUT_MESH as LUMEN_MESH
from .biro_phantom_root import Wall, extend

OUT_PATH = DATA_DIR / "phantom_lumen_all.json"
R_FLOOR_MM = 0.3


def remeasure(spec: dict, mesh) -> dict:
    out = json.loads(json.dumps(spec))
    aorta = np.asarray(out["branches"]["aorta"]["coordinates"], float)
    wall = Wall(mesh, mesh.bounds[0], mesh.bounds[1], inside_point=aorta[100])
    floored, change = 0, {}
    for name, branch in out["branches"].items():
        old = np.asarray(branch["radii"], float)
        distance = wall.distance(np.asarray(branch["coordinates"], float))
        floored += int(np.sum(distance < R_FLOOR_MM))
        radii = np.maximum(distance, R_FLOOR_MM)
        branch["radii"] = radii.round(3).tolist()
        change[name] = radii - old
    task = out["task_branches"]
    out["lumen_wall"] = {
        "what": "radii measured to the phantom model's own inner wall; points as phantom_arch_all.json "
                "except the aortic root, re-routed on this wall",
        "wall": f"phantom_lumen.obj (sha256 {hashlib.sha256(LUMEN_MESH.read_bytes()).hexdigest()})",
        "script": "guidewire_rl/env/vessels/biro_phantom_lumen_centerlines.py",
        "radius": f"exact distance to the wall, at least {R_FLOOR_MM} mm",
        "points_at_the_floor": floored,
        "radius_change_mm_median": {name: round(float(np.median(change[name])), 2) for name in task},
        "radius_change_mm_all_branches": {
            "median": round(float(np.median(np.concatenate(list(change.values())))), 2),
            "p10": round(float(np.percentile(np.concatenate(list(change.values())), 10)), 2),
            "p90": round(float(np.percentile(np.concatenate(list(change.values())), 90)), 2)},
    }
    return out


def main() -> None:
    import trimesh

    mesh = trimesh.load(LUMEN_MESH, process=True)
    root = extend(json.loads(SPEC_PATH.read_text()), mesh, wall_path=LUMEN_MESH)
    full = build(root, np.load(TREE_PATH), mesh, wall_path=LUMEN_MESH,
                 root_from="phantom_arch.json with the aortic root re-routed on phantom_lumen.obj "
                           "(biro_phantom_root.extend)")
    out = remeasure(full, mesh)
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    info, ext = out["lumen_wall"], out["all_vessels"]
    print(f"wrote {OUT_PATH}: {len(out['branches'])} branches (left out {ext['left_out_not_in_mesh']}); "
          f"root: kept {ext['root_extension']['kept_lab_points']} lab points, {ext['root_extension']['new_points']} new, "
          f"end {ext['root_extension']['end_point']}; radius change on the six {info['radius_change_mm_median']}, "
          f"all {info['radius_change_mm_all_branches']}; at the floor {info['points_at_the_floor']}")


if __name__ == "__main__":
    main()
