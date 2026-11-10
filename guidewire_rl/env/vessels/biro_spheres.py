"""The inscribed-sphere phantom (内接球版) from the phantom model alone.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_spheres

reads ``biro_phantom/rebuilt_full.json`` (the centerlines ``biro_rebuild.py`` computes from the
phantom model, the STL, and nothing else) and writes ``biro_phantom/spheres_full.obj`` /
``spheres_full.json`` (collision wall and centerlines), then cuts the right arm off
(``biro_cut.py``) into ``spheres_cut.obj`` / ``spheres_cut.json``. About a minute.

The wall is made the way the lab made its own (``19-phantom-sofa/mesh.py``), only the input
differs: the union of the inscribed spheres along every centerline. Each centerline radius is
the distance from that point to the model's inner wall, so a sphere there is the largest that
fits; spheres narrower than ``R_MIN_MM`` are widened to it (the contact model keeps the wire
0.3 mm off the wall and needs room), as the lab did. On a ``GRID_MM`` grid every node gets
max over the spheres of (radius - distance to the centre); the surface where that is 0
(marching cubes, the call stEVE's own meshes use) is decimated to ``FACES`` triangles keeping
the volume, and the largest connected piece is kept.

The centerlines are the same as the model-inner-wall version's (task branches, junctions,
insertion), with every radius raised to ``R_MIN_MM`` like the wall, so the lumen the
observation and the reward assume is the one the wire meets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).with_name("biro_phantom")
SOURCE = DATA_DIR / "rebuilt_full.json"
OUT_MESH, OUT_SPEC = DATA_DIR / "spheres_full.obj", DATA_DIR / "spheres_full.json"

R_MIN_MM = 1.0          # lab 19 R_MIN
GRID_MM = 0.5           # lab 19's grid (its lumen voxels)
FACES = 60_000          # lab 19 TRIANGLES
PAD = 6                 # grid nodes around the spheres


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arclength(points) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def sphere_centres(branches: dict):
    """Sphere centres and radii along every centerline: every 0.5 mm where the vessel's median
    radius is under 3 mm, every 1 mm elsewhere (lab 19); radius at least R_MIN_MM."""
    centres, radii = [], []
    for b in branches.values():
        p, r = np.asarray(b["coordinates"], float), np.asarray(b["radii"], float)
        a = arclength(p)
        step = 0.5 if np.median(r) < 3.0 else 1.0
        s = np.linspace(0.0, a[-1], max(int(np.ceil(a[-1] / step)), 1) + 1)
        centres.append(np.stack([np.interp(s, a, p[:, k]) for k in range(3)], 1))
        radii.append(np.maximum(np.interp(s, a, r), R_MIN_MM))
    return np.vstack(centres), np.concatenate(radii)


def sphere_field(centres, radii):
    """max over the spheres of (radius - distance), exact within two grid steps of each sphere's
    surface, -2 grid steps farther out. Returns the field and the grid's origin."""
    reach = radii + 2 * GRID_MM
    origin = (centres - reach[:, None]).min(axis=0) - PAD * GRID_MM
    shape = tuple(int(n) for n in np.ceil(((centres + reach[:, None]).max(axis=0) - origin) / GRID_MM) + PAD + 1)
    f = np.full(shape, -2.0 * GRID_MM, np.float32)
    axes = [origin[k] + GRID_MM * np.arange(shape[k]) for k in range(3)]
    for c, r, rr in zip(centres, radii, reach):
        lo = [max(int(np.floor((c[k] - rr - origin[k]) / GRID_MM)), 0) for k in range(3)]
        hi = [min(int(np.ceil((c[k] + rr - origin[k]) / GRID_MM)) + 1, shape[k]) for k in range(3)]
        d2 = ((axes[0][lo[0]:hi[0]] - c[0]) ** 2)[:, None, None] + ((axes[1][lo[1]:hi[1]] - c[1]) ** 2)[None, :, None] \
            + ((axes[2][lo[2]:hi[2]] - c[2]) ** 2)[None, None, :]
        box = f[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.maximum(box, (r - np.sqrt(d2)).astype(np.float32), out=box)
    return f, origin


def build_wall(branches: dict):
    """The union of the spheres as a closed triangle mesh (training frame)."""
    import pyvista as pv
    import trimesh
    from skimage import measure

    centres, radii = sphere_centres(branches)
    f, origin = sphere_field(centres, radii)
    v, faces, _, _ = measure.marching_cubes(f, 0.0, spacing=(GRID_MM,) * 3, gradient_direction="descent")
    poly = pv.PolyData(v + origin, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    n_full = poly.n_cells
    small = poly.decimate(1.0 - FACES / n_full, volume_preservation=True).triangulate().clean()
    small = pv.PolyData(small.points, small.faces).connectivity("largest")
    mesh = trimesh.Trimesh(np.asarray(small.points), small.faces.reshape(-1, 4)[:, 1:], process=False)
    # decimation can leave a flat two-triangle sliver on an edge of the surface (the lab's wall has one
    # too, phantom_arch.obj near [11.5, 1.0, 23.0]); keep the wall itself
    pieces = mesh.split(only_watertight=False)
    mesh = max(pieces, key=lambda p: len(p.faces))
    record = {"spheres": int(len(centres)), "faces_before": int(n_full), "slivers_dropped": len(pieces) - 1,
              "faces": int(len(mesh.faces)),
              "volume_ml": round(abs(float(mesh.volume)) / 1000, 1), "watertight": bool(mesh.is_watertight)}
    return mesh, record


def main() -> None:
    from . import biro_cut
    from .biro_rebuild import write_obj

    spec = json.loads(SOURCE.read_text())
    mesh, record = build_wall(spec["branches"])
    write_obj(mesh, OUT_MESH, "the inscribed-sphere phantom's wall from the model alone (guidewire_rl/env/vessels/biro_spheres.py)")
    out = copy.deepcopy(spec)
    out["what"] = ("the inscribed-sphere phantom (内接球版) from the phantom model alone: the union of the inscribed "
                   "spheres along the model-inner-wall version's centerlines as the wall; the same centerlines with radii "
                   f"at least {R_MIN_MM} mm")
    for b in out["branches"].values():
        b["radii"] = np.maximum(np.asarray(b["radii"], float), R_MIN_MM).round(3).tolist()
    out["radius_median_mm"] = {n: round(float(np.median(out["branches"][n]["radii"])), 2) for n in out["task_branches"]}
    out["source"] = {"centerlines": f"{SOURCE.name} (sha256 {sha256(SOURCE)}), made from the model by biro_rebuild.py",
                     "model": spec["source"]["model"], "script": "guidewire_rl/env/vessels/biro_spheres.py",
                     "wall": f"{OUT_MESH.name} (sha256 {sha256(OUT_MESH)})"}
    out["spheres"] = {**record, "r_min_mm": R_MIN_MM, "grid_mm": GRID_MM}
    OUT_SPEC.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT_MESH.name} ({record['faces']} faces, {record['volume_ml']} ml, watertight {record['watertight']}) "
          f"and {OUT_SPEC.name}")
    biro_cut.write_cut("spheres", OUT_MESH, OUT_SPEC)


if __name__ == "__main__":
    main()
