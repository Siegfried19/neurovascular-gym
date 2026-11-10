"""Give every vessel of the lab phantom a centerline.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_phantom_all

reads ``biro_phantom/phantom_arch_root.json`` (the six training branches, the aorta extended
to the bottom of the root), ``biro_phantom/vessel_tree.npz`` (the lab's vessels) and
``phantom_arch.obj``, and writes ``biro_phantom/phantom_arch_all.json`` (seconds; needs trimesh
and scipy, which only the ``guidewire_rl`` env has).

Why: the collision mesh is the union of balls along all of the lab's vessels
(``19-phantom-sofa/mesh.py``, radius at least 1 mm), but the six training centerlines cover
only about 72 % of it. Where the wire tip is in the rest, the vessel geometry observation says it
is outside the vessel, the reward's path length is measured from whichever of the six is
nearest, and stEVE zeroes the push at the cut ends of rsa, rcca, lcca and lsa (cut to the paper
arch's lengths) although the vessels go on.

What is added, as extra branches named after the lab's vessels:
  - every lab vessel the six do not use, whole;
  - what is left of the vessels the six were cut from: ``v0_inlet`` (v0 below the aorta's
    start, the 70.7 mm up from the phantom's entrance) and ``v6_rest``, ``v7_rest``,
    ``v8_rest``, ``v10_rest`` (past the ends of lsa, lcca, rsa, rcca).
Points are moved into the training frame with the lab's transform (``47-steve-agent-phantom``:
subtract ``origin_in_phantom_mm``, rotate about z by ``rotation_z_deg``) and get radius
max(r, 1 mm), as the mesh does. A vessel whose centerline is not inside the mesh (``mesh.py``
keeps only the largest connected piece) is left out and listed. The six branches and every
other field are copied unchanged; ``task_branches`` names the six (targets and the 2D
coordinate range keep coming from them only, see ``biro_phantom.py``), ``all_vessels``
records what was done.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .biro_phantom import CENTERLINES, DATA_DIR, MESH_PATH

TREE_PATH = DATA_DIR / "vessel_tree.npz"
ROOT_PATH = CENTERLINES["root_extended"]
OUT_PATH = DATA_DIR / "phantom_arch_all.json"
R_MIN_MM = 1.0                          # 19-phantom-sofa/mesh.py R_MIN
CUT_FROM = {"rsa": "v8", "rcca": "v10", "lcca": "v7", "lsa": "v6"}   # 47-steve-agent-phantom OURS
USED = {"v0", "v6", "v7", "v8", "v10"}
INSIDE_SHARE = 0.9                      # share of a vessel's points that must lie inside the mesh


def arclength(points) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def point_at(points, s: float) -> np.ndarray:
    a = arclength(points)
    return np.array([np.interp(s, a, points[:, k]) for k in range(3)])


def nearest_s(points, query) -> tuple[float, float]:
    """Arc length of the polyline point nearest to ``query``, and the distance to it."""
    a = arclength(points)
    start, vec = points[:-1], np.diff(points, axis=0)
    t = np.clip(np.einsum("ij,ij->i", query - start, vec) / np.einsum("ij,ij->i", vec, vec), 0, 1)
    gap = np.linalg.norm(start + t[:, None] * vec - query, axis=1)
    i = int(np.argmin(gap))
    return float(a[i] + t[i] * (a[i + 1] - a[i])), float(gap[i])


def piece(points, radii, s0: float, s1: float):
    """The polyline between arc lengths s0 and s1, its end points interpolated."""
    a = arclength(points)
    s1 = min(s1, a[-1])
    inner = (a > s0 + 1e-6) & (a < s1 - 1e-6)
    coordinates = np.vstack([point_at(points, s0), points[inner], point_at(points, s1)])
    r = np.r_[np.interp(s0, a, radii), radii[inner], np.interp(s1, a, radii)]
    return coordinates, r


def drop_repeats(coordinates, radii, tol: float = 1e-3):
    keep = np.r_[True, np.linalg.norm(np.diff(coordinates, axis=0), axis=1) > tol]
    return coordinates[keep], radii[keep]


def signed_distance(mesh):
    """f(points): distance to the mesh, signed by the face normal at the nearest surface sample
    (the caller fixes which sign is inside; error about 0.1 mm)."""
    import trimesh
    from scipy.spatial import cKDTree

    samples, faces = trimesh.sample.sample_surface(mesh, int(mesh.area * 20), seed=0)
    normals = mesh.face_normals[faces]
    tree = cKDTree(samples)

    def signed(p):
        _, j = tree.query(p, workers=-1)
        return np.einsum("ij,ij->i", p - samples[j], normals[j])

    return signed


def build(root: dict, vessel_tree, mesh, wall_path=MESH_PATH, root_from=None) -> dict:
    """``wall_path`` and ``root_from`` are only recorded: the mesh given is the wall, ``root`` the six."""
    theta = np.radians(root["rotation_z_deg"])
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    origin = np.asarray(root["origin_in_phantom_mm"], float)
    vessels = {k[3:]: ((np.asarray(vessel_tree[k], float) - origin) @ rotation.T,
                       np.maximum(np.asarray(vessel_tree["vr_" + k[3:]], float), R_MIN_MM))
               for k in vessel_tree.files if k.startswith("vp_")}
    six = root["branches"]

    extra, source = {}, {}
    p, r = vessels["v0"]
    start = root["aorta_start_s_mm"]
    extra["v0_inlet"] = piece(p, r, 0.0, start)
    gap = float(np.linalg.norm(extra["v0_inlet"][0][-1] - np.asarray(six["aorta"]["coordinates"][0])))
    if gap > 0.05:
        raise RuntimeError(f"v0 at s={start} is {gap:.3f} mm from the aorta's start")
    source["v0_inlet"] = f"v0, s 0 .. {start} mm"
    for name, vid in CUT_FROM.items():
        p, r = vessels[vid]
        s_end, gap = nearest_s(p, np.asarray(six[name]["coordinates"][-1], float))
        if gap > 0.05:
            raise RuntimeError(f"{name}'s end is {gap:.3f} mm off lab vessel {vid}")
        extra[f"{vid}_rest"] = piece(p, r, s_end, np.inf)
        source[f"{vid}_rest"] = f"{vid}, s {s_end:.1f} mm .. end (past the end of {name})"
    for vid in sorted(set(vessels) - USED, key=lambda v: int(v[1:])):
        extra[vid] = vessels[vid]
        source[vid] = f"{vid}, whole"

    signed = signed_distance(mesh)
    flip = 1.0 if signed(np.asarray(six["aorta"]["coordinates"][100:101], float))[0] > 0 else -1.0
    kept, dropped = {}, {}
    for name, (coordinates, radii) in extra.items():
        coordinates, radii = drop_repeats(coordinates, radii)
        share = float(np.mean(flip * signed(coordinates) > 0))
        if len(coordinates) < 2 or share < INSIDE_SHARE:
            dropped[name] = round(share, 2)
            continue
        kept[name] = (coordinates, radii)

    out = json.loads(json.dumps(root))
    out.pop("root_extension")
    out["task_branches"] = list(six)
    for name, (coordinates, radii) in kept.items():
        out["branches"][name] = {"coordinates": coordinates.round(3).tolist(), "radii": radii.round(3).tolist()}
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    out["all_vessels"] = {
        "what": "every lab vessel added as a centerline branch, so the branch tubes fill the collision mesh",
        "from": f"{root_from or f'phantom_arch_root.json (sha256 {sha(ROOT_PATH)})'} for the six task branches; "
                f"vessel_tree.npz (sha256 {sha(TREE_PATH)}) for the rest",
        "wall": f"{Path(wall_path).name} (sha256 {sha(wall_path)})",
        "script": "guidewire_rl/env/vessels/biro_phantom_all.py",
        "radius": f"max(lab radius, {R_MIN_MM} mm), as 19-phantom-sofa/mesh.py",
        "added": {name: source[name] for name in kept},
        "left_out_not_in_mesh": dropped,
        "root_extension": root["root_extension"],
    }
    return out


def main() -> None:
    import trimesh

    root = json.loads(ROOT_PATH.read_text())
    out = build(root, np.load(TREE_PATH), trimesh.load(MESH_PATH, process=True))
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    info = out["all_vessels"]
    n = sum(len(b["coordinates"]) for b in out["branches"].values())
    print(f"wrote {OUT_PATH}: {len(out['branches'])} branches ({len(info['added'])} added, "
          f"left out {info['left_out_not_in_mesh']}), {n} points")


if __name__ == "__main__":
    main()
