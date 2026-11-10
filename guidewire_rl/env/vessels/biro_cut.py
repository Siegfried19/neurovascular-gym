"""Cut the right arm off a phantom version: the physical phantom has only the chest and the head.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_cut

writes ``biro_phantom/lab_cut.obj`` / ``lab_cut.json`` from the lab version (``phantom_arch.obj``
+ ``phantom_arch_all.json``); ``biro_rebuild.py`` calls ``write_cut`` for the rebuilt version.
Seconds.

The model (the vessel-wall STL) carries the right arm down to the wrist, the physical phantom
does not; the lab never cut it (its lumen, its wall of balls and its tree all keep the arm; lab 47
only shortened the task branches). Where the physical phantom's chest ends on the right side is
not recorded anywhere, so the cut is placed by symmetry with the left side, which the model
already ends at the chest:

  - L = the length of the left subclavian along its centerline, from its origin on the aorta to
    its open end (lsa + its continuation);
  - the right subclavian is cut at the same length from its origin, where the brachiocephalic
    trunk divides (rsa + its continuation), by the plane across the centerline there.

Everything of the wall beyond that plane on the arm's side is removed and the cut is closed with
a flat fan; every centerline piece is cut at the plane, pieces wholly beyond it (the forearm) are
dropped, as are junctions that lose a branch. The right vertebral artery and every other branch
that leaves before the cut stay. The coronary arteries stay (they are in the chest).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).with_name("biro_phantom")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arclength(points) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def chain(branches: dict, first: str) -> tuple[list[str], np.ndarray]:
    """``first`` and, from its end on, the piece that starts there and goes on longest: the names
    and the joined line."""
    names, line = [first], np.asarray(branches[first]["coordinates"], float)
    while True:
        end = line[-1]
        nxt = [n for n, b in branches.items() if n not in names
               and np.linalg.norm(np.asarray(b["coordinates"][0], float) - end) < 0.05]
        if not nxt:
            return names, line
        n = max(nxt, key=lambda n: arclength(np.asarray(branches[n]["coordinates"], float))[-1])
        names.append(n)
        line = np.vstack([line, np.asarray(branches[n]["coordinates"], float)[1:]])


def cut_place(spec: dict) -> dict:
    """Where the right subclavian is cut (see the module docstring)."""
    br = spec["branches"]
    _, left = chain(br, "lsa")
    length = float(arclength(left)[-1])
    names, right = chain(br, "rsa")
    arc = arclength(right)
    if arc[-1] <= length:
        raise RuntimeError("the right subclavian is shorter than the left; nothing to cut")
    point = np.array([np.interp(length, arc, right[:, k]) for k in range(3)])
    ahead = np.array([np.interp(length + 2.0, arc, right[:, k]) for k in range(3)])
    behind = np.array([np.interp(length - 2.0, arc, right[:, k]) for k in range(3)])
    normal = (ahead - behind) / np.linalg.norm(ahead - behind)
    return {"left_subclavian_mm": round(length, 1), "point": point, "normal": normal, "right_chain": names}


def clip_to_plane(mesh, point, normal):
    """The part of the triangles on the near side of the plane (``(x - point) . normal <= 0``),
    triangles crossing it cut at the plane, winding kept. A point on an edge is computed from the
    edge's two vertices in a fixed order, so neighbouring triangles get the same point."""
    import trimesh

    v = np.asarray(mesh.vertices, float)
    d = (v - point) @ normal
    new_v, faces, made = list(v), [], {}

    def crossing(a, b):
        key = (min(a, b), max(a, b))
        if key not in made:
            i, j = key
            t = d[i] / (d[i] - d[j])
            new_v.append(v[i] + t * (v[j] - v[i]))
            made[key] = len(new_v) - 1
        return made[key]

    for tri in np.asarray(mesh.faces):
        inside = d[tri] <= 0
        if inside.all():
            faces.append(tri)
            continue
        if not inside.any():
            continue
        poly = []
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            if d[a] <= 0:
                poly.append(a)
            if (d[a] <= 0) != (d[b] <= 0):
                poly.append(crossing(a, b))
        faces += [[poly[0], poly[k], poly[k + 1]] for k in range(1, len(poly) - 1)]
    return trimesh.Trimesh(np.asarray(new_v), np.asarray(faces, int).reshape(-1, 3), process=False)


def cut_wall(mesh, point, normal, near_mm: float):
    """The wall without the arm beyond the plane, the cut closed with a fan. Faces within
    ``near_mm`` of the cut point are split exactly at the plane; farther away the wall falls apart
    there into the body and the arm, and the arm (the piece next to the cut, beyond the plane) is
    dropped."""
    import trimesh

    from .biro_rebuild import cap_open_ends

    near = np.linalg.norm(mesh.triangles_center - point, axis=1) < near_mm
    local = mesh.submesh([np.flatnonzero(near)], append=True)
    keep = [clip_to_plane(local, point, normal)]
    for part in mesh.submesh([np.flatnonzero(~near)], append=True).split(only_watertight=False):
        beside_cut = np.linalg.norm(part.vertices - point, axis=1).min() < near_mm + 2.0
        if beside_cut and np.median((part.triangles_center - point) @ normal) > 0:
            continue                                                       # the arm
        keep.append(part)
    merged = trimesh.util.concatenate(keep)
    merged.merge_vertices(digits_vertex=5)
    merged.remove_unreferenced_vertices()
    body = max(merged.split(only_watertight=False), key=lambda b: len(b.faces))
    closed, centres, radii = cap_open_ends(body)
    return closed, centres, radii


def cut_spec(spec: dict, place: dict) -> dict:
    """The centerlines cut at the plane: the piece of the right subclavian chain that crosses it is
    shortened, pieces beyond it and everything starting on them are dropped."""
    point, normal = place["point"], place["normal"]
    out = copy.deepcopy(spec)
    br = out["branches"]
    arm, cut_piece, cut_off = set(), None, None
    for name in place["right_chain"]:
        c = np.asarray(br[name]["coordinates"], float)
        side = (c - point) @ normal
        if cut_piece is None and side[-1] > 0:
            cut_piece = name
            k = int(np.argmax(side > 0))                     # first point beyond the plane
            a, b = c[k - 1], c[k]
            t = side[k - 1] / (side[k - 1] - side[k])
            end = a + t * (b - a)
            cut_off = np.vstack([end, c[k:]])                # the removed part, for what starts on it
            r = np.asarray(br[name]["radii"], float)
            br[name]["coordinates"] = np.vstack([c[:k], end]).round(3).tolist()
            br[name]["radii"] = np.r_[r[:k], r[k - 1] + t * (r[k] - r[k - 1])].round(3).tolist()
        elif cut_piece is not None:
            arm.add(name)
    grown = True
    while grown:                                             # anything that starts on the arm
        grown = False
        lines = {n: np.asarray(br[n]["coordinates"], float) for n in arm}
        if cut_off is not None:
            lines[None] = cut_off
        for name, b in br.items():
            if name in arm or name == cut_piece:
                continue
            start = np.asarray(b["coordinates"][0], float)
            on_arm = any(np.linalg.norm(line - start, axis=1).min() < 0.5 for line in lines.values())
            if on_arm and (start - point) @ normal > 0:
                arm.add(name)
                grown = True
    for name in arm:
        del br[name]
    if "junctions" in out:
        out["junctions"] = [{"point": j["point"], "branches": [n for n in j["branches"] if n in br]}
                            for j in out["junctions"]]
        out["junctions"] = [j for j in out["junctions"] if len(j["branches"]) >= 2]
    if "lengths_mm" in out:
        out["lengths_mm"] = {n: round(float(arclength(np.asarray(b["coordinates"], float))[-1]), 1)
                             for n, b in br.items()}
    out["cut"] = {"what": "right arm cut off: the physical phantom has only the chest and the head",
                  "rule": "the right subclavian cut at the left subclavian's length from its origin "
                          "(biro_cut.py)",
                  "left_subclavian_mm": place["left_subclavian_mm"],
                  "point": np.round(point, 3).tolist(), "normal": np.round(normal, 6).tolist(),
                  "shortened": cut_piece, "dropped": sorted(arm)}
    return out


def write_obj(mesh, path, title: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"# {title}\n")
        for v in mesh.vertices:
            fh.write("v %.5f %.5f %.5f\n" % tuple(v))
        for f in np.asarray(mesh.faces) + 1:
            fh.write("f %d %d %d\n" % tuple(f))


def write_cut(prefix: str, mesh_path, spec_path) -> tuple[Path, Path]:
    """``<prefix>_cut.obj`` and ``<prefix>_cut.json`` next to the inputs."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    spec = json.loads(Path(spec_path).read_text())
    place = cut_place(spec)
    br = spec["branches"]
    names, line = chain(br, "rsa")
    radii = np.concatenate([np.asarray(br[n]["radii"], float) for n in names])
    r_cut = float(radii[np.argmin(np.linalg.norm(
        np.vstack([np.asarray(br[n]["coordinates"], float) for n in names]) - place["point"], axis=1))])
    near = 2.0 * r_cut + 3.0
    wall, _, _ = cut_wall(mesh, place["point"], place["normal"], near)
    if not wall.is_watertight and mesh.is_watertight:
        raise RuntimeError(f"{prefix}: the cut wall is not closed")
    out_mesh, out_spec = DATA_DIR / f"{prefix}_cut.obj", DATA_DIR / f"{prefix}_cut.json"
    write_obj(wall, out_mesh, f"{prefix} phantom without the right arm (guidewire_rl/env/vessels/biro_cut.py)")
    cut = cut_spec(spec, place)
    cut["cut"].update({"from_wall": f"{Path(mesh_path).name} (sha256 {sha256(mesh_path)})",
                       "from_centerlines": f"{Path(spec_path).name} (sha256 {sha256(spec_path)})",
                       "wall": f"{out_mesh.name} (sha256 {sha256(out_mesh)})",
                       "split_within_mm": round(near, 2),
                       "wall_volume_ml": round(abs(float(wall.volume)) / 1000, 1),
                       "wall_faces": int(len(wall.faces))})
    out_spec.write_text(json.dumps(cut, indent=1) + "\n")
    print(f"wrote {out_mesh.name} ({len(wall.faces)} faces, {abs(wall.volume) / 1000:.1f} ml) and {out_spec.name}: "
          f"cut {place['left_subclavian_mm']} mm along the right subclavian at {np.round(place['point'], 1)}, "
          f"dropped {sorted(cut['cut']['dropped'])}")
    return out_mesh, out_spec


def main() -> None:
    write_cut("lab", DATA_DIR / "phantom_arch.obj", DATA_DIR / "phantom_arch_all.json")


if __name__ == "__main__":
    main()
