"""The phantom's own inner wall surface as a collision mesh.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_phantom_lumen

reads the phantom model and the lab's voxel lumen (both only in the local ``reference/``
checkout, see ``SOURCES``) and ``biro_phantom/phantom_arch.json`` (the lab's transform into the
training frame), and writes ``biro_phantom/phantom_lumen.obj`` with ``phantom_lumen.json``
(what was done, the numbers below). Seconds; needs trimesh, scipy and pyvista (``guidewire_rl``
env only).

Why: the lab's collision mesh (``phantom_arch.obj``, ``19-phantom-sofa/mesh.py``) is the union
of the largest balls that fit along the lab's centerlines. It is smooth, but it fills only part
of the lumen: 208 ml against the 256 ml inside this surface; its wall sits a median 0.55 mm
inside the real one and up to ~9 mm at junctions and the aortic root, where balls cannot fill
the corners. The lab chose balls because a surface made from its 0.5 mm voxels is a staircase
that snags a simulated wire. The phantom model itself is a smooth designed surface, so this
takes its inner face directly.

How:
  1. The model (``biro-phantom_vessel-wall_v2.stl``, Weibo Gao, 2026-10-08; the copy in
     Wilson's repo is byte-identical) is the silicone wall: one closed shell made of the inner
     face, the outer face and the rims at the open vessel ends. Its normals point out of the
     silicone.
  2. A face is on the inner face when a point a little way along its normal (0.4, 0.7, 1.0 or
     1.4 mm) is in the lab's voxel lumen (``18-biro-phantom/derived/lumen.npz``, 0.5 mm); outer
     faces and rims face the air. Only the largest connected piece is kept (the rest are a few
     dozen fragments where the voxel lumen misses distal coronaries).
  3. Each open vessel end (34, the lab skeleton's 34 end points) is closed with a fan around
     its centroid, wound like the faces next to it, so the mesh is watertight with its normals
     pointing into the lumen, as the lab's mesh.
  4. Quadric decimation to ``FACES`` triangles with volume preservation (the call the lab uses),
     then the lab's transform (subtract ``origin_in_phantom_mm``, rotate about z by
     ``rotation_z_deg``).
Everything in the model is kept, the right arm included (the physical phantom has only the
chest and head; trimming it is left for later).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from guidewire_rl import WORKSPACE_ROOT

from .biro_phantom import DATA_DIR, SPEC_PATH

SOURCES = {
    "model": (WORKSPACE_ROOT / "reference" / "contact-aware-tdmpc2-training" / "VasselTree" / "Neurovascular model.stl",
              "d75daffc4e94e94c3ecb8a71031cbdbf3becfd8b29af2e38eecdd3a60f0067f3"),
    "voxel_lumen": (WORKSPACE_ROOT / "reference" / "surgery-all-in-one" / "18-biro-phantom" / "derived" / "lumen.npz",
                    "7ff66ef63a3adfc7c0931e3c3ea76265b8a236503e49f3aa58317c23585a4b36"),
}
OUT_MESH = DATA_DIR / "phantom_lumen.obj"
OUT_INFO = DATA_DIR / "phantom_lumen.json"
FACES = 120_000
PROBE_MM = (0.4, 0.7, 1.0, 1.4)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inner_faces(model, voxels) -> np.ndarray:
    shape = tuple(int(v) for v in voxels["shape"])
    spacing, origin = float(voxels["spacing"]), np.asarray(voxels["origin"], float)
    lumen = np.unpackbits(voxels["bits"])[:int(np.prod(shape))].reshape(shape).astype(bool)
    normals = model.face_normals * (1.0 if model.volume > 0 else -1.0)      # out of the silicone
    centres = model.triangles_center
    inner = np.zeros(len(centres), bool)
    for d in PROBE_MM:
        idx = np.floor((centres + d * normals - origin) / spacing).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        inner[ok] |= lumen[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return inner


def cap_open_ends(mesh):
    """Close every boundary loop with a fan around its centroid, wound like its neighbours."""
    import scipy.sparse as sp
    import trimesh
    from scipy.sparse.csgraph import connected_components

    faces = mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, count = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = directed[count[inverse.ravel()] == 1]
    n = len(mesh.vertices)
    graph = sp.coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    centres, fans, radii = [], [], []
    for k, loop in enumerate(np.unique(label[boundary[:, 0]])):
        edges = boundary[label[boundary[:, 0]] == loop]
        ring = mesh.vertices[np.unique(edges)]
        centres.append(ring.mean(axis=0))
        radii.append(float(np.linalg.norm(ring - ring.mean(axis=0), axis=1).mean()))
        fans.append(np.column_stack([edges[:, 1], edges[:, 0], np.full(len(edges), n + k)]))
    capped = trimesh.Trimesh(np.vstack([mesh.vertices, centres]), np.vstack([faces] + fans), process=False)
    return capped, np.asarray(centres), radii


def build():
    import pyvista as pv
    import trimesh

    for name, (path, expected) in SOURCES.items():
        if sha256(path) != expected:
            raise RuntimeError(f"{name}: {path} is not the expected file")
    model = trimesh.load(SOURCES["model"][0], process=True)
    inner = inner_faces(model, np.load(SOURCES["voxel_lumen"][0]))
    pieces = model.submesh([np.flatnonzero(inner)], append=True).split(only_watertight=False)
    surface = max(pieces, key=lambda p: len(p.faces))
    capped, ends, end_radii = cap_open_ends(surface)
    if not (capped.is_watertight and capped.is_winding_consistent):
        raise RuntimeError("capped inner surface is not closed")
    poly = pv.PolyData(capped.vertices, np.column_stack([np.full(len(capped.faces), 3), capped.faces]).ravel())
    small = poly.decimate(1.0 - FACES / poly.n_cells, volume_preservation=True).triangulate().clean()
    mesh = trimesh.Trimesh(small.points, small.faces.reshape(-1, 4)[:, 1:], process=False)
    if not mesh.is_watertight:
        raise RuntimeError("decimated mesh is not closed")

    spec = json.loads(SPEC_PATH.read_text())
    theta = np.radians(spec["rotation_z_deg"])
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    place = lambda x: (np.asarray(x, float) - np.asarray(spec["origin_in_phantom_mm"])) @ rotation.T

    # how far the decimated surface strays from the model's inner face (caps left out)
    reference, _ = trimesh.sample.sample_surface(surface, 3_000_000, seed=0)
    decimated, _ = trimesh.sample.sample_surface(mesh, 3_000_000, seed=1)
    from scipy.spatial import cKDTree
    stray, _ = cKDTree(decimated).query(reference[::10], workers=-1)

    mesh.vertices = place(mesh.vertices)
    info = {
        "what": "the phantom model's inner wall surface, capped at its open ends, as a collision mesh",
        "model": f"biro-phantom_vessel-wall_v2.stl (sha256 {SOURCES['model'][1]}; Weibo Gao 2026-10-08; "
                 "copy: contact-aware-tdmpc2-training/VasselTree/Neurovascular model.stl)",
        "voxel_lumen": f"18-biro-phantom/derived/lumen.npz (sha256 {SOURCES['voxel_lumen'][1]}), used only to tell "
                       "inner faces from outer ones",
        "transform": f"phantom_arch.json (sha256 {sha256(SPEC_PATH)}): origin_in_phantom_mm, rotation_z_deg",
        "script": "guidewire_rl/env/vessels/biro_phantom_lumen.py",
        "model_faces": int(len(model.faces)),
        "inner_faces": int(inner.sum()),
        "fragments_dropped": int(len(pieces) - 1),
        "open_ends_capped": [{"centre": c.round(2).tolist(), "mean_radius_mm": round(r, 2)}
                             for c, r in zip(place(ends), end_radii)],
        "faces": int(len(mesh.faces)),
        "volume_ml": round(abs(float(mesh.volume)) / 1000, 1),
        "model_inner_face_to_mesh_mm": {"median": round(float(np.median(stray)), 3),
                                        "p99": round(float(np.percentile(stray, 99)), 3),
                                        "max": round(float(stray.max()), 3)},
        "settings": {"faces": FACES, "probe_mm": list(PROBE_MM)},
    }
    return mesh, info


def write_obj(mesh, path) -> None:
    with open(path, "w") as fh:
        fh.write("# the phantom model's inner wall surface (guidewire_rl/env/vessels/biro_phantom_lumen.py)\n")
        for v in mesh.vertices:
            fh.write("v %.5f %.5f %.5f\n" % tuple(v))
        for f in mesh.faces + 1:
            fh.write("f %d %d %d\n" % tuple(f))


def main() -> None:
    mesh, info = build()
    write_obj(mesh, OUT_MESH)
    info["sha256"] = sha256(OUT_MESH)
    OUT_INFO.write_text(json.dumps(info, indent=1) + "\n")
    print(f"wrote {OUT_MESH}: {info['faces']} faces, {info['volume_ml']} ml, {len(info['open_ends_capped'])} ends capped, "
          f"model inner face to mesh median {info['model_inner_face_to_mesh_mm']['median']} / "
          f"p99 {info['model_inner_face_to_mesh_mm']['p99']} / max {info['model_inner_face_to_mesh_mm']['max']} mm")


if __name__ == "__main__":
    main()
