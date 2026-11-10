"""Extend the phantom aorta's centerline to the bottom of the aortic root.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_phantom_root

reads ``biro_phantom/phantom_arch.json`` and ``phantom_arch.obj`` (both unchanged) and writes
``biro_phantom/phantom_arch_root.json`` (about a minute; needs trimesh, scipy and
scikit-image, which only the ``guidewire_rl`` env has).

Why: the lab's aorta centerline stops at z 20.6, but the collision mesh's aortic root goes
down to z 4.4. stEVE uses the centerline's end as the end of the vessel: it zeroes the
push once the tip is past the last point (``MonoPlaneStatic(stop_instruments_at_tree_end)``),
and the path length in the reward and the vessel geometry observation only know the
centerline. The lab line stops early because a skeleton cannot reach a closed end (the
largest ball sitting on the bottom has its centre one radius above it) and because the
lab's skeleton splits there into the two coronary arteries.

Method: the standard centerline definition. Between two end points the centerline is the
path that stays as far from the wall as it can: VMTK takes the path on the medial surface
that minimises the integral of 1 / R, R being the distance to the wall (the radius of the
largest inscribed sphere); each point's radius is R there. We search the whole lumen instead
of the medial surface, so the cost is 1 / R**2 (``COST_POWER``), which keeps the path on the
ridge of R: every new point is within 0.35 mm of the largest R in its cross-section (0.7 mm
with 1 / R). The wall is the collision mesh, the surface the simulated wire touches.

1. Signed distance to the wall (positive inside) on a 0.25 mm grid around the root.
2. End point: the lowest grid point within 15 mm sideways of the lab line's end whose distance
   to the wall is at least ``END_RADIUS_MM`` (1 mm; a wire tip's centre gets no closer to the
   wall: wire radius 0.35-0.45 mm plus the 0.3 mm contact distance).
3. The minimal path from lab point ``START_INDEX`` to the end point. The lab line is kept for
   as long as it stays within ``KEEP_TOLERANCE_MM`` of that path (up to point 167). After that
   it heads for the junction where the coronaries leave rather than for the bottom (its last
   four points are up to 2.1 mm off), so those points are replaced.
4. The path is smoothed (Gaussian, sigma 1.5 mm along its length, both ends pinned),
   resampled at the aorta's point spacing, and each new point gets its exact distance to the
   wall as radius.

The other five branches, the insertion point and every other field are copied unchanged;
``lengths_mm.aorta`` is recomputed and ``root_extension`` records what was done.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .biro_phantom import DATA_DIR, MESH_PATH, SPEC_PATH

OUT_PATH = DATA_DIR / "phantom_arch_root.json"
GRID_MM = 0.25
END_RADIUS_MM = 1.0
END_SIDEWAYS_MM = 15.0
START_INDEX = 156
KEEP_TOLERANCE_MM = 0.3
SMOOTH_SIGMA_MM = 1.5
COST_POWER = 2
BOX_HALF_WIDTH_MM = 25.0
BOX_Z_MM = (0.0, 45.0)


class Wall:
    """Exact signed distance to the collision mesh near the root, positive inside the lumen."""

    def __init__(self, mesh, low, high, inside_point, margin: float = 4.0) -> None:
        from scipy.spatial import cKDTree

        centres = mesh.triangles_center
        keep = np.all((centres > np.asarray(low) - margin) & (centres < np.asarray(high) + margin), axis=1)
        self.triangles = mesh.triangles[keep]
        self.normals = mesh.face_normals[keep]
        self.tree = cKDTree(self.triangles.mean(axis=1))
        self.reach = float(np.linalg.norm(
            self.triangles - self.triangles.mean(axis=1, keepdims=True), axis=2).max())
        self.flip = 1.0
        self.flip = 1.0 if self.distance(np.asarray(inside_point, float)[None])[0] > 0 else -1.0

    def distance(self, points) -> np.ndarray:
        import trimesh

        points = np.atleast_2d(np.asarray(points, float))
        out = np.empty(len(points))
        for i, point in enumerate(points):
            nearest, _ = self.tree.query(point)
            candidates = np.asarray(self.tree.query_ball_point(point, nearest + 2 * self.reach))
            closest = trimesh.triangles.closest_point(self.triangles[candidates],
                                                      np.repeat(point[None], len(candidates), 0))
            gaps = np.linalg.norm(closest - point, axis=1)
            best = int(np.argmin(gaps))
            # at an edge or a vertex several faces are equally near: average their normals
            normal = self.normals[candidates[gaps <= gaps[best] + 1e-6]].mean(axis=0)
            sign = np.sign(np.dot(point - closest[best], normal)) or 1.0
            out[i] = self.flip * sign * gaps[best]
        return out

    def grid(self, low, high):
        """Signed distance on a GRID_MM grid, from dense surface samples (error ~0.05 mm)."""
        import trimesh
        from scipy.spatial import cKDTree

        surface = trimesh.Trimesh(self.triangles.reshape(-1, 3),
                                  np.arange(3 * len(self.triangles)).reshape(-1, 3), process=False)
        samples, faces = trimesh.sample.sample_surface(surface, int(surface.area * 60), seed=0)
        normals = self.normals[faces] * self.flip
        axes = [np.arange(low[k], high[k] + 1e-9, GRID_MM) for k in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        flat = points.reshape(-1, 3)
        gap, nearest = cKDTree(samples).query(flat, workers=-1)
        sign = np.sign(np.einsum("ij,ij->i", flat - samples[nearest], normals[nearest]))
        return points, (sign * gap).reshape(points.shape[:3])


def minimal_path(points, distance, start, end) -> np.ndarray:
    """Grid path from start to end through the lumen minimising the sum of length / R**COST_POWER."""
    from scipy.ndimage import label
    from skimage.graph import MCP_Geometric

    low = points[0, 0, 0]
    index = lambda p: tuple(int(round((p[k] - low[k]) / GRID_MM)) for k in range(3))
    regions, _ = label(distance > 0)
    lumen = regions == regions[index(start)]
    if not lumen[index(end)]:
        raise RuntimeError("start and end are not in the same lumen")
    cost = np.where(lumen, 1.0 / np.maximum(distance, 0.05) ** COST_POWER, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    mcp.find_costs([index(start)], [index(end)])
    path = np.asarray(mcp.traceback(index(end)), float) * GRID_MM + low
    path[0], path[-1] = start, end
    return path


def smooth(path, sigma_mm: float = SMOOTH_SIGMA_MM, spacing: float = GRID_MM / 2) -> np.ndarray:
    """Gaussian smoothing along the arc length; the correction fades out towards both ends."""
    from scipy.ndimage import gaussian_filter1d

    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    u = np.arange(0.0, arc[-1], spacing)
    u = np.r_[u, arc[-1]] if arc[-1] - u[-1] > 1e-9 else u
    dense = np.stack([np.interp(u, arc, path[:, k]) for k in range(3)], axis=1)
    blurred = gaussian_filter1d(dense, sigma_mm / spacing, axis=0, mode="nearest")
    weight = np.clip(np.minimum(u, u[-1] - u) / (3 * sigma_mm), 0, 1)[:, None]
    return dense + weight * (blurred - dense)


def resample(line, step: float) -> np.ndarray:
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    n = max(int(round(arc[-1] / step)), 1)
    t = np.linspace(0.0, arc[-1], n + 1)
    return np.stack([np.interp(t, arc, line[:, k]) for k in range(3)], axis=1)


def extend(spec: dict, mesh, wall_path=MESH_PATH) -> dict:
    """``wall_path`` is only recorded: the mesh given is the wall."""
    from scipy.spatial import cKDTree

    aorta = np.asarray(spec["branches"]["aorta"]["coordinates"], float)
    radii = np.asarray(spec["branches"]["aorta"]["radii"], float)
    step = float(np.median(np.linalg.norm(np.diff(aorta, axis=0), axis=1)))
    end = aorta[-1]
    low = np.array([end[0] - BOX_HALF_WIDTH_MM, end[1] - BOX_HALF_WIDTH_MM, BOX_Z_MM[0]])
    high = np.array([end[0] + BOX_HALF_WIDTH_MM, end[1] + BOX_HALF_WIDTH_MM, BOX_Z_MM[1]])
    wall = Wall(mesh, low, high, inside_point=aorta[START_INDEX])
    points, distance = wall.grid(low, high)

    sideways = np.linalg.norm(points[..., :2] - end[:2], axis=-1)
    candidates = points[(distance >= END_RADIUS_MM) & (sideways <= END_SIDEWAYS_MM)]
    bottom = candidates[np.argmin(candidates[:, 2])]

    line = smooth(minimal_path(points, distance, aorta[START_INDEX], bottom))
    offset = cKDTree(line).query(aorta[START_INDEX:])[0]
    off = np.flatnonzero(offset > KEEP_TOLERANCE_MM)
    kept = START_INDEX + (int(off[0]) if len(off) else len(offset))
    junction = int(np.argmin(np.linalg.norm(line - aorta[kept - 1], axis=1)))
    tail = resample(np.vstack([aorta[kept - 1], line[junction + 1:]]), step)[1:]
    tail_radii = wall.distance(tail)

    coordinates = np.vstack([aorta[:kept], tail.round(3)])
    new_radii = np.r_[radii[:kept], tail_radii.round(3)]
    out = json.loads(json.dumps(spec))
    out["branches"]["aorta"] = {"coordinates": coordinates.tolist(), "radii": new_radii.tolist()}
    out["lengths_mm"]["aorta"] = round(float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum()), 1)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    out["root_extension"] = {
        "what": "aorta centerline extended to the bottom of the aortic root",
        "from": f"phantom_arch.json (sha256 {sha(SPEC_PATH)})",
        "wall": f"{Path(wall_path).name} (sha256 {sha(wall_path)})",
        "script": "guidewire_rl/env/vessels/biro_phantom_root.py",
        "method": f"path through the lumen minimising the integral of 1/R^{COST_POWER} "
                  "(R = distance to the wall), smoothed and resampled; radius = exact distance to the wall",
        "kept_lab_points": kept,
        "replaced_lab_points": len(aorta) - kept,
        "new_points": len(tail),
        "max_offset_of_replaced_lab_points_mm": round(float(offset[kept - START_INDEX:].max()), 2),
        "end_point": tail[-1].round(3).tolist(),
        "end_radius_mm": round(float(tail_radii[-1]), 3),
        "settings": {"grid_mm": GRID_MM, "end_radius_mm": END_RADIUS_MM, "end_sideways_mm": END_SIDEWAYS_MM,
                     "start_index": START_INDEX, "keep_tolerance_mm": KEEP_TOLERANCE_MM,
                     "smooth_sigma_mm": SMOOTH_SIGMA_MM, "cost_power": COST_POWER},
    }
    return out


def main() -> None:
    import trimesh

    spec = json.loads(SPEC_PATH.read_text())
    out = extend(spec, trimesh.load(MESH_PATH, process=True))
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    info = out["root_extension"]
    print(f"wrote {OUT_PATH}: kept {info['kept_lab_points']} lab points, replaced "
          f"{info['replaced_lab_points']}, added {info['new_points']}; end {info['end_point']}, "
          f"radius {info['end_radius_mm']} mm")


if __name__ == "__main__":
    main()
