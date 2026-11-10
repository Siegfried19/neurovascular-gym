"""The rebuilt phantom: everything from the phantom model (STL) alone, with our own code.

    conda run --no-capture-output -n guidewire_rl python -m guidewire_rl.env.vessels.biro_rebuild

reads only the phantom model (``SOURCE``, the vessel-wall STL) and writes
``biro_phantom/rebuilt_full.obj`` and ``rebuilt_full.json`` (collision wall, centerlines,
task branches, junctions, insertion), then cuts the right arm off and writes
``rebuilt_cut.obj`` / ``rebuilt_cut.json`` (``biro_cut.py``). A few minutes; ``guidewire_rl``
env only (trimesh, scipy, pyvista, scikit-image).

What is fixed by hand, and nothing else:
  - ``FRAME``: the training frame, the rotation about z and the origin lab 47 fitted to the
    paper's test arch. Every phantom version uses the same frame, so targets, the 2D range and
    C-arm angles stay comparable.
  - ``TASK``: lab 47's task definition (the test arch's numbers): the aorta starts 146.7 mm
    before its first branch, side branches are cut to the test arch's lengths, the
    brachiocephalic trunk ends where the right common carotid leaves; point spacing.
  - The settings below (probe distances, grid, cost power, end radius, smoothing, the 50 degree
    rule).

Steps:
  1. Inner wall. The model is the silicone wall: one closed shell (inner face, outer face, the
     rims at the open vessel ends), normals out of the silicone. A face is on the inner face
     when a point a little way along its normal (``PROBE_MM``) is enclosed: the wall blocks it on
     both sides along at least two of the three axes (lab 18's rule for its voxels, here tested
     with rays along the axes, no voxels). The largest connected piece is kept, each open end is
     closed with a fan, the surface is decimated to ``FACES`` triangles (volume preserving) and
     moved into ``FRAME``.
  2. Centerlines, by the standard definition (as for the aortic root, ``biro_phantom_root.py``):
     between two ends the centerline is the path through the lumen that minimises the integral
     of 1 / R**2, R the distance to the wall; each point's radius is R. The lumen is filled with
     a ``GRID_MM`` grid, R comes from dense wall samples, one Dijkstra from the phantom's
     entrance (its widest open end) gives the path to every other open end; then lumen left
     uncovered by the paths is either a loop (joined at both ends), a closed vessel (a path to
     its end is added) or a bulge (left alone), see ``find_paths``; the aorta is extended to the
     bottom of its root (``extend_aorta``). The paths share their common start, so apart from
     the loops they are a tree; each piece between two junctions is smoothed and resampled and
     every point gets its exact distance to the wall as radius.
  3. Vessels and names. At each junction the vessel carries on into a branch that turns by less
     than 90 degrees and keeps at least half its size: the largest if clearly larger (1.25 times),
     otherwise the straightest that is not a short closed stub (``build_vessels``; lab 19 took the
     straightest within 50 degrees whatever the size, which on this wall follows the left
     subclavian instead of the arch); the others start new vessels. The
     aorta is the vessel from the entrance; its three largest branches near the top of the arch
     are, from the entrance, the left subclavian, the left common carotid and the
     brachiocephalic trunk; where the trunk divides, the branch rising more is the right common
     carotid and the other the right subclavian. The six task branches are cut out by ``TASK``;
     what is left of a vessel is kept as ``<name>_rest`` / ``aorta_inlet``, every other vessel as
     ``v<n>``. ``junctions`` lists where pieces meet (the simulator joins branches there instead
     of guessing from the radii).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from guidewire_rl import WORKSPACE_ROOT

DATA_DIR = Path(__file__).with_name("biro_phantom")
SOURCE = (WORKSPACE_ROOT / "reference" / "contact-aware-tdmpc2-training" / "VasselTree" / "Neurovascular model.stl",
          "d75daffc4e94e94c3ecb8a71031cbdbf3becfd8b29af2e38eecdd3a60f0067f3")
OUT_MESH, OUT_SPEC = DATA_DIR / "rebuilt_full.obj", DATA_DIR / "rebuilt_full.json"

FRAME = {"rotation_z_deg": 37.67, "origin_in_phantom_mm": [45.661, 107.61, -312.8]}     # lab 47 (phantom_arch.json)
TASK = {"first_ostium_from_start_mm": 146.7,                                           # lab 47, from the test arch
        "branch_lengths_mm": {"rcca": 37.8, "rsa": 42.4, "lcca": 58.9, "lsa": 58.9},
        "step_aorta_mm": 1.47, "step_branch_mm": 0.73}

PROBE_MM = (0.4, 0.7, 1.0, 1.4)     # how far along a face's normal the enclosure is tested
RAY_GRID_MM = 0.2                   # spacing of the rays along the axes for that test
FACES = 120_000
GRID_MM = 0.3                       # lumen grid for the centerline search
SAMPLES_PER_MM2 = 60                # wall samples for R
COST_POWER = 2
MIN_RADIUS_MM = 0.3                 # no centerline radius below this (the narrowest lumen)
END_RADIUS_MM = 1.0                 # a closed end's centerline stops where R falls to this
COVER_MM = 1.0                      # lumen this far outside every path's inscribed spheres is uncovered
MIN_POCKET_MM3 = 5.0                # smaller uncovered pockets are ignored
LOOP_FACTOR, LOOP_MARGIN_MM = 3.0, 20.0   # two touches farther apart along the paths than this are a loop
ROUNDS = 12
STUB_MM = 30.0                      # a branch ending within this does not carry a vessel on
SIZE_MM = 20.0                      # a branch's size: median radius over its first this many mm
ARCH_DEPTH_MM = 40.0                # the arch's branches leave within this of the top of the aorta
RISE_MM = 20.0                      # over this length the right common carotid rises more than the subclavian
ROOT_REACH_MM = 15.0                # the root's pocket touches the paths within this of the aorta's end
SMOOTH_SIGMA_MM = 1.5
DIRECTION_MM = 5.0                  # length over which a piece's direction is taken
JITTER = np.array([0.1234567, 0.2718281, 0.3141592])   # keeps grid rays off mesh edges and vertices


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arclength(points) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def place(points) -> np.ndarray:
    """Phantom STL frame -> training frame."""
    t = np.radians(FRAME["rotation_z_deg"])
    rotation = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(points, float) - np.asarray(FRAME["origin_in_phantom_mm"])) @ rotation.T


# ------------------------------------------------------------------ rays along the axes --
class AxisRays:
    """Every crossing of the triangles with rays parallel to one axis, through a square grid of
    the other two coordinates (spacing ``h``)."""

    def __init__(self, triangles: np.ndarray, axis: int, h: float, chunk: int = 20_000_000) -> None:
        self.axis, self.h = axis, h
        self.others = [k for k in range(3) if k != axis]
        flat = triangles[:, :, self.others]
        self.lo = flat.reshape(-1, 2).min(axis=0) - 2 * h + JITTER[self.others] * h
        self.shape = (np.ceil((flat.reshape(-1, 2).max(axis=0) - self.lo) / h).astype(int) + 2)
        lo_i = np.ceil((flat.min(axis=1) - self.lo) / h).astype(np.int64)
        hi_i = np.floor((flat.max(axis=1) - self.lo) / h).astype(np.int64)
        n = np.maximum(hi_i - lo_i + 1, 0)
        counts = n[:, 0] * n[:, 1]
        cells, coords = [], []
        csum = np.cumsum(counts)
        start = 0
        while start < len(counts):
            base = csum[start - 1] if start else 0
            stop = max(int(np.searchsorted(csum, base + chunk, "right")), start + 1)
            idx = np.arange(start, stop)
            c = counts[idx]
            tri = np.repeat(idx, c)
            local = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
            nj = n[tri, 1]
            ii = lo_i[tri, 0] + local // np.maximum(nj, 1)
            jj = lo_i[tri, 1] + local % np.maximum(nj, 1)
            gx, gy = self.lo[0] + ii * h, self.lo[1] + jj * h
            a, b, cc = flat[tri, 0], flat[tri, 1], flat[tri, 2]
            e0, e1 = b - a, cc - a
            px, py = gx - a[:, 0], gy - a[:, 1]
            det = e0[:, 0] * e1[:, 1] - e1[:, 0] * e0[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (px * e1[:, 1] - e1[:, 0] * py) / det
                v = (e0[:, 0] * py - px * e0[:, 1]) / det
            inside = (det != 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
            z = triangles[tri, :, axis]
            coord = z[:, 0] + u * (z[:, 1] - z[:, 0]) + v * (z[:, 2] - z[:, 0])
            cells.append((ii * self.shape[1] + jj)[inside])
            coords.append(coord[inside])
            start = stop
        cell, coord = np.concatenate(cells), np.concatenate(coords)
        order = np.lexsort((coord, cell))
        self.cell, self.coord = cell[order], coord[order]

    def cell_of(self, points) -> np.ndarray:
        ij = np.rint((np.asarray(points)[:, self.others] - self.lo) / self.h).astype(np.int64)
        ij = np.clip(ij, 0, self.shape - 1)
        return ij[:, 0] * self.shape[1] + ij[:, 1]

    def blocked_both_sides(self, points) -> np.ndarray:
        """True where the ray through the point crosses the wall both before and after it."""
        cell = self.cell_of(points)
        first = np.searchsorted(self.cell, cell, "left")
        last = np.searchsorted(self.cell, cell, "right")
        has = last > first
        x = np.asarray(points)[:, self.axis]
        lo = np.where(has, self.coord[np.minimum(first, len(self.coord) - 1)], np.inf)
        hi = np.where(has, self.coord[np.maximum(last - 1, 0)], -np.inf)
        return has & (lo < x) & (hi > x)

    def inside_runs(self, step: float):
        """Grid points (spacing ``step`` along the axis) between pairs of crossings, column by column:
        the inside of a closed surface by parity. Returns (cell, index along the axis), and the
        number of columns whose crossings did not pair up."""
        bounds = np.flatnonzero(np.r_[True, self.cell[1:] != self.cell[:-1], True])
        starts, counts = bounds[:-1], np.diff(bounds)
        odd = counts % 2 == 1
        pairs_cell, pairs_in, pairs_out = [], [], []
        for k in range(0, int(counts.max(initial=0)), 2):
            ok = (counts > k + 1) & ~odd
            pairs_cell.append(self.cell[starts[ok]])
            pairs_in.append(self.coord[starts[ok] + k])
            pairs_out.append(self.coord[starts[ok] + k + 1])
        cell, z0, z1 = (np.concatenate(x) for x in (pairs_cell, pairs_in, pairs_out))
        self.lo_axis = float(np.floor(self.coord.min() / step) * step - step) + JITTER[self.axis] * step
        k0 = np.ceil((z0 - self.lo_axis) / step).astype(np.int64)
        k1 = np.floor((z1 - self.lo_axis) / step).astype(np.int64)
        n = np.maximum(k1 - k0 + 1, 0)
        cells = np.repeat(cell, n)
        ks = np.repeat(k0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        return cells, ks, int(odd.sum())


# -------------------------------------------------------------------------- 1. inner wall --
def inner_faces(model) -> np.ndarray:
    """Faces of the silicone shell that face the lumen (see the module docstring)."""
    normals = model.face_normals * (1.0 if model.volume > 0 else -1.0)      # out of the silicone
    centres = model.triangles_center
    rays = [AxisRays(model.triangles, axis, RAY_GRID_MM) for axis in range(3)]
    inner = np.zeros(len(centres), bool)
    for d in PROBE_MM:
        probe = centres + d * normals
        blocked = sum(r.blocked_both_sides(probe).astype(int) for r in rays)
        inner |= blocked >= 2
    return inner


def cap_open_ends(mesh):
    """Close every boundary loop with a fan around its centroid, wound like its neighbours.
    Returns the closed mesh, and per loop its centre and mean radius."""
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
    return capped, np.asarray(centres), np.asarray(radii)


def decimate(mesh, faces: int):
    import pyvista as pv
    import trimesh

    poly = pv.PolyData(mesh.vertices, np.column_stack([np.full(len(mesh.faces), 3), mesh.faces]).ravel())
    small = poly.decimate(1.0 - faces / poly.n_cells, volume_preservation=True).triangulate().clean()
    return trimesh.Trimesh(small.points, small.faces.reshape(-1, 4)[:, 1:], process=False)


def build_wall():
    """Step 1. Returns the wall in the training frame, the open ends (centre, radius; training
    frame) and a record."""
    import trimesh

    if sha256(SOURCE[0]) != SOURCE[1]:
        raise RuntimeError(f"{SOURCE[0]} is not the expected phantom model")
    model = trimesh.load(SOURCE[0], process=True)
    inner = inner_faces(model)
    pieces = model.submesh([np.flatnonzero(inner)], append=True).split(only_watertight=False)
    surface = max(pieces, key=lambda p: len(p.faces))
    capped, ends, end_radii = cap_open_ends(surface)
    if not (capped.is_watertight and capped.is_winding_consistent):
        raise RuntimeError("capped inner surface is not closed")
    mesh = decimate(capped, FACES)
    if not mesh.is_watertight:
        raise RuntimeError("decimated wall is not closed")

    from scipy.spatial import cKDTree
    reference, _ = trimesh.sample.sample_surface(surface, 3_000_000, seed=0)
    decimated, _ = trimesh.sample.sample_surface(mesh, 3_000_000, seed=1)
    stray, _ = cKDTree(decimated).query(reference[::10], workers=-1)
    mesh.vertices = place(mesh.vertices)
    record = {
        "model_faces": int(len(model.faces)), "inner_faces": int(inner.sum()),
        "fragments_dropped": int(len(pieces) - 1), "open_ends": int(len(ends)),
        "faces": int(len(mesh.faces)), "volume_ml": round(abs(float(mesh.volume)) / 1000, 1),
        "model_inner_face_to_wall_mm": {"median": round(float(np.median(stray)), 3),
                                        "p99": round(float(np.percentile(stray, 99)), 3),
                                        "max": round(float(stray.max()), 3)},
    }
    return mesh, place(ends), end_radii, record, inner


# ------------------------------------------------------------------------ 2. centerlines --
def exact_radius(points, wall) -> np.ndarray:
    """Distance from each point to the wall (VTK's implicit distance to the triangles)."""
    import pyvista as pv

    poly = pv.PolyData(np.asarray(wall.vertices), np.column_stack([np.full(len(wall.faces), 3), wall.faces]).ravel())
    return np.abs(pv.PolyData(np.asarray(points, float)).compute_implicit_distance(poly)["implicit_distance"])


def lumen_grid(wall):
    """Grid points (spacing GRID_MM) inside the closed wall, their integer indices and R."""
    import trimesh
    from scipy.spatial import cKDTree

    rays = AxisRays(wall.triangles, axis=2, h=GRID_MM)
    cells, ks, odd = rays.inside_runs(GRID_MM)
    i, j = cells // rays.shape[1], cells % rays.shape[1]
    ijk = np.column_stack([i, j, ks]).astype(np.int64)
    ijk -= ijk.min(axis=0)
    base = np.array([rays.lo[0] + i.min() * GRID_MM, rays.lo[1] + j.min() * GRID_MM,
                     rays.lo_axis + ks.min() * GRID_MM])
    points = base + ijk * GRID_MM
    samples, _ = trimesh.sample.sample_surface(wall, int(wall.area * SAMPLES_PER_MM2), seed=0)
    radius, _ = cKDTree(samples).query(points, workers=-1)
    return points, ijk, radius, odd


def lumen_graph(ijk, radius):
    """Grid graph (26 neighbours), edge cost = length * mean(1 / R**COST_POWER) of its two ends."""
    import scipy.sparse as sp

    dims = ijk.max(axis=0) + 2
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(key)
    skey = key[order]
    weight = 1.0 / np.maximum(radius, 0.05) ** COST_POWER
    rows, cols, costs = [], [], []
    offsets = [(a, b, c) for a in (0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
    for a, b, c in offsets:
        target = ijk + (a, b, c)
        ok = np.all((target >= 0) & (target < dims), axis=1)
        tkey = (target[:, 0] * dims[1] + target[:, 1]) * dims[2] + target[:, 2]
        pos = np.clip(np.searchsorted(skey, tkey), 0, len(skey) - 1)
        hit = ok & (skey[pos] == tkey)
        src, dst = np.flatnonzero(hit).astype(np.int32), order[pos[hit]].astype(np.int32)
        rows.append(src)
        cols.append(dst)
        costs.append((GRID_MM * np.sqrt(a * a + b * b + c * c) * 0.5 * (weight[src] + weight[dst])).astype(np.float32))
    n = len(ijk)
    upper = sp.csr_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return (upper + upper.T).tocsr()


def path_to(predecessor, node: int) -> list[int]:
    path = [node]
    while predecessor[path[-1]] >= 0:
        path.append(int(predecessor[path[-1]]))
    return path[::-1]


def find_paths(points, ijk, radius, ends, end_radii):
    """Grid paths through the lumen, all from the entrance (the widest open end):
      1. to every other open end: one Dijkstra, so the paths form a tree;
      2. then, while some lumen is uncovered (more than COVER_MM outside the inscribed spheres of
         every path and away from the open ends), each uncovered pocket of at least
         MIN_POCKET_MM3 is looked at:
           - it touches the paths at two places that are far apart along them: a loop (e.g. the
             circle of Willis); the cheapest path across it joins the two places;
           - it touches them at one place and reaches far enough to be a vessel (at least
             max(3 mm, 2 R) beyond the paths, R the radius where it leaves them): a closed vessel;
             the path to its farthest point is added;
           - otherwise it is a bulge of the lumen and is left alone.
    The aortic root (the aorta's closed end) is added later, once the aorta is known
    (``extend_aorta``). Returns the Dijkstra predecessors, the tree paths, the loop paths (joined
    at both ends) and a record."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    graph = lumen_graph(ijk, radius)
    end_nodes = cKDTree(points).query(ends)[1]
    inlet = int(np.argmax(end_radii))
    source = int(end_nodes[inlet])
    cost, predecessor = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
    reach = [k for k in range(len(ends)) if k != inlet and np.isfinite(cost[end_nodes[k]])]
    lost = [k for k in range(len(ends)) if k != inlet and not np.isfinite(cost[end_nodes[k]])]
    paths = [path_to(predecessor, int(end_nodes[k])) for k in reach]
    loops, closed, bulges = [], [], 0
    open_ends = (np.asarray(ends), np.asarray(end_radii))
    grid_tree = cKDTree(points)
    for _ in range(ROUNDS):
        changed = False
        for pocket, touch in uncovered_pockets(graph, points, radius, paths + loops, open_ends):
            groups = touch_groups(points, touch)
            if len(groups) >= 2:
                bridge = cheapest_link(graph, grid_tree, points, radius, pocket, groups[0], groups[1], paths + loops)
                if bridge is not None:
                    loops.append(bridge)
                    changed = True
                    continue
            far = int(pocket[np.argmax(far_from(points, radius, pocket, paths + loops))])
            path = path_to(predecessor, far)
            on = set(np.concatenate(paths + loops).tolist())
            leave = next(i for i, n in enumerate(path) if n not in on)
            reach_mm = arclength(points[path[leave - 1:]])[-1]
            if reach_mm >= max(3.0, 2.0 * radius[path[leave - 1]]):
                paths.append(path)
                closed.append((far, reach_mm))
                changed = True
            else:
                bulges += 1
        if not changed:
            break
    record = {"entrance": points[source].round(2).tolist(), "open_ends_reached": len(reach),
              "open_ends_not_reached": [ends[k].round(2).tolist() for k in lost],
              "loops_joined": len(loops), "bulges_left": bulges,
              "closed_vessels": [{"end": points[n].round(2).tolist(), "beyond_paths_mm": round(r, 1)}
                                 for n, r in closed]}
    return graph, predecessor, paths, loops, record


def far_from(points, radius, nodes, paths) -> np.ndarray:
    """How far each node lies outside the inscribed spheres of the path nodes."""
    from scipy.spatial import cKDTree

    on = np.unique(np.concatenate(paths))
    gap, nearest = cKDTree(points[on]).query(points[nodes], workers=-1)
    return gap - radius[on][nearest]


def uncovered_pockets(graph, points, radius, paths, open_ends):
    """Connected pockets of lumen more than COVER_MM outside every path's inscribed spheres and
    farther than twice its radius + COVER_MM from every open end's centre (next to an open end the
    radius is measured to the cap, so the paths' spheres are small there); each with the path
    nodes it touches."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    on = np.unique(np.concatenate(paths))
    gap, nearest = cKDTree(points[on]).query(points, workers=-1)
    excess = gap - radius[on][nearest]
    centres, end_r = open_ends
    near_end, which = cKDTree(centres).query(points, workers=-1)
    loose = (excess > COVER_MM) & (near_end > 2 * end_r[which] + COVER_MM)
    nodes = np.flatnonzero(loose)
    if not len(nodes):
        return []
    _, label = connected_components(graph[nodes][:, nodes], directed=False)
    out = []
    for c in np.argsort(np.bincount(label))[::-1]:
        pocket = nodes[label == c]
        if len(pocket) * GRID_MM ** 3 < MIN_POCKET_MM3:
            break
        rim = pocket[excess[pocket] < COVER_MM + 2 * GRID_MM]
        out.append((pocket, np.unique(on[nearest[rim]])))
    return out


def touch_groups(points, touch, apart_mm: float = 8.0):
    """The touched path nodes in groups that lie more than ``apart_mm`` apart; largest first."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(touch) == 0:
        return []
    pairs = cKDTree(points[touch]).query_pairs(apart_mm, output_type="ndarray")
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(touch),) * 2)
    _, label = connected_components(g, directed=False)
    groups = [touch[label == k] for k in range(label.max() + 1)]
    return sorted(groups, key=len, reverse=True)


def cheapest_link(graph, grid_tree, points, radius, pocket, group_a, group_b, paths):
    """Cheapest path from a path node of group a to one of group b through the pocket and the
    inscribed spheres around the two groups; None if the two are close along the paths (a bulge
    beside a junction, not a loop): their distance along the paths must exceed LOOP_FACTOR times
    their straight distance plus LOOP_MARGIN_MM."""
    from scipy.sparse.csgraph import dijkstra

    around = np.r_[group_a, group_b]
    near = grid_tree.query_ball_point(points[around], radius[around] + 1.0, workers=-1)
    region = np.unique(np.r_[pocket, around, np.concatenate([np.asarray(n, int) for n in near])])
    position = np.searchsorted(region, np.r_[group_a, group_b])
    start, stop = position[:len(group_a)], position[len(group_a):]
    dist, pred, _ = dijkstra(graph[region][:, region], directed=False, indices=start, min_only=True,
                             return_predecessors=True)
    best = stop[int(np.argmin(dist[stop]))]
    if not np.isfinite(dist[best]):
        return None
    link = [int(region[i]) for i in path_to(pred, int(best))]
    a, b = link[0], link[-1]
    along = path_network_distance(points, paths, a, b)
    if along <= LOOP_FACTOR * np.linalg.norm(points[a] - points[b]) + LOOP_MARGIN_MM:
        return None
    return link


def path_network_distance(points, paths, a: int, b: int) -> float:
    """Length of the shortest way from path node a to path node b along the paths."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    edges = np.concatenate([np.column_stack([p[:-1], p[1:]]) for p in paths if len(p) > 1])
    nodes, inverse = np.unique(edges, return_inverse=True)
    e = inverse.reshape(-1, 2)
    length = np.linalg.norm(points[nodes[e[:, 0]]] - points[nodes[e[:, 1]]], axis=1)
    g = sp.coo_matrix((length, (e[:, 0], e[:, 1])), shape=(len(nodes),) * 2).tocsr()
    ia, ib = np.searchsorted(nodes, a), np.searchsorted(nodes, b)
    return float(dijkstra(g, directed=False, indices=ia)[ib])


def path_segments(paths, loops, source: int):
    """The paths and loops as segments between junctions (nodes where they meet or end). A segment
    made of path steps points the way the paths go (away from the entrance) and its parent is the
    segment arriving at its start; a segment containing a loop step is oriented from its end that
    the paths reach first and ``joins`` the paths at its other end. Parents come before children."""
    from collections import defaultdict

    step_of = {}                                   # node -> the node before it on the paths
    order = {source: 0}
    for p in paths:
        for i, (a, b) in enumerate(zip(p[:-1], p[1:])):
            step_of[b] = a
    for p in paths:                                # steps from the entrance, along the paths
        for i, n in enumerate(p):
            order[n] = min(order.get(n, i), i)
    neighbours = defaultdict(set)
    for p in paths + loops:
        for a, b in zip(p[:-1], p[1:]):
            if a != b:
                neighbours[a].add(b)
                neighbours[b].add(a)
    special = {n for n, nb in neighbours.items() if len(nb) != 2} | {source}
    chains, seen = [], set()
    for start in sorted(special):
        for nb in sorted(neighbours[start]):
            if (start, nb) in seen:
                continue
            chain = [start, nb]
            while chain[-1] not in special:
                chain.append(next(n for n in neighbours[chain[-1]] if n != chain[-2]))
            seen.add((chain[0], chain[1]))
            seen.add((chain[-1], chain[-2]))
            chains.append(chain)
    segments = []
    for chain in chains:
        forward = all(step_of.get(b) == a for a, b in zip(chain[:-1], chain[1:]))
        backward = all(step_of.get(a) == b for a, b in zip(chain[:-1], chain[1:]))
        if backward:
            chain = chain[::-1]
        joins = not (forward or backward)
        if joins and order.get(chain[-1], np.inf) < order.get(chain[0], np.inf):
            chain = chain[::-1]
        segments.append({"nodes": chain, "joins": joins})
    segments.sort(key=lambda s: (order.get(s["nodes"][0], np.inf), s["joins"], s["nodes"][1]))
    arriving = {s["nodes"][-1]: i for i, s in enumerate(segments) if not s["joins"]}
    for s in segments:
        s["parent"] = arriving.get(s["nodes"][0])
    return segments


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def heading(line, at_start: bool) -> np.ndarray:
    """Direction over the first (or last) DIRECTION_MM of a polyline."""
    arc = arclength(line)
    if at_start:
        k = min(int(np.searchsorted(arc, DIRECTION_MM)), len(line) - 1)
        return unit(line[k] - line[0])
    k = max(int(np.searchsorted(arc, arc[-1] - DIRECTION_MM)) - 1, 0)
    return unit(line[-1] - line[k])


def own_radius(r, line) -> float:
    """A segment's own size: median radius beyond its first 3 mm (all of it when shorter)."""
    arc = arclength(line)
    beyond = r[arc >= min(3.0, arc[-1] / 2)]
    return float(np.median(beyond if len(beyond) else r))


def build_vessels(segments, lines, radii, follow=frozenset()):
    """Join segments into vessels. At a junction the vessel carries on into a branch that turns by
    less than 90 degrees and keeps at least half its size (median radius over SIZE_MM once it is
    out of the parent's inscribed sphere);
    of those the largest if it is at least 1.25 times the next, otherwise the straightest that
    goes on for at least STUB_MM (short closed side branches never carry a vessel on). The others
    start new vessels."""
    from collections import defaultdict, deque

    kids = defaultdict(list)
    for i, seg in enumerate(segments):
        kids[seg["parent"]].append(i)
    length = [float(arclength(l)[-1]) for l in lines]
    reach = [0.0] * len(segments)
    for i in reversed(range(len(segments))):            # children come after their parent
        reach[i] = length[i] + max((reach[k] for k in kids[i]), default=0.0)

    def size(i):
        """Median radius over SIZE_MM, starting where the branch leaves the parent's inscribed
        sphere at the junction (before that its radius is the parent's)."""
        skip = max(3.0, float(radii[i][0]))
        line, r = lines[i], radii[i]
        while arclength(line)[-1] < skip + SIZE_MM and kids[i]:
            i = max(kids[i], key=lambda k: reach[k])
            line, r = np.vstack([line, lines[i][1:]]), np.r_[r, radii[i][1:]]
        arc = arclength(line)
        part = r[(arc >= min(skip, arc[-1] / 2)) & (arc <= skip + SIZE_MM)]
        return float(np.median(part if len(part) else r[-1:]))

    sizes = [size(i) for i in range(len(segments))]
    vessels, todo = [], deque((i, None) for i in kids[None])
    while todo:
        first, up = todo.popleft()
        chain, line, r = [first], lines[first], radii[first]
        while kids[chain[-1]]:
            incoming = heading(line, at_start=False)
            arc = arclength(line)
            inside = float(np.median(r[arc >= arc[-1] - SIZE_MM]))
            options = kids[chain[-1]]
            turn = {o: np.degrees(np.arccos(np.clip(incoming @ heading(lines[o], True), -1, 1))) for o in options}
            fit = sorted((o for o in options if turn[o] < 90 and sizes[o] >= 0.5 * inside), key=lambda o: -sizes[o])
            carry = next((o for o in options if segments[o]["nodes"][-1] in follow), None)
            if carry is not None:
                pass
            elif len(fit) == 1 or (len(fit) > 1 and sizes[fit[0]] >= 1.25 * sizes[fit[1]]):
                carry = fit[0]
            elif fit:
                lasting = [o for o in fit if reach[o] >= STUB_MM] or fit
                carry = min(lasting, key=lambda o: turn[o])
            for o in options:
                if o != carry:
                    todo.append((o, len(vessels)))
            if carry is None:
                break
            chain.append(carry)
            line = np.vstack([line, lines[carry][1:]])
            r = np.r_[r, radii[carry][1:]]
        vessels.append({"segments": chain, "parent": up, "line": line, "radius": r,
                        "joins": segments[chain[-1]]["joins"]})
    return vessels


def name_vessels(vessels):
    """Indices of the six training vessels and where the brachiocephalic trunk divides:
    {aorta, lsa, lcca, bct, rcca, rsa: (vessel, start arc length)}, bct split arc length."""
    aorta = 0
    line = vessels[aorta]["line"]
    arc = arclength(line)
    top = line[:, 2].max()
    branches = []
    for k, v in enumerate(vessels):
        if v["parent"] != aorta:
            continue
        s = float(arc[np.argmin(np.linalg.norm(line - v["line"][0], axis=1))])
        size = own_radius(v["radius"], v["line"])
        if v["line"][0, 2] > top - ARCH_DEPTH_MM and size >= 2.0:
            branches.append((size, s, k))
    if len(branches) < 3:
        raise RuntimeError(f"found {len(branches)} large branches at the top of the arch, not 3")
    lsa, lcca, bct = (k for _, s, k in sorted(sorted(branches, reverse=True)[:3], key=lambda b: b[1]))
    trunk = vessels[bct]
    t_arc = arclength(trunk["line"])
    split, child = None, None
    for k, v in enumerate(vessels):
        if v["parent"] == bct and own_radius(v["radius"], v["line"]) >= 1.5:
            s = float(t_arc[np.argmin(np.linalg.norm(trunk["line"] - v["line"][0], axis=1))])
            if split is None or s < split:
                split, child = s, k
    if split is None:
        raise RuntimeError("the brachiocephalic trunk does not divide")

    def rise(points, s0):
        a = arclength(points)
        return float(np.interp(s0 + RISE_MM, a, points[:, 2]) - np.interp(s0, a, points[:, 2]))

    on_trunk = (bct, split)                       # the trunk carrying on past the split
    leaving = (child, 0.0)
    rcca, rsa = ((on_trunk, leaving) if rise(trunk["line"], split) > rise(vessels[child]["line"], 0.0)
                 else (leaving, on_trunk))
    return {"aorta": (aorta, 0.0), "lsa": (lsa, 0.0), "lcca": (lcca, 0.0), "bct": (bct, 0.0),
            "rcca": rcca, "rsa": rsa}, split


def smooth_line(path, sigma_mm: float = SMOOTH_SIGMA_MM, spacing: float = 0.15) -> np.ndarray:
    """Gaussian smoothing along the arc length; the correction fades out towards both ends."""
    from scipy.ndimage import gaussian_filter1d

    arc = arclength(path)
    if arc[-1] < 1e-9:
        return np.asarray(path, float)[:1]
    u = np.arange(0.0, arc[-1], spacing)
    u = np.r_[u, arc[-1]] if arc[-1] - u[-1] > 1e-9 else u
    dense = np.stack([np.interp(u, arc, path[:, k]) for k in range(3)], axis=1)
    blurred = gaussian_filter1d(dense, sigma_mm / spacing, axis=0, mode="nearest")
    weight = np.clip(np.minimum(u, u[-1] - u) / (3 * sigma_mm), 0, 1)[:, None]
    return dense + weight * (blurred - dense)


def resample(line, s0: float, s1: float, step: float) -> np.ndarray:
    """Points from arc length s0 to s1 about ``step`` apart (both ends included)."""
    arc = arclength(line)
    s1 = min(s1, arc[-1])
    n = max(int(round((s1 - s0) / step)), 1)
    s = np.linspace(s0, s1, n + 1)
    return np.stack([np.interp(s, arc, line[:, k]) for k in range(3)], axis=1)


def extend_aorta(graph, points, radius, predecessor, paths, loops, ends, end_radii, end_point):
    """Path to the bottom of the aortic root: the lowest point with R >= END_RADIUS_MM of the
    uncovered pocket next to the aorta's end (as ``biro_phantom_root.py``, on this wall)."""
    from scipy.spatial import cKDTree

    pockets = uncovered_pockets(graph, points, radius, paths + loops, (np.asarray(ends), np.asarray(end_radii)))
    best, gap = None, np.inf
    for pocket, touch in pockets:
        if len(touch):
            d = float(cKDTree(points[touch]).query(end_point)[0])
            if d < gap:
                best, gap = pocket, d
    if best is None or gap > ROOT_REACH_MM:
        return None
    deep = best[radius[best] >= END_RADIUS_MM]
    return path_to(predecessor, int(deep[np.argmin(points[deep, 2])])) if len(deep) else None


def open_end_radii(coords, radii, ends) -> np.ndarray:
    """Radii with the caps left out: at an open end the distance to the wall is to the flat cap,
    which is not part of the vessel. From the end back to the first point farther from the end
    than its own radius, the radius is that point's. Then at least MIN_RADIUS_MM everywhere."""
    radii = np.asarray(radii, float).copy()
    for forward in (True, False):
        c, r = (coords, radii) if forward else (coords[::-1], radii[::-1])
        if np.linalg.norm(np.asarray(ends) - c[-1], axis=1).min() > 1.0:
            continue
        back = arclength(c[::-1])                               # distance from the end
        k = next((i for i in range(len(c)) if back[i] > r[::-1][i] + 0.05), len(c) - 1)
        tail = len(c) - 1 - k
        r[tail:] = r[tail]
    return np.maximum(radii, MIN_RADIUS_MM)


def task_pieces(vessels, names, split):
    """Cut the vessels into named pieces: (name, vessel, s0, s1, step), entrance to the distal end."""
    lengths, step_a, step_b = TASK["branch_lengths_mm"], TASK["step_aorta_mm"], TASK["step_branch_mm"]
    ends = {k: float(arclength(v["line"])[-1]) for k, v in enumerate(vessels)}
    aorta = names["aorta"][0]
    a_line = vessels[aorta]["line"]
    ostium = float(arclength(a_line)[np.argmin(np.linalg.norm(a_line - vessels[names["lsa"][0]]["line"][0], axis=1))])
    start = ostium - TASK["first_ostium_from_start_mm"]
    if start <= 0:
        raise RuntimeError("the aorta is shorter than the task needs before its first branch")
    pieces = [("aorta_inlet", aorta, 0.0, start, step_b), ("aorta", aorta, start, ends[aorta], step_a)]
    bct = names["bct"][0]
    pieces.append(("bct", bct, 0.0, split, step_b))
    for name in ("rcca", "rsa", "lcca", "lsa"):
        k, s0 = names[name]
        s1 = min(s0 + lengths[name], ends[k])
        pieces.append((name, k, s0, s1, step_b))
        if ends[k] - s1 > step_b:
            pieces.append((f"{name}_rest", k, s1, ends[k], step_b))
    named = {k for _, k, _, _, _ in pieces}
    n = 0
    for k in range(len(vessels)):
        if k not in named:
            n += 1
            pieces.append((f"v{n}", k, 0.0, ends[k], step_b))
    return pieces


def build_centerlines(wall, ends, end_radii):
    """Steps 2 and 3. Returns the branches, task branch names, junctions, insertion and a record."""
    points, ijk, radius, odd = lumen_grid(wall)
    graph, predecessor, paths, loops, record = find_paths(points, ijk, radius, ends, end_radii)
    source = paths[0][0]

    def assemble(paths, follow=frozenset()):
        segments = path_segments(paths, loops, source)
        lines = [smooth_line(points[s["nodes"]]) for s in segments]
        node_r = [np.interp(arclength(l), arclength(points[s["nodes"]]), radius[s["nodes"]]) if len(l) > 1
                  else radius[s["nodes"][:1]] for s, l in zip(segments, lines)]
        vessels = build_vessels(segments, lines, node_r, follow)
        return segments, vessels

    segments, vessels = assemble(paths)
    root = extend_aorta(graph, points, radius, predecessor, paths, loops, ends, end_radii, vessels[0]["line"][-1])
    if root is not None:
        paths = paths + [root]
        segments, vessels = assemble(paths, frozenset(root))
        record["aortic_root_end"] = points[root[-1]].round(2).tolist()
    names, split = name_vessels(vessels)
    pieces = task_pieces(vessels, names, split)

    branches, owner = {}, {}
    for name, k, s0, s1, step in pieces:
        coords = resample(vessels[k]["line"], s0, s1, step)
        radii = open_end_radii(coords, exact_radius(coords, wall), ends)
        branches[name] = {"coordinates": coords.round(3), "radii": radii.round(3)}
        owner[name] = (k, s0, s1)
    junctions = []

    def pieces_at(k, s, tol=1e-6):
        return [n for n, (kk, s0, s1) in owner.items() if kk == k and s0 - tol <= s <= s1 + tol]

    for k, v in enumerate(vessels):
        first = next(n for n, (kk, s0, _) in owner.items() if kk == k and s0 == 0.0)
        if v["parent"] is not None:
            p_line = vessels[v["parent"]]["line"]
            s = float(arclength(p_line)[np.argmin(np.linalg.norm(p_line - v["line"][0], axis=1))])
            junctions.append((v["line"][0], pieces_at(v["parent"], s) + [first]))
        if v["joins"]:
            end = v["line"][-1]
            last = max((n for n, (kk, _, _) in owner.items() if kk == k), key=lambda n: owner[n][2])
            other = min((n for n in branches if owner[n][0] != k),
                        key=lambda n: np.linalg.norm(branches[n]["coordinates"] - end, axis=1).min())
            junctions.append((end, [last, other]))
        cuts = sorted((s0, n) for n, (kk, s0, _) in owner.items() if kk == k and s0 > 0)
        for s0, n in cuts:
            junctions.append((resample(v["line"], s0, s0, 1.0)[0], pieces_at(k, s0)))
    merged = []
    for point, names_here in junctions:
        for m in merged:
            if np.linalg.norm(m["point"] - point) < 0.05:
                m["branches"] = sorted(set(m["branches"]) | set(names_here))
                break
        else:
            merged.append({"point": np.asarray(point, float), "branches": sorted(set(names_here))})
    aorta = branches["aorta"]["coordinates"]
    insertion = {"position": aorta[1].round(3).tolist(), "direction": unit(aorta[2] - aorta[1]).round(6).tolist()}
    record.update({"grid_points": int(len(points)), "grid_columns_unpaired": odd,
                   "segments": len(segments), "vessels": len(vessels), "pieces": len(pieces),
                   "brachiocephalic_split_mm": round(split, 1)})
    return branches, list(names), merged, insertion, record


# ------------------------------------------------------------------------------- output --
TASK_ORDER = ("aorta", "bct", "rsa", "rcca", "lcca", "lsa")


def spec_json(branches, junctions, insertion, record, mesh_path) -> dict:
    order = list(TASK_ORDER) + [n for n in branches if n not in TASK_ORDER]
    return {
        "what": "the rebuilt phantom: wall, centerlines, task branches and junctions, all from the phantom model",
        "frame": "training frame: z towards the head, the aorta's start near the origin; mm",
        "source": {"model": f"{SOURCE[0].name} (sha256 {SOURCE[1]})", "script": "guidewire_rl/env/vessels/biro_rebuild.py",
                   "wall": f"{Path(mesh_path).name} (sha256 {sha256(mesh_path)})"},
        "frame_transform": FRAME, "task_definition": TASK,
        "insertion": insertion,
        "task_branches": list(TASK_ORDER),
        "branches": {n: {"coordinates": np.asarray(branches[n]["coordinates"]).tolist(),
                         "radii": np.asarray(branches[n]["radii"]).tolist()} for n in order},
        "junctions": [{"point": np.asarray(j["point"]).round(3).tolist(), "branches": j["branches"]} for j in junctions],
        "lengths_mm": {n: round(float(arclength(np.asarray(branches[n]["coordinates"]))[-1]), 1) for n in order},
        "radius_median_mm": {n: round(float(np.median(branches[n]["radii"])), 2) for n in TASK_ORDER},
        "build": record,
        "settings": {"probe_mm": list(PROBE_MM), "ray_grid_mm": RAY_GRID_MM, "faces": FACES, "grid_mm": GRID_MM,
                     "samples_per_mm2": SAMPLES_PER_MM2, "cost_power": COST_POWER, "end_radius_mm": END_RADIUS_MM,
                     "min_radius_mm": MIN_RADIUS_MM,
                     "size_mm": SIZE_MM, "stub_mm": STUB_MM, "cover_mm": COVER_MM, "min_pocket_mm3": MIN_POCKET_MM3, "loop_factor": LOOP_FACTOR,
                     "loop_margin_mm": LOOP_MARGIN_MM, "smooth_sigma_mm": SMOOTH_SIGMA_MM,
                     "direction_mm": DIRECTION_MM, "arch_depth_mm": ARCH_DEPTH_MM, "rise_mm": RISE_MM,
                     "root_reach_mm": ROOT_REACH_MM},
    }


def write_obj(mesh, path, title: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"# {title}\n")
        for v in mesh.vertices:
            fh.write("v %.5f %.5f %.5f\n" % tuple(v))
        for f in np.asarray(mesh.faces) + 1:
            fh.write("f %d %d %d\n" % tuple(f))


def main() -> None:
    from . import biro_cut

    wall, ends, end_radii, wall_record, _ = build_wall()
    write_obj(wall, OUT_MESH, "the rebuilt phantom's wall (guidewire_rl/env/vessels/biro_rebuild.py)")
    branches, _, junctions, insertion, record = build_centerlines(wall, ends, end_radii)
    record = {"wall": wall_record, "centerlines": record}
    spec = spec_json(branches, junctions, insertion, record, OUT_MESH)
    OUT_SPEC.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"wrote {OUT_MESH.name} ({wall_record['faces']} faces, {wall_record['volume_ml']} ml) and {OUT_SPEC.name} "
          f"({len(spec['branches'])} branches, {len(spec['junctions'])} junctions)")
    biro_cut.write_cut("rebuilt", OUT_MESH, OUT_SPEC)


if __name__ == "__main__":
    main()
