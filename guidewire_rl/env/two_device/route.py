"""The route and the ruler of the two-device task (todo_list/two_devices_training.md, sections 1-2).

Pure geometry on the vessel tree's centerlines and listed junctions; no stEVE pathfinder. Every
length is in mm, every point in the vessel frame (stEVE's "vessel coordinate system"; the
fluoroscopy's tracking frame is that frame rotated by ``image_rot_zx``).

The route runs from the insertion point up the descending aorta, over the arch to O (where the
brachiocephalic trunk leaves the aorta), up the trunk and into the right common carotid:
``aorta`` (to O) -> ``bct`` -> ``rcca`` -> ``rcca_rest``.

How the tree is measured
------------------------
* The tree is a graph of centerline pieces joined at the vessel tree's junctions
  (``vessel_tree.branching_points``; for the rebuilt phantom the junctions its spec lists).
  Distances are along it (``path``).
* A vessel that leaves a wide parent (the aorta, 26 mm across) starts on the parent's centerline,
  so its first 19-21 mm run inside the parent's lumen. Its *mouth* is where its centerline leaves
  the parent's tube (the lab's ``arch_catheter.Mouth``; the trunk's is 19.7 mm in, at z 95.1).
  That inner piece counts as zero length and points there count as being in the parent, so a tip
  crossing the mouth moves continuously on the ruler. Vessels off a narrow parent open at the
  junction itself (the carotid opens at the trunk's end: the plan's "右颈总开口 = 头臂干末端").
* ``locate(p)``: the centerline piece whose tube holds ``p`` deepest (largest radius minus
  distance), and how far along it. Negative clearance means outside every tube; the tubes follow
  the model's inner wall to about 0.2 mm (median) and at most 3.7 mm near the route (measured
  2026-10-31 on the wall's vertices), so ``outside`` uses a margin.
* ``depth(p, branch)``: signed distance from ``branch``'s mouth, positive inside ``branch`` and
  everything beyond it, negative elsewhere (the distance back to the mouth).
* ``s(p)``: arc length along the route; for a point off the route, the route position where its
  side vessel leaves the route plus how far into the side vessel (so the ascending aorta past O
  continues the aorta's arc length).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

ROUTE_CHAIN = ("aorta", "bct", "rcca", "rcca_rest")
ROOT_BRANCH = "aorta"
WIDE_PARENTS = ("aorta",)
JUNCTION_TOLERANCE_MM = 3.0     # a listed junction farther than this from a branch it names is an error
DIRECTION_MM = 10.0             # the junction map's direction: first 10 mm of the branch to enter
TASK_BOUNDS_BRANCHES = ("aorta", "bct", "rsa", "rcca", "lcca", "lsa")
TASK_BOUNDS_RCCA_REST_MM = 100.0   # the carotid chain counts for the 2D box up to here (first fork: 64)
TASK_BOUNDS_MARGIN_MM = 5.0


def arclen(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros(len(points))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def point_on(points: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    """The point at arc length ``s`` along a polyline (clamped to its ends)."""
    return np.array([np.interp(s, cum, points[:, k]) for k in range(points.shape[1])])


def project(points: np.ndarray, cum: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Closest point of a polyline to ``p``: (arc length there, distance)."""
    a, b = points[:-1], points[1:]
    v = b - a
    l2 = np.maximum(np.einsum("ij,ij->i", v, v), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", p - a, v) / l2, 0.0, 1.0)
    dist = np.linalg.norm(a + t[:, None] * v - p, axis=1)
    i = int(np.argmin(dist))
    return float(cum[i] + t[i] * (cum[i + 1] - cum[i])), float(dist[i])


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


@dataclass(frozen=True)
class Location:
    branch: str          # centerline piece whose tube holds the point deepest
    arc: float           # mm along that branch from its first point
    lateral: float       # mm from its centerline
    clearance: float     # tube radius minus lateral, mm; negative: outside every tube


class VesselGraph:
    """Every centerline of the tree as one graph (module docstring)."""

    def __init__(self, branches, junctions: Sequence[Tuple[np.ndarray, Sequence[str]]],
                 root: str = ROOT_BRANCH, wide_parents: Sequence[str] = WIDE_PARENTS) -> None:
        self.names: List[str] = [b.name for b in branches]
        self.index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}
        if root not in self.index:
            raise KeyError(f"root branch {root!r} not in the vessel tree")
        self.coords = [np.asarray(b.coordinates, dtype=np.float64) for b in branches]
        self.radii = [np.asarray(b.radii, dtype=np.float64) for b in branches]
        self.cum = [arclen(c) for c in self.coords]
        self.length = [float(c[-1]) for c in self.cum]

        # junction positions on the branches they join
        node_at: List[List[Tuple[float, int]]] = [[] for _ in self.names]
        junction_branches: List[List[int]] = []
        for k, (point, names) in enumerate(junctions):
            point = np.asarray(point, dtype=np.float64)
            members = []
            for name in names:
                if name not in self.index:
                    continue
                b = self.index[name]
                arc, dist = project(self.coords[b], self.cum[b], point)
                if dist > JUNCTION_TOLERANCE_MM:
                    raise ValueError(f"junction {k} at {point} is {dist:.1f} mm from {name}")
                node_at[b].append((arc, k))
                members.append(b)
            junction_branches.append(members)
        n_nodes = len(junctions)
        for b in range(len(self.names)):          # open vessel ends are nodes too
            for end in (0.0, self.length[b]):
                if not any(abs(arc - end) < 1e-6 for arc, _ in node_at[b]):
                    node_at[b].append((end, n_nodes))
                    n_nodes += 1
        self.n_nodes = n_nodes

        # parent of each branch: breadth first from the root through the junctions
        self.parent: Dict[int, Optional[int]] = {self.index[root]: None}
        self.attach_arc: Dict[int, float] = {}     # where on its parent a branch is joined
        queue = [self.index[root]]
        while queue:
            b = queue.pop(0)
            for arc_b, k in node_at[b]:
                if k >= len(junctions):
                    continue
                for other in junction_branches[k]:
                    if other not in self.parent:
                        self.parent[other] = b
                        self.attach_arc[other] = arc_b
                        queue.append(other)
        missing = [self.names[b] for b in range(len(self.names)) if b not in self.parent]
        if missing:
            raise ValueError(f"branches not joined to {root!r} by any junction: {missing}")

        # mouths: a child of a wide parent, joined inside the parent (not at its ends), opens
        # where its centerline leaves the parent's tube; every other branch at its start
        self.mouth_arc = np.zeros(len(self.names))
        wide = {self.index[n] for n in wide_parents if n in self.index}
        for b, parent in self.parent.items():
            if parent is None or parent not in wide:
                continue
            a = self.attach_arc[b]
            if a < 1e-6 or a > self.length[parent] - 1e-6:
                continue
            if np.linalg.norm(self.coords[b][0] - point_on(self.coords[parent], self.cum[parent], a)) > \
                    JUNCTION_TOLERANCE_MM:
                raise ValueError(f"{self.names[b]} does not start at its junction with "
                                 f"{self.names[parent]}")
            for i, q in enumerate(self.coords[b]):
                s, dist = project(self.coords[parent], self.cum[parent], q)
                if dist >= float(np.interp(s, self.cum[parent], self.radii[parent])):
                    self.mouth_arc[b] = self.cum[b][i]
                    break

        # graph: consecutive nodes along each branch, lengths in effective arc; two pieces
        # between the same two nodes keep the shorter one
        self.nodes: List[Tuple[np.ndarray, np.ndarray]] = []
        edges: Dict[Tuple[int, int], float] = {}
        for b in range(len(self.names)):
            pos = sorted(node_at[b])
            arcs = np.array([p[0] for p in pos])
            ids = np.array([p[1] for p in pos], dtype=int)
            self.nodes.append((arcs, ids))
            for (a0, k0), (a1, k1) in zip(pos[:-1], pos[1:]):
                if k0 == k1:
                    continue
                key = (min(k0, k1), max(k0, k1))
                w = self.effective(b, a1) - self.effective(b, a0)
                edges[key] = min(w, edges.get(key, np.inf))
        rows = [k for k, _ in edges] + [k for _, k in edges]
        cols = [k for _, k in edges] + [k for k, _ in edges]
        # a zero-length edge still connects: csgraph would drop an explicit zero
        weights = np.maximum(np.asarray(list(edges.values()) * 2, dtype=np.float64), 1e-9)
        graph = coo_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        self.D = shortest_path(graph, method="D", directed=False)

        # tubes for locate(): every segment except the inner piece of a child of a wide parent
        starts, vectors, r0, dr, seg_branch, seg_arc0, seg_len = [], [], [], [], [], [], []
        for b in range(len(self.names)):
            c, r, cum = self.coords[b], self.radii[b], self.cum[b]
            keep = cum[1:] > self.mouth_arc[b] + 1e-9
            if self.mouth_arc[b] <= 0:
                keep[:] = True
            idx = np.nonzero(keep)[0]
            starts.append(c[idx])
            vectors.append(c[idx + 1] - c[idx])
            r0.append(r[idx])
            dr.append(r[idx + 1] - r[idx])
            seg_branch.append(np.full(len(idx), b))
            seg_arc0.append(cum[idx])
            seg_len.append(cum[idx + 1] - cum[idx])
        self._s = np.concatenate(starts)
        self._v = np.concatenate(vectors)
        self._l2 = np.maximum(np.einsum("ij,ij->i", self._v, self._v), 1e-12)
        self._r0 = np.concatenate(r0)
        self._dr = np.concatenate(dr)
        self._seg_branch = np.concatenate(seg_branch)
        self._seg_arc0 = np.concatenate(seg_arc0)
        self._seg_len = np.concatenate(seg_len)

    # ---------------------------------------------------------------------------------
    def effective(self, b: int, arc: float) -> float:
        """Arc length with the inner piece (before the mouth) counted as zero."""
        return max(0.0, float(arc) - float(self.mouth_arc[b]))

    def locate(self, p: np.ndarray) -> Location:
        p = np.asarray(p, dtype=np.float64).reshape(3)
        t = np.clip(np.einsum("ij,ij->i", p - self._s, self._v) / self._l2, 0.0, 1.0)
        lateral = np.linalg.norm(self._s + t[:, None] * self._v - p, axis=1)
        clearance = self._r0 + t * self._dr - lateral
        i = int(np.argmax(clearance))
        b = int(self._seg_branch[i])
        return Location(self.names[b], float(self._seg_arc0[i] + t[i] * self._seg_len[i]),
                        float(lateral[i]), float(clearance[i]))

    def point(self, branch: str, arc: float) -> np.ndarray:
        b = self.index[branch]
        return point_on(self.coords[b], self.cum[b], arc)

    def radius(self, branch: str, arc: float) -> float:
        b = self.index[branch]
        return float(np.interp(arc, self.cum[b], self.radii[b]))

    def _ends(self, loc: Location) -> List[Tuple[int, float]]:
        """The graph nodes on either side of a location and the distance to each."""
        b = self.index[loc.branch]
        arcs, ids = self.nodes[b]
        e = self.effective(b, loc.arc)
        hi = int(np.searchsorted(arcs, loc.arc, side="left"))
        hi = min(max(hi, 0), len(arcs) - 1)
        lo = max(hi - 1, 0)
        if arcs[hi] < loc.arc:            # past the last node (clamped arc): only the last one
            lo = hi
        out = []
        for j in {lo, hi}:
            out.append((int(ids[j]), abs(e - self.effective(b, arcs[j]))))
        return out

    def path(self, a: Location, b: Location) -> float:
        """Distance along the tree between two located points (effective arc)."""
        best = np.inf
        if a.branch == b.branch:
            i = self.index[a.branch]
            best = abs(self.effective(i, a.arc) - self.effective(i, b.arc))
        for na, da in self._ends(a):
            row = self.D[na]
            for nb, db in self._ends(b):
                best = min(best, da + row[nb] + db)
        return float(best)

    def ancestors(self, branch: str) -> List[str]:
        """The branch itself, its parent, ... up to the root."""
        out, b = [], self.index[branch]
        while b is not None:
            out.append(self.names[b])
            b = self.parent[b]
        return out

    def beyond_mouth(self, loc: Location, branch: str) -> bool:
        """Whether a location is inside ``branch`` past its mouth or in anything beyond it."""
        if loc.branch == branch:
            return self.effective(self.index[branch], loc.arc) > 0.0
        return branch in self.ancestors(loc.branch)[1:]

    def mouth_location(self, branch: str) -> Location:
        b = self.index[branch]
        arc = float(self.mouth_arc[b])
        return Location(branch, arc, 0.0, self.radius(branch, arc))

    def depth(self, loc: Location, branch: str) -> float:
        distance = self.path(self.mouth_location(branch), loc)
        return distance if self.beyond_mouth(loc, branch) else -distance


class Route:
    """The task's route on the vessel tree (module docstring). Build with ``Route.of(vessel_tree)``
    to share one per tree."""

    _cache: Dict[Tuple[int, int, Tuple[str, ...]], "Route"] = {}

    def __init__(self, vessel_tree, chain: Sequence[str] = ROUTE_CHAIN) -> None:
        if vessel_tree.branches is None or not vessel_tree.branching_points:
            raise ValueError("Route needs a vessel tree with branches and junctions")
        junctions = [(np.asarray(bp.coordinates, dtype=np.float64), [c.name for c in bp.connections])
                     for bp in vessel_tree.branching_points]
        self.graph = VesselGraph(vessel_tree.branches, junctions, root=chain[0])
        self.chain = tuple(chain)
        g = self.graph
        for parent, child in zip(self.chain[:-1], self.chain[1:]):
            if g.names[g.parent[g.index[child]]] != parent:
                raise ValueError(f"route chain {self.chain}: {child} is not joined to {parent}")
        # route arc length: the root branch up to where the next one is joined (O), then the rest
        self.s_O = float(g.attach_arc[g.index[self.chain[1]]])
        self.offsets: Dict[str, float] = {self.chain[0]: 0.0}
        s = self.s_O
        for name in self.chain[1:]:
            self.offsets[name] = s
            b = g.index[name]
            s += g.effective(b, g.length[b])
        self.length = s
        self.O = g.point(self.chain[0], self.s_O)
        self.s_rcca = self.offsets[self.chain[2]] if len(self.chain) > 2 else self.length
        self.insertion = np.asarray(vessel_tree.insertion.position, dtype=np.float64)
        self.s_insertion = self.s(self.insertion)
        # the route's decision points: where it turns into a branch off its parent
        self.junctions: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for name in self.chain[1:]:
            b = g.index[name]
            parent = g.parent[b]
            if parent is not None and g.attach_arc[b] >= g.length[parent] - 1e-6 and \
                    g.names[parent] in self.chain and self._continues(parent, b):
                continue                  # a continuation of the same vessel, not a turn
            mouth = g.point(name, float(g.mouth_arc[b]))
            ahead = self.point_at(self.offsets[name] + DIRECTION_MM)
            self.junctions.append((name, mouth, unit(ahead - mouth)))
        self._target: Optional[Location] = None
        self._target_point: Optional[np.ndarray] = None
        self._F: Optional[Location] = None

    def _continues(self, parent: int, child: int) -> bool:
        """``child`` carries on ``parent`` (joined at its end, the only branch there)."""
        g = self.graph
        siblings = [b for b, p in g.parent.items() if p == parent
                    and abs(g.attach_arc[b] - g.attach_arc[child]) < 1e-6]
        return len(siblings) == 1

    @classmethod
    def of(cls, vessel_tree, chain: Sequence[str] = ROUTE_CHAIN) -> "Route":
        key = (id(vessel_tree), id(vessel_tree.branches), tuple(chain))
        route = cls._cache.get(key)
        if route is None:
            route = cls._cache[key] = cls(vessel_tree, chain)
        return route

    # ------------------------------------------------------------------ positions ----
    def locate(self, p: np.ndarray) -> Location:
        return self.graph.locate(p)

    def _route_arc(self, s: float) -> Tuple[str, float]:
        """(branch, arc along it) of route arc length ``s`` (clamped to the route)."""
        g = self.graph
        s = float(np.clip(s, 0.0, self.length))
        name = self.chain[0]
        for candidate in self.chain[1:]:
            if s >= self.offsets[candidate]:
                name = candidate
        if name == self.chain[0]:
            return name, s
        return name, s - self.offsets[name] + float(g.mouth_arc[g.index[name]])

    def point_at(self, s: float) -> np.ndarray:
        """The route centerline point at route arc length ``s`` (clamped to the route)."""
        return self.graph.point(*self._route_arc(s))

    def aorta_location(self, arc: float) -> Location:
        """A point on the root branch's (the aorta's) centerline, ``arc`` mm from its start; past O
        this is the ascending aorta, off the route."""
        name = self.chain[0]
        b = self.graph.index[name]
        arc = float(np.clip(arc, 0.0, self.graph.length[b]))
        return Location(name, arc, 0.0, self.graph.radius(name, arc))

    def location_at(self, s: float) -> Location:
        name, arc = self._route_arc(s)
        return Location(name, arc, 0.0, self.graph.radius(name, arc))

    def s(self, p, loc: Optional[Location] = None) -> float:
        """Arc length along the route (module docstring); off the route, the route position of
        the side vessel's junction plus the distance into it."""
        loc = loc or self.locate(p)
        g = self.graph
        if loc.branch in self.offsets:
            b = g.index[loc.branch]
            if loc.branch == self.chain[0]:
                return float(loc.arc)
            return self.offsets[loc.branch] + g.effective(b, loc.arc)
        # walk up to the first branch on the route
        child = g.index[loc.branch]
        while g.parent[child] is not None and g.names[g.parent[child]] not in self.offsets:
            child = g.parent[child]
        on_route = g.names[g.parent[child]]
        joined = Location(on_route, g.attach_arc[child], 0.0, 0.0)
        return self.s(None, joined) + g.path(joined, loc)

    def branch_of(self, p, loc: Optional[Location] = None) -> str:
        return (loc or self.locate(p)).branch

    def depth(self, p, branch: str, loc: Optional[Location] = None) -> float:
        """Signed mm beyond ``branch``'s mouth (module docstring)."""
        return self.graph.depth(loc or self.locate(p), branch)

    def aorta_position(self, p, loc: Optional[Location] = None) -> float:
        """mm along the aorta from O (positive towards the root, past O); a point off the aorta
        counts where its vessel leaves the aorta."""
        loc = loc or self.locate(p)
        g = self.graph
        root = self.chain[0]
        if loc.branch == root:
            return float(loc.arc) - self.s_O
        child = g.index[loc.branch]
        while g.parent[child] is not None and g.names[g.parent[child]] != root:
            child = g.parent[child]
        return float(g.attach_arc[child]) - self.s_O

    def path(self, a, b) -> float:
        la = a if isinstance(a, Location) else self.locate(a)
        lb = b if isinstance(b, Location) else self.locate(b)
        return self.graph.path(la, lb)

    # --------------------------------------------------------------- target and F ----
    def set_target(self, point: np.ndarray, catheter_goal_depth_mm: float = 15.0) -> None:
        """The episode's target (vessel frame) and F, the catheter's goal: ``catheter_goal_depth``
        into the right carotid (the success condition)."""
        self._target_point = np.asarray(point, dtype=np.float64).reshape(3)
        self._target = self.locate(self._target_point)
        self._F_depth = float(catheter_goal_depth_mm)
        self._F = self.location_at(self.s_rcca + self._F_depth)

    @property
    def target(self) -> np.ndarray:
        if self._target_point is None:
            raise RuntimeError("Route.set_target was not called")
        return self._target_point

    @property
    def F(self) -> np.ndarray:
        return self.point_at(self.s_rcca + self._F_depth)

    def path_to_target(self, p, loc: Optional[Location] = None) -> float:
        if self._target is None:
            raise RuntimeError("Route.set_target was not called")
        return self.graph.path(loc or self.locate(p), self._target)

    def path_to_F(self, p, loc: Optional[Location] = None) -> float:
        """Distance to F; zero once at or beyond F in the right carotid (going deeper is allowed)."""
        if self._F is None:
            raise RuntimeError("Route.set_target was not called")
        loc = loc or self.locate(p)
        if self.graph.depth(loc, self.chain[2]) >= self._F_depth:
            return 0.0
        return self.graph.path(loc, self._F)

    def target_direction(self) -> np.ndarray:
        """Route direction at the target (unit, vessel frame)."""
        s = self.s(self.target, self._target)
        return unit(self.point_at(s + 2.0) - self.point_at(s - 2.0))

    # ------------------------------------------------------------------------ 2D ----
    def task_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Centerline points (with radii) of the task region, for the 2D box."""
        g = self.graph
        pts, rad = [], []
        for name in TASK_BOUNDS_BRANCHES:
            if name in g.index:
                b = g.index[name]
                pts.append(g.coords[b])
                rad.append(g.radii[b])
        if len(self.chain) > 3 and self.chain[3] in g.index:
            b = g.index[self.chain[3]]
            rcca_len = g.length[g.index[self.chain[2]]]
            keep = g.cum[b] <= max(0.0, TASK_BOUNDS_RCCA_REST_MM - rcca_len)
            pts.append(g.coords[b][keep])
            rad.append(g.radii[b][keep])
        return np.concatenate(pts), np.concatenate(rad)


def to_tracking2d(points: np.ndarray, fluoroscopy) -> np.ndarray:
    """Vessel-frame points -> the fluoroscopy's 2D image coordinates (mm)."""
    from eve.util.coordtransform import tracking3d_to_2d, vessel_cs_to_tracking3d
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return tracking3d_to_2d(vessel_cs_to_tracking3d(
        points, fluoroscopy.image_rot_zx, fluoroscopy.image_center, None)).astype(np.float64)


def task_box_2d(route: Route, fluoroscopy) -> Tuple[np.ndarray, np.ndarray]:
    """(low, high) of the task region in image coordinates: every task centerline point widened by
    its radius (a sphere projects to a disc of the same radius) and a margin."""
    points, radii = route.task_points()
    projected = to_tracking2d(points, fluoroscopy)
    low = np.min(projected - radii[:, None], axis=0) - TASK_BOUNDS_MARGIN_MM
    high = np.max(projected + radii[:, None], axis=0) + TASK_BOUNDS_MARGIN_MM
    return low, high
