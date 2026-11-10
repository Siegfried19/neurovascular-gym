"""Scripted controller for the two-device task (guidewire + the lab's hook catheter): the teacher
of the second training round (start-point curriculum, demonstrations, evaluation reference, reward
sanity check). It reads 3D truth from the simulator and is only used in training, never deployed.

Origin: the lab's ``reference/surgery-all-in-one/19-phantom-sofa/arch_catheter.py`` (``Mouth``,
``Stage``, ``Params``, ``Procedure``, ``split``), carried over to our env, retuned on 2026-11-06 for
the real-mass setting (``configs/two_devices_real_mass.yaml``: real masses, 2 ms physics step, the
hook catheter at a real catheter's stiffness, each device touching the wall with its own radius)
and extended past the right common carotid into the right internal carotid (``configs/
two_devices_ica.yaml``). The geometry is our own numpy (no ``tree.py`` / ``vendor/pnav``). Every
step it reads both instruments' polylines out of ``simulation.dof_positions`` (vessel frame, tip
first; the shorter instrument's tip is counted back from the leading one), ``inserted_lengths``,
``rotations``, and the env's referee (``env.monitors["phase_detector"]``) for the success test.

The task's end point comes from the target: a target past the carotid's ICA/ECA split (the
``two_devices_ica`` config: split + 20-40 mm) means wire to the target, catheter into the ICA;
a target in the common carotid (the round-one configs: 35-55 mm) means the old end point (catheter
>= 15 mm into the carotid). In both the wire must lead by 15-45 mm, held for 8 steps.

Stages (``stage`` / ``substage``) and the referee's phases they pass through:

=========  ==================================================================  ==============
stage      what                                                                phases
=========  ==================================================================  ==============
approach   both round the arch, wire ``lead_mm`` (40) ahead, until the catheter  P0 -> P1
           tip is ``past_mouth_mm`` (33) past the trunk's mouth along the aorta
place      wire back behind the hook (d = -37); hook down; its base pulled to   P1 -> P2
           the mouth + offset (4 mm; 0, 8, -4, 12 on later tries), reached from
           beyond
turn       hook turned up towards the trunk's axis, standing still; the soft    P2
           hook flips over the top and settles about 50 deg off, so it is aimed
           at -30 deg with a 20 deg tolerance; a timeout places it again
pull       3 mm/s pull until the tip slides into the trunk (>= 2 mm)            P2
seat       pull on until 20 mm deep                                            P2 -> P3
wire       ``out``: wire 16 mm ahead of the catheter (the catheter waits: pushing P3 -> P4
           it would only bow the shared shaft); ``aim``: the wire turned until
           its J bend points away from the subclavian; ``fork``: 25 mm/s
           bursts, one step back whenever the tip gains nothing for two steps
           or the catheter tip drops 3 mm; ``rsaback``: into the subclavian ->
           back to 30 mm in the trunk, aim, burst again; ``up``: up the carotid
ica        (ICA targets) the wire on up the carotid: aimed away from the         P4
           downward branch at 64 mm and towards the ICA 20 mm below the split,
           pushed slowly while aimed; into the ECA / the downward branch ->
           back below, aim again; anchored ``ica_anchor_mm`` (30) past the split
follow     catheter over the anchored wire: 8 mm/s in the trunk, 3 mm/s at its  P4
           end and past the split, the wire's base left alone (feeding a wedged
           tip only bows the chain; a tip that slips below the split is sent up
           again);
           a stall becomes push-3-pause-3 pulses, a slip (>= 3 mm in a step)
           a short brake; stops short of the catheter's goal
trim       wire to the target, catheter to its goal, wire 15-45 mm ahead, held   P4 -> success
           8 steps
=========  ==================================================================  ==============

A catheter that comes out of the trunk by more than 6 mm while the wire works sends the wire back
behind the hook and repeats turn / pull / seat (a try); the brief dip to 1-2 mm outside during the
bursts is left alone (the wire's breakthrough pulls it back).

What the simulator does that shaped it (measured 2026-11-06 on ``two_devices_real_mass.yaml``;
the 2026-10-31 numbers were for the stiff, heavy devices and no longer hold):

* Both devices are one chain of nodes in BeamAdapter, its length the leading device's inserted
  length. Pushing the leading wire while its tip is blocked only bows the chain in the arch, and
  every point of the chain, the catheter tip included, moves back; pushing the trailing catheter
  adds no length, it slides the catheter 1:1 along the wire towards the wire's tip (which it can
  never pass unless it becomes the leader). So the catheter is lost whenever the wire is pushed
  against an obstacle for long (the old "climb" lost it in 8 steps), and the catheter advances only
  when the wire's tip is free or anchored.
* The soft hook flips past the top when turned and settles about 50 deg off; aiming at 0 deg never
  settles (40 steps lost per try), aiming at -30 deg with 20 deg tolerance settled 4/4 first time.
* Pulling the seated hook deeper stops at ~23 mm (the stiff hook used to go on); 35 mm of pull
  took it out of the trunk.
* The wire's 30 mm soft tip stops against the trunk's wall 9-13 mm short of its end. Slow pushes
  (4-8 mm/s) with or without turning never passed in 20-25 tries; 25 mm/s bursts passed 8/8 within
  10-20 steps, the tip jumping 10 mm in a step once through (the stored bow releasing), with the
  catheter tip dipping to 1-2.5 mm outside the mouth meanwhile and back to 18-20 mm after.
* Which branch the burst enters depends on where the J bend points: unaimed 4 carotid / 4
  subclavian; aimed away from the subclavian (the bend's direction read off the polyline follows
  the turn about 2/3) 4/4 carotid. At the ICA/ECA split (both branches 3.5 mm radius, 4 deg apart)
  aiming at the ICA side gave 6/6 ICA; the downward branch at 64 mm catches an unaimed tip.
* The catheter following the anchored wire moves 1:1 in the trunk and the carotid (0.5-1.3 mm per
  step at 8 mm/s) and catches at the trunk's end and at the split; push-pause pulses get it through
  in 10-20 steps. Its tip force rises to 2.8-3.2 N going into the ICA (the shell's 2.42 N limit
  holds a push for a step or two).
* The safety shell (``actions.TwoDeviceGuarded``) holds a push back when that tip's contact force
  is over its limit, and lets the two tips cross the same-length band only by speeding up the device
  moving that way: the controller holds the catheter still for the crossing step and lets the wire
  jump (``_cross``), and backs the wire off when the shell keeps holding its push back.

Result (acceptance, 2026-11-07, ``configs/two_devices_ica.yaml``, seeds 1000-1019, from the insertion
point, 800 steps at most, 4 processes at a time): **17/20 reached the referee's success** in 493-587
steps (median 514); every stage up to the bursts passed 20/20, the bursts took the wire into the
carotid in 17/20, and every run that got there finished. The three failures are one trajectory
family (the stages before the target matters are the same on every seed up to the simulator's
thread timing): the wire tip reached the trunk's end but never left the bulb for either branch,
55 stalls and 5 buckles of bursting dragged the catheter out past -12 mm, and the three re-seats
that followed got no further (an untried remedy: after a few stalls at the bulb, a longer retreat
with the wire turned by a quarter turn before bursting again). Forces: wire tip at most 0.43 N on
the successes (0.76 N on a failure), catheter tip 2.8-3.5 N, the peaks while it enters the ICA.
Per run 16-23 minutes of wall time. One brief referee ``dislodged`` per success (the bursts),
``wrong_vessel`` 3 times in 20 runs (the subclavian, left again). Earlier version: 2/4 on the
first acceptance attempt, the follow stage fed the wire against a wedged tip (fixed) and the
bursts lost the catheter at -6 mm (the loss line is -12 mm during the bursts now).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


# ------------------------------------------------------------------ geometry ----
def arclen(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, float)
    if len(points) < 2:
        return np.zeros(len(points))
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros_like(v)


def point_at(points: np.ndarray, s: float, cum: Optional[np.ndarray] = None) -> np.ndarray:
    """The point at arc length ``s`` along a polyline (clamped to its ends)."""
    points = np.asarray(points, float)
    cum = arclen(points) if cum is None else cum
    return np.array([np.interp(s, cum, points[:, k]) for k in range(3)])


def project(points: np.ndarray, p: np.ndarray, cum: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Closest point on a polyline to ``p``: (arc length there, distance)."""
    points = np.asarray(points, float)
    p = np.asarray(p, float)
    cum = arclen(points) if cum is None else cum
    a, b = points[:-1], points[1:]
    ab = b - a
    len2 = np.maximum(np.einsum("ij,ij->i", ab, ab), EPS)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / len2, 0.0, 1.0)
    feet = a + t[:, None] * ab
    dist = np.linalg.norm(feet - p, axis=1)
    i = int(np.argmin(dist))
    return float(cum[i] + t[i] * (cum[i + 1] - cum[i])), float(dist[i])


def signed_angle(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float:
    """Right-handed angle about ``axis`` that turns ``v_from`` onto ``v_to`` (both projected
    onto the plane normal to the axis); 0 when either is degenerate."""
    axis = unit(axis)
    a = np.asarray(v_from, float) - axis * float(np.dot(v_from, axis))
    b = np.asarray(v_to, float) - axis * float(np.dot(v_to, axis))
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return 0.0
    return float(np.arctan2(np.dot(axis, np.cross(a, b)), np.dot(a, b)))


def split(dofs_tip_first: np.ndarray, inserted: Sequence[float], step: float = 1.0) -> List[np.ndarray]:
    """Each instrument's deployed part, tip first, resampled every ``step`` mm, out of the
    combined beam (tip first). Coaxial instruments: one that is ``x`` mm shorter than the
    longest starts ``x`` mm down the same curve, counted back from the leading tip (the node
    polyline can be a few mm shorter than the inserted length, so counting from the base
    would misplace the shorter one's tip). Same as the lab's ``arch_catheter.split``."""
    dofs = np.asarray(dofs_tip_first, float)
    cum = arclen(dofs)
    lead = min(float(max(inserted)), float(cum[-1]))
    out = []
    for length in inserted:
        s = np.arange(max(lead - float(length), 0.0), lead + 1e-6, step)
        if len(s) == 0:
            s = np.array([lead])
        out.append(np.stack([np.interp(s, cum, dofs[:, k]) for k in range(3)], 1))
    return out


# ------------------------------------------------------------------ anatomy ----
class Mouth:
    """Where a child vessel leaves its parent: the first point of the child's centerline outside
    the parent's lumen (the centerline starts in the middle of the parent, so its first millimetres
    are inside it). The lab's ``arch_catheter.Mouth``."""

    def __init__(self, parent: np.ndarray, parent_r: np.ndarray, child: np.ndarray,
                 child_r: np.ndarray) -> None:
        self.parent, self.child = np.asarray(parent, float), np.asarray(child, float)
        self._cp, self._cc = arclen(self.parent), arclen(self.child)
        self._rp = np.asarray(parent_r, float)
        k = len(self.child) - 1
        for i, q in enumerate(self.child):
            s, d = project(self.parent, q, self._cp)
            if d >= float(np.interp(s, self._cp, self._rp)):
                k = i
                break
        self.a0 = float(self._cc[k])
        self.centre = self.child[k].copy()
        self.axis = unit(point_at(self.child, self.a0 + 8.0, self._cc) - self.centre)
        near = (self._cc >= self.a0) & (self._cc <= self.a0 + 10.0)
        self.radius = float(np.median(np.asarray(child_r, float)[near])) if near.any() else float(child_r[-1])
        self.s_parent = project(self.parent, self.centre, self._cp)[0]

    def depth(self, p: np.ndarray) -> Tuple[float, float]:
        """(mm along the child beyond the mouth, mm from the child's centerline)."""
        s, d = project(self.child, p, self._cc)
        return s - self.a0, d

    def outside_parent(self, p: np.ndarray) -> float:
        s, d = project(self.parent, p, self._cp)
        return d - float(np.interp(s, self._cp, self._rp))


# vessels the controller tells apart; every other centerline of the tree is "other"
GROUPS = {"aorta": ("aorta", "aorta_inlet"), "bct": ("bct",), "rcca": ("rcca", "rcca_rest"),
          "rsa": ("rsa", "rsa_rest"), "lcca": ("lcca", "lcca_rest"), "lsa": ("lsa", "lsa_rest")}


def _chain(tree, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    pts, rad = [], []
    for name in names:
        if name in tree.keys():
            c = np.asarray(tree[name].coordinates, float)
            r = np.asarray(getattr(tree[name], "radii", np.full(len(c), 1.0)), float)
            if pts and np.linalg.norm(c[0] - pts[-1][-1]) < 1e-6:
                c, r = c[1:], r[1:]
            pts.append(c)
            rad.append(r)
    return np.vstack(pts), np.concatenate(rad)


class Anatomy:
    """The route's vessels from the vessel tree (BiroPhantomArch ``rebuilt_cut`` names): aorta with
    the three arch mouths, brachiocephalic trunk, right carotid chain (``rcca`` then
    ``rcca_rest``, arc length 0 at the trunk's end = the carotid's opening), the carotid's ICA/ECA
    split and downward branch, and the lateral directions the wire's J bend is aimed at."""

    def __init__(self, vessel_tree, eca_branch: str = "v8", down_branch: str = "v7") -> None:
        self.tree = vessel_tree
        self.aorta, self.aorta_r = _chain(vessel_tree, ["aorta"])
        self._ca = arclen(self.aorta)
        self.lines: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for branch in vessel_tree.branches:
            c = np.asarray(branch.coordinates, float)
            if len(c) < 2:
                continue
            r = np.asarray(getattr(branch, "radii", np.full(len(c), 1.0)), float)
            self.lines[branch.name] = (c, r, arclen(c))
        self.group_of = {name: group for group, names in GROUPS.items() for name in names}
        self.mouths = {}
        for child in ("lsa", "lcca", "bct"):
            c, r, _ = self.lines[child]
            self.mouths[child] = Mouth(self.aorta, self.aorta_r, c, r)
        self.bct_mouth = self.mouths["bct"]
        self.bct, self.bct_r = _chain(vessel_tree, ["bct"])
        self._cb = arclen(self.bct)
        self.bct_end = self.bct[-1].copy()
        self.bct_beyond_mouth = float(self._cb[-1] - self.bct_mouth.a0)
        self.rcca, self.rcca_r = _chain(vessel_tree, ["rcca", "rcca_rest"])
        self._cr = arclen(self.rcca)
        self.rcca_task_len = float(arclen(np.asarray(vessel_tree["rcca"].coordinates, float))[-1])
        self.rsa, self.rsa_r = _chain(vessel_tree, ["rsa", "rsa_rest"])
        self._cs = arclen(self.rsa)
        # route: aorta up to the trunk's junction, trunk, carotid chain (for look-ahead steering)
        s_junction = project(self.aorta, self.bct[0], self._ca)[0]
        keep = self._ca < s_junction
        self.route = np.vstack([self.aorta[keep], self.bct, self.rcca[1:]])
        self._cro = arclen(self.route)
        self.s_route_bct0 = float(self._cro[int(keep.sum())])
        self.s_route_rcca0 = self.s_route_bct0 + float(self._cb[-1])
        ins = np.asarray(vessel_tree.insertion.position, float)
        self.s_insertion = project(self.aorta, ins, self._ca)[0]
        # the carotid beyond its first forks (ICA targets): the ICA/ECA split is where the first
        # side branch that goes up leaves the chain (``eca_branch``, the rebuilt phantom's ``v8``, at
        # 90.9 mm); the chain beyond it is the ICA. ``down_branch`` (``v7``, 63.8 mm) goes downwards
        # and catches an unaimed tip. Trees without them: no split (inf), no aiming targets
        self.eca_branch, self.down_branch = eca_branch, down_branch
        self.split_mm, self.down_mm = np.inf, np.inf
        self.eca = self.down = None
        if eca_branch in self.lines:
            self.eca, self.eca_r, self._ce = self.lines[eca_branch]
            self.split_mm = project(self.rcca, self.eca[0], self._cr)[0]
        if down_branch in self.lines:
            self.down, self.down_r, self._cd = self.lines[down_branch]
            self.down_mm = project(self.rcca, self.down[0], self._cr)[0]
        # lateral directions the J bend is aimed at: away from the subclavian at the trunk's end,
        # the ICA's side at the split, away from the downward branch next to it
        end, axis_end = self.bct[-1], unit(self.bct[-1] - point_at(self.bct, self._cb[-1] - 8.0, self._cb))
        self.lat_trunk = -lateral(point_at(self.rsa, 10.0, self._cs), end, axis_end)
        self.lat_ica = self.lat_down_away = None
        if np.isfinite(self.split_mm):
            o = point_at(self.rcca, self.split_mm, self._cr)
            axis = unit(o - point_at(self.rcca, self.split_mm - 8.0, self._cr))
            self.lat_ica = lateral(point_at(self.rcca, self.split_mm + 10.0, self._cr), o, axis)
        if np.isfinite(self.down_mm):
            o = point_at(self.rcca, self.down_mm, self._cr)
            axis = unit(o - point_at(self.rcca, self.down_mm - 8.0, self._cr))
            self.lat_down_away = -lateral(point_at(self.down, 8.0, self._cd), o, axis)

    # -- where a point is --------------------------------------------------------------
    def s_aorta(self, p: np.ndarray) -> Tuple[float, float]:
        """(arc length along the aorta's centerline, distance from it)."""
        return project(self.aorta, p, self._ca)

    def in_aorta(self, p: np.ndarray, margin: float = 0.0) -> bool:
        s, d = self.s_aorta(p)
        return d < float(np.interp(s, self._ca, self.aorta_r)) + margin

    def vessel_of(self, p: np.ndarray) -> str:
        """``aorta`` while inside the aorta's lumen, else the vessel group of the centerline whose
        tube the point is deepest in (``other:<name>`` for vessels off the route's groups)."""
        if self.in_aorta(p):
            return "aorta"
        best, best_gap = None, np.inf
        for name, (c, r, cum) in self.lines.items():
            s, d = project(c, p, cum)
            gap = d - float(np.interp(s, cum, r))
            if gap < best_gap:
                best, best_gap = name, gap
        group = self.group_of.get(best)
        return group if group is not None else f"other:{best}"

    def depth_bct(self, p: np.ndarray) -> Tuple[float, float]:
        """(mm into the trunk beyond its mouth, mm from its centerline)."""
        return self.bct_mouth.depth(p)

    def depth_rcca(self, p: np.ndarray) -> Tuple[float, float]:
        """(mm along the right carotid chain from its opening, mm from its centerline)."""
        return project(self.rcca, p, self._cr)

    def depth_rsa(self, p: np.ndarray) -> Tuple[float, float]:
        return project(self.rsa, p, self._cs)

    def s_route(self, p: np.ndarray) -> float:
        return project(self.route, p, self._cro)[0]

    def route_point(self, s: float) -> np.ndarray:
        return point_at(self.route, s, self._cro)

    def rcca_point(self, depth: float) -> np.ndarray:
        return point_at(self.rcca, depth, self._cr)


    def carotid_of(self, p: np.ndarray) -> Tuple[str, float, float]:
        """Where a point is on the carotid tree: (``cca`` / ``ica`` / ``eca`` / ``down`` / the
        vessel group elsewhere, mm into that branch (the ICA's from the split; nan elsewhere), mm
        along the carotid chain from its opening)."""
        v = self.vessel_of(p)
        s, dc = self.depth_rcca(p)
        if self.eca is not None and v == f"other:{self.eca_branch}":
            return "eca", project(self.eca, p, self._ce)[0], s
        if self.down is not None and v == f"other:{self.down_branch}":
            return "down", project(self.down, p, self._cd)[0], s
        if v == "rcca":
            if s > self.split_mm + 2.0:
                if self.eca is not None:
                    s8, d8 = project(self.eca, p, self._ce)
                    if d8 < dc - FORK_MARGIN_MM and s8 > 2.0:
                        return "eca", s8, s
                return "ica", s - self.split_mm, s
            return "cca", s, s
        return v, np.nan, s


def lateral(p: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Unit direction from ``origin`` to ``p`` with the ``axis`` component removed."""
    v = np.asarray(p, float) - np.asarray(origin, float)
    v = v - axis * float(np.dot(v, axis))
    return unit(v)


def wire_bend(wire: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(unit direction of the J bend across the shaft, unit shaft direction towards the tip) read
    off the wire polyline (tip first, 1 mm): the tip's first 6 mm against the shaft 12-24 mm back.
    None when the bend is too small to read (or the line too short)."""
    w = np.asarray(wire, float)
    if len(w) < 25:
        return None, None
    tipdir, shaft = unit(w[0] - w[6]), unit(w[12] - w[24])
    bend = tipdir - shaft * float(np.dot(tipdir, shaft))
    if np.linalg.norm(bend) < 0.15:
        return None, shaft
    return unit(bend), shaft


def aim_angle(wire: np.ndarray, lateral_target: np.ndarray) -> Optional[float]:
    """Signed angle (rad, right-handed about the shaft axis towards the tip) that turns the J bend
    onto the wanted lateral direction; a positive wire rotation command turns the bend that way
    (the bend follows the command by about two thirds). None when the bend cannot be read."""
    bend, shaft = wire_bend(wire)
    if bend is None:
        return None
    t = np.asarray(lateral_target, float)
    t = t - shaft * float(np.dot(t, shaft))
    if np.linalg.norm(t) < 1e-6:
        return None
    return signed_angle(bend, unit(t), shaft)


# ---------------------------------------------------------------- state readout ----
WIRE, CATH = 0, 1
FORK_MARGIN_MM = 0.5


def scene_of(env):
    """(intervention, simulation) from a GuidewireEnv, a stEVE env or an intervention."""
    intervention = getattr(env, "intervention", env)
    return intervention, intervention.simulation


def hook_frame(cath: np.ndarray, hook_len: float, shaft_back: float = 15.0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tip, base, shaft) points of the catheter polyline (tip first, 1 mm): the tip, the start of
    the hook's bend ``hook_len`` mm back, a shaft point ``shaft_back`` mm further back."""
    k = int(min(round(hook_len), len(cath) - 1))
    j = int(min(k + round(shaft_back), len(cath) - 1))
    return cath[0], cath[k], cath[j]


def bend_angle(tip: np.ndarray, base: np.ndarray, shaft: np.ndarray, desired: np.ndarray,
               min_bend_mm: float = 1.0) -> float:
    """Signed angle about the distal shaft axis (shaft -> base) from the bend (tip - base, across
    the axis) to ``desired``; 0 when the bend is shorter than ``min_bend_mm`` (the lab's ``_phi``).
    A positive rotation command turns the bend right-handed about the axis, so ``w = k * angle``
    turns the bend towards ``desired``."""
    axis = unit(base - shaft)
    bend = (tip - base) - axis * float(np.dot(tip - base, axis))
    if np.linalg.norm(bend) < min_bend_mm:
        return 0.0
    return signed_angle(bend, desired, axis)


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class Readout:
    """What the controller reads from the simulator each step (vessel frame; mm, rad).

    Depths: ``depth_bct_*`` mm into the brachiocephalic trunk beyond its mouth (only meaningful
    when that tip's vessel is ``bct``); ``depth_rcca_*`` mm along the right carotid chain from its
    opening (= the trunk's end), -inf unless that tip is in it. ``wire_branch`` resolves the trunk's
    fork: ``rcca`` / ``rsa`` once the tip is more than 3 mm along one of them and nearer its
    centerline, else ``bct``; elsewhere the tip's vessel."""

    wire: np.ndarray
    cath: np.ndarray
    L: np.ndarray
    rot: np.ndarray
    target: np.ndarray
    d: float = 0.0
    s_wire: float = 0.0
    s_cath: float = 0.0
    vessel_wire: str = "aorta"
    vessel_cath: str = "aorta"
    depth_bct_wire: float = -np.inf
    depth_bct_cath: float = -np.inf
    depth_rcca_wire: float = -np.inf
    depth_rcca_cath: float = -np.inf
    wire_branch: str = "aorta"
    wire_branch_depth: float = np.nan
    target_depth: float = np.nan
    s_base: float = 0.0           # catheter hook's base along the aorta
    phi_up: float = 0.0           # hook's bend against the trunk's axis (0 = turned up)
    phi_down: float = 0.0         # against minus that axis (0 = turned down)
    up_angle_deg: float = np.nan  # catheter tip direction against +z
    wire_carotid: str = ""        # cca / ica / eca / down / the vessel group elsewhere
    wire_carotid_depth: float = np.nan   # mm into that branch (the ICA's from the split)
    wire_chain: float = -np.inf   # wire tip, mm along the carotid chain (-inf unless in it)
    cath_carotid: str = ""
    cath_carotid_depth: float = np.nan
    cath_chain: float = -np.inf

    @property
    def wire_tip(self) -> np.ndarray:
        return self.wire[0]

    @property
    def cath_tip(self) -> np.ndarray:
        return self.cath[0]

    @property
    def cath_in_bct(self) -> bool:
        return self.vessel_cath == "bct"

    @property
    def wire_in_rcca(self) -> float:
        """Wire tip depth in the carotid chain, -inf when not in it (fork resolved)."""
        return self.wire_branch_depth if self.wire_branch == "rcca" else -np.inf


def target_vessel(intervention) -> np.ndarray:
    """The target point in the vessel frame (stEVE keeps it in the tracking frame)."""
    target = intervention.target
    if hasattr(target, "coordinates_vessel"):
        return np.asarray(target.coordinates_vessel, float).reshape(3)
    from eve.util.coordtransform import tracking3d_to_vessel_cs
    fl = intervention.fluoroscopy
    return np.asarray(tracking3d_to_vessel_cs(np.asarray(target.coordinates3d, float).reshape(1, 3),
                                              fl.image_rot_zx, fl.image_center), float).reshape(3)


def read(env, anatomy: Anatomy, hook_len: float = 26.0) -> Readout:
    """Both instruments from the simulator: ``simulation.dof_positions`` (the combined beam, vessel
    frame, tip first; the same points ``fluoroscopy.instrument_trackings3d`` rotates into the image
    frame), ``inserted_lengths`` and ``rotations``."""
    intervention, sim = scene_of(env)
    L = np.asarray(sim.inserted_lengths, float).reshape(-1)
    rot = np.asarray(sim.rotations, float).reshape(-1)
    wire, cath = split(np.asarray(sim.dof_positions, float), L)
    out = Readout(wire=wire, cath=cath, L=L, rot=rot, target=target_vessel(intervention))
    out.d = float(L[WIRE] - L[CATH])
    an = anatomy
    out.s_wire = an.s_aorta(wire[0])[0]
    out.s_cath = an.s_aorta(cath[0])[0]
    out.vessel_wire = an.vessel_of(wire[0])
    out.vessel_cath = an.vessel_of(cath[0])
    out.depth_bct_wire = an.depth_bct(wire[0])[0]
    out.depth_bct_cath = an.depth_bct(cath[0])[0]
    if out.vessel_cath == "rcca":
        out.depth_rcca_cath = an.depth_rcca(cath[0])[0]
    if out.vessel_wire == "rcca":
        out.depth_rcca_wire = an.depth_rcca(wire[0])[0]
    out.wire_branch, out.wire_branch_depth = out.vessel_wire, np.nan
    if out.vessel_wire in ("bct", "rcca", "rsa"):
        # the two vessels leave the trunk's end through a common bulb (radius ~5.6 mm for ~8 mm):
        # a tip counts as in one of them only when clearly nearer its centerline
        sc, dc = an.depth_rcca(wire[0])
        ss, ds = an.depth_rsa(wire[0])
        if dc <= ds - FORK_MARGIN_MM and sc > 3.0:
            out.wire_branch, out.wire_branch_depth = "rcca", sc
        elif ds < dc - FORK_MARGIN_MM and ss > 3.0:
            out.wire_branch, out.wire_branch_depth = "rsa", ss
        else:
            out.wire_branch, out.wire_branch_depth = "bct", out.depth_bct_wire
    out.target_depth = an.depth_rcca(out.target)[0]
    label, side, chain = an.carotid_of(wire[0])
    out.wire_carotid, out.wire_carotid_depth = label, side
    out.wire_chain = chain if label in ("cca", "ica", "eca", "down") else -np.inf
    label, side, chain = an.carotid_of(cath[0])
    out.cath_carotid, out.cath_carotid_depth = label, side
    out.cath_chain = chain if label in ("cca", "ica", "eca", "down") else -np.inf
    tip, base, shaft = hook_frame(cath, hook_len)
    axis = an.bct_mouth.axis
    out.s_base = an.s_aorta(base)[0]
    out.phi_up = bend_angle(tip, base, shaft, axis)
    out.phi_down = bend_angle(tip, base, shaft, -axis)
    k = min(3, len(cath) - 1)
    if k > 0:
        direction = unit(cath[0] - cath[k])
        out.up_angle_deg = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
    return out


# ---------------------------------------------------------------- controller ----
@dataclass
class ScriptParams:
    """Speeds in mm/s and rad/s, lengths in mm, times in control steps. Numbers measured on
    ``configs/two_devices_real_mass.yaml`` (2026-11-06) unless the comment says otherwise."""

    # caps (the plan's TwoDeviceGuarded speed limits)
    v_wire_max: float = 40.0
    w_wire_max: float = float(np.pi)
    v_cath_max: float = 25.0
    w_cath_max: float = float(np.pi / 2)
    cross_band_mm: float = 2.5      # the shell's same-length band (protector_mm): crossed in one step
    cross_edge_margin_mm: float = 0.15
    cross_min_move_mm: float = 1.5  # moves smaller than this stop at the band's edge instead of jumping
    blocked_steps: int = 3          # wire push held back by the shell this many steps in a row: back off
    # approach: both round the arch, wire ahead, until the catheter tip is past the trunk's mouth
    v_approach: float = 25.0
    lead_mm: float = 40.0
    lead_start_mm: float = 15.0     # catheter starts when the wire is this far ahead
    k_d: float = 3.0                # 1/s, keeps d where a stage wants it
    past_mouth_mm: float = 33.0     # P0 -> P1 needs 30
    redirect_back_mm: float = 15.0
    redirect_turn_rad: float = 1.6
    # place: wire behind the hook, the hook's base to the trunk's mouth + offset (reached by pulling)
    behind_mm: float = 37.0         # d = -37 (P1 needs <= -35)
    v_retract: float = 40.0
    v_cath_retract: float = 4.0
    offsets_mm: Tuple[float, ...] = (4.0, 0.0, 8.0, -4.0, 12.0)
    place_aim_deg: float = 180.0    # hook turned away from the roof (down) while it is pulled to its place
    over_mm: float = 12.0
    over_tol_mm: float = 2.0
    place_tol_mm: float = 1.5
    place_down_deg: float = 30.0
    k_place: float = 2.0
    v_place: float = 8.0
    place_timeout: int = 45
    # turn: the hook up towards the trunk's axis. The soft hook (E 660 MPa) flips over the top and
    # settles about -50 deg; aimed at 0 it timed out every time, at -30 with 20 deg tolerance it
    # settled first time on 4/4 seeds
    k_rot: float = 1.0
    aims_deg: Tuple[float, ...] = (-30.0, -20.0, 20.0, -35.0, 35.0)
    turn_tol_deg: float = 20.0
    turn_slow_deg: float = 60.0
    w_turn_near: float = float(np.pi / 2)
    turn_hold: int = 3
    turn_timeout: int = 40
    retry_on_turn_timeout: bool = True    # turn timed out: hook down and place again (next offset/aim)
    adapt_aim_deg: float = 30.0     # no tries left: pull aiming where the hook settled, within +-this
    # pull / seat: a slow pull slides the hook's tip into the trunk
    v_pull: float = 3.0
    pull_max_mm: float = 18.0
    engage_mm: float = 2.0
    wrong_mm: float = 3.0
    v_free: float = 3.5             # under the shell's 3.75 mm/s cap for a catheter pushed in a branch ahead of the wire
    lower_timeout: int = 25
    v_seat: float = 3.0
    seat_mm: float = 20.0           # pulled this deep before the wire goes out (pulling further stops at ~23)
    seat_hold: int = 2
    seat_pull_max_mm: float = 25.0
    # wire out: the wire ahead of the catheter; the catheter waits (pushing it bows the shaft)
    out_lead_mm: float = 16.0
    v_wire_out: float = 30.0
    reseat_mm: float = 8.0          # catheter tip shallower than this in the trunk: a slow pull
    v_reseat: float = 3.0
    lost_mm: float = -6.0           # catheter tip this far out of the trunk while the wire works: lost
    fork_lost_mm: float = -12.0     # ... during the bursts (they dip it to -1..-5 mm; the breakthrough or
                                    # a retreat of the wire brings it back along the chain)
    fast_recover: bool = True       # lost near the mouth: wire back, turn up and pull again (no new place)
    fast_recover_mm: float = 15.0
    # aim: turn the wire until its J bend (read off the polyline) points where the next branch is
    aim: bool = True
    aim_tol_deg: float = 35.0
    aim_k: float = 2.0              # rad/s per rad of error, capped at w_wire_max
    aim_steps: int = 12             # give up aiming after this many steps and push anyway
    aim_from_mm: float = 20.0       # (carotid) aim at the ICA side from this far below the split
    aim_v: float = 4.0              # push while aimed near the split
    # fork: bursts through the trunk's end. The soft tip stops against the wall 9-13 mm short of
    # the end; 25 mm/s bursts pass (8/8), slower pushes never did
    v_fork: float = 25.0
    w_fork: float = 0.0             # turning during the bursts (the aim stage turns instead)
    fork_stall_steps: int = 2       # tip gained < 1 mm over this many steps: back off one step
    fork_retreat_mm: float = 1.5    # tip lost this much over two steps: back off
    fork_cath_drop_mm: float = 3.0  # catheter tip retreated this much since the burst began: back off
    fork_backoff_steps: int = 1
    v_backoff: float = 15.0
    fork_max_tries: int = 60
    rsa_back_mm: float = 7.0        # wire this deep in the right subclavian: retreat (the referee counts 8)
    rsa_retreat_to_mm: float = 30.0 # ... back into the trunk to this depth, then aim and burst again
    rsa_retreat_turn: float = float(np.pi)   # turned while retreating when aiming is off
    in_rcca_mm: float = 6.0         # wire this deep in the carotid (fork resolved): it is in
    v_trunk: float = 8.0            # (kept for the climb to the fork when aiming is off)
    # up: the wire up the common carotid
    v_rcca: float = 10.0
    up_goal_mm: float = 50.0        # (ICA targets) the ica stage takes over at this carotid depth
    wire_extra_mm: float = -5.0     # (carotid targets) the wire's goal while the catheter follows
    # ica: on up the carotid, through the ICA/ECA split
    v_ica: float = 10.0
    ica_anchor_mm: float = 30.0     # anchored this far past the split (at least the target depth)
    eca_back_mm: float = 6.0        # into the ECA this far: back below the split, aim again
    eca_back_to_mm: float = 10.0
    v7_back_mm: float = 6.0         # into the downward branch this far: back below it
    v_ica_backoff: float = 15.0
    ica_stall_steps: int = 15       # pushed this long without 2 mm gain: back 8 mm
    ica_timeout: int = 150
    # follow: catheter over the anchored wire
    v_follow: float = 8.0
    v_follow_slow: float = 3.0
    slow_from_mm: float = 34.0      # catheter tip this deep in the trunk: slow push towards its end
    slow_below_split_mm: float = 5.0
    follow_stall_steps: int = 8     # catheter tip not gaining 1 mm in this many steps: pulses
    pulse_push_steps: int = 3
    pulse_pause_steps: int = 3
    slip_mm: float = 3.0            # catheter tip moving this much in one step: it slips, brake
    v_brake: float = 15.0
    brake_min_steps: int = 2
    brake_steps: int = 6
    follow_timeout: int = 400
    # trim: wire to the target, catheter to its goal, wire 15-45 mm ahead; then hold
    cath_ica_mm: float = 5.0        # (ICA targets) catheter this far past the split counts as in
    cath_min_mm: float = 15.0       # (carotid targets) catheter this far into the carotid
    cath_goal_offset_mm: float = 22.0   # catheter goal = target depth - this, within the limits below
    k_trim_cath: float = 1.0
    v_trim_cath: Tuple[float, float] = (-15.0, 3.0)
    k_trim_wire: float = 1.5
    v_trim_wire: float = 10.0
    wire_near_mm: float = 6.0       # wire tip this close to the target depth: trim it slowly
    v_trim_wire_near: float = 3.0
    d_lo: float = 17.0
    d_hi: float = 43.0
    d_range: Tuple[float, float] = (15.0, 45.0)
    success_hold: int = 8
    success_margin_mm: float = 0.5  # the hand-made success test keeps this much inside the referee's limits
    hand_tol_mm: float = 5.0
    trim_timeout: int = 120
    # giving up (``StagedController.failed``; user decision 2026-11-09): the start-point curriculum
    # hands the episode to the policy where the script stops, so a stuck script should say so soon
    # instead of burning the step limit. PROVISIONAL numbers from the 2026-11-07 acceptance run
    # (17/20; the per-stage times of that run were lost with the scratch directory): the successes
    # spent about 80 steps in the wire stage (wire out to the carotid), about 100 in ica, 100-120 in
    # follow, placed the hook at the first try and never lost the catheter; the failures cycled at
    # the trunk's end for 500+ steps, losing and re-seating the catheter three times. Budgets are
    # about three times the successes' longest; to be set from the next acceptance run's per-stage
    # times (work log 2026-11-09). A budget counts the steps since the stage was entered (a return
    # to the stage starts it again); no entry = no budget.
    stage_budgets: Dict[str, int] = field(default_factory=lambda: {"wire": 240, "ica": 300, "follow": 450})
    give_up_rewires: int = 2        # the catheter lost out of the trunk and re-seated this many times
    give_up_tries: int = 6          # hook placements (``tries``: a pull that found nothing or another
                                    # vessel, a turn that timed out, a re-seat after a loss); above the
                                    # script's own five places (``offsets_mm``), which it runs through first


STAGES = ("approach", "place", "turn", "pull", "seat", "wire", "ica", "follow", "trim", "done")


class StagedController:
    """The lab's ``arch_catheter.Procedure`` (19-phantom-sofa) carried over to the two-device env and
    extended to the whole task, on three-dimensional truth from the simulator (module docstring).

    Interface:

    * ``act(env)`` -> the normalized 4-vector [wire push, wire turn, catheter push, catheter turn]
      to pass to ``env.step`` (each axis / the instrument's velocity limit). Advances the stages.
    * ``reset(env)`` after every ``env.reset`` (``reset()`` / ``reset(episode_nr)`` use the
      intervention given at construction, as ``NominalAdvisor`` calls it).
    * ``nominal_for_phase(env, phase)`` or ``nominal_for_phase(phase_state)``: the stateless
      suggestion for P1 (hook set), zeros in every other phase; never advances the stages.
    * ``stage`` / ``substage`` (where it is), ``log`` ([(stage, step), ...]), ``tries``, ``success``
      (its own test of the referee's success, met for 8 steps), ``last`` (diagnostics).
    * ``failed`` / ``failure_reason``: the script gave up (``ScriptParams``, "giving up": a stage
      ran past its step budget, too many hook placements, the catheter lost and re-seated too
      often). From then on ``act`` returns zeros; the caller ends the episode or hands it over
      (the start-point curriculum takes the policy over where the script stopped, user decision
      2026-11-09). Logged once as ``("failed:<reason>", step)``.

    ``params`` (``ScriptParams``) or keyword overrides of its fields; ``intervention`` /
    ``phase_detector`` are accepted so ``NominalAdvisor`` can build it (``controller: {type:
    guidewire_rl.env.two_device.script.StagedController}``).
    """

    def __init__(self, params: Optional[ScriptParams] = None, hook_len: float = 26.15,
                 limits: Optional[Sequence[float]] = None, intervention=None, phase_detector=None,
                 **overrides) -> None:
        if overrides:
            params = replace(params or ScriptParams(), **overrides)
        self.p = params or ScriptParams()
        self.hook_len = float(hook_len)       # lab HOOK: arc 8 mm x 130 deg + 8 mm straight tip
        self._limits_override = None if limits is None else np.asarray(limits, float)
        self._given_intervention = intervention
        self.phase_detector = phase_detector
        self.anatomy: Optional[Anatomy] = None
        self._tree_id = None
        self.dt = 0.132
        self.shell_dt = 1.0 / 7.5
        self._intervention = None
        self.limits = np.array([40.0, np.pi, 25.0, np.pi / 2])
        self.reset_state()

    # -- bookkeeping -------------------------------------------------------------------
    def reset_state(self) -> None:
        self.stage, self.substage = "approach", ""
        self.steps = 0
        self.tries = 0
        self.rewires = 0
        self.success = False
        self.failed = False
        self.failure_reason = ""
        self._stage_entered = 0
        self.log: List[Tuple[str, int]] = [("approach", 0)]
        self.last: Dict[str, float] = {}
        self._since = 0
        self._from_L = 0.0
        self._over = False
        self._hold = 0
        self._spin = 1.0
        self._count = 0
        self._redirect_from = 0.0
        self._prev_depth_c = np.nan
        self._aim_override: Optional[float] = None
        self._wire_blocked = 0
        self._referee = None
        self.last_physical = np.zeros(4)
        self.readout: Optional[Readout] = None
        self._fork_since, self._fork_hist, self._fork_dbc0, self._fork_tries = -1, [], np.nan, 0
        self._ica_why, self._ica_back_to, self._ica_hist, self._ica_pushed, self._ica_aiming = "", 0.0, [], True, 0
        self._anchor, self._f_hist, self._f_mode, self._f_mode_since = np.nan, [], "steady", 0

    def _env_or_given(self, env):
        if env is None or isinstance(env, (int, np.integer)):
            if self._given_intervention is None:
                raise ValueError("StagedController: no env given and no intervention at construction")
            return self._given_intervention
        return env

    def reset(self, env=None) -> None:
        env = self._env_or_given(env)
        intervention, sim = scene_of(env)
        tree = intervention.vessel_tree
        if self.anatomy is None or self._tree_id != id(tree):
            self.anatomy = Anatomy(tree)
            self._tree_id = id(tree)
        n_sub = int((1.0 / float(intervention.fluoroscopy.image_frequency)) / float(sim.dt_simulation))
        self.dt = max(n_sub, 1) * float(sim.dt_simulation)          # what one step really moves
        self.shell_dt = 1.0 / float(intervention.fluoroscopy.image_frequency)   # what the shell predicts with
        self._intervention = intervention
        if self._limits_override is not None:
            self.limits = self._limits_override.copy()
        elif hasattr(env, "raw_action_limits"):
            self.limits = np.abs(np.asarray(env.raw_action_limits[1], float).reshape(-1))
        else:
            self.limits = np.abs(np.asarray(intervention.velocity_limits, float).reshape(-1))
        self.reset_state()

    def _go(self, stage: str, substage: str = "", L_cath: float = 0.0) -> None:
        if stage != self.stage or substage != self.substage:
            self.log.append((stage if not substage else f"{stage}:{substage}", self.steps))
        if stage != self.stage:
            self._stage_entered = self.steps
        self.stage, self.substage = stage, substage
        self._since, self._from_L, self._hold, self._count = self.steps, L_cath, 0, 0

    def _elapsed(self) -> int:
        return self.steps - self._since

    def _next_try(self, r: Readout) -> None:
        self.tries += 1
        self._over = False
        self._aim_override = None
        self._go("place", "lower", r.L[CATH])

    @property
    def offset(self) -> float:
        return self.p.offsets_mm[min(self.tries, len(self.p.offsets_mm) - 1)]

    @property
    def aim(self) -> float:
        if self._aim_override is not None:
            return self._aim_override
        return float(np.radians(self.p.aims_deg[min(self.tries, len(self.p.aims_deg) - 1)]))

    # -- helpers -----------------------------------------------------------------------
    def _keep_d(self, r: Readout, v_c: float, d_goal: float) -> float:
        """Wire speed that keeps d = L_wire - L_cath at ``d_goal`` while the catheter moves at v_c."""
        return float(v_c + self.p.k_d * (d_goal - r.d))

    def _cross(self, r: Readout, v_w: float, v_c: float) -> Tuple[float, float]:
        """Never end a step with the two tips within the shell's same-length band (``cross_band_mm``,
        the shell's ``protector_mm``) of each other. The shell completes a crossing only by speeding
        up the device already moving that way and never moves the other one, so: a step that would
        land in the band holds the catheter still and lets the wire jump across in one step (the
        band, 5 mm, is less than the wire's 5.3 mm per step); out of reach from here, the wire stops
        just outside the near edge and jumps next step. Predicted with the shell's step time."""
        p, T = self.p, self.shell_dt
        band = p.cross_band_mm
        edge = band + p.cross_edge_margin_mm
        d0 = r.d
        d1 = d0 + (v_w - v_c) * T
        if abs(d1) >= edge:
            return v_w, v_c
        direction = float(np.sign(d1 - d0)) if d1 != d0 else (float(np.sign(d0)) or 1.0)
        if abs(d0) >= edge and abs(d1 - d0) < p.cross_min_move_mm:
            # a small trim that would just touch the band: stop at the edge, do not jump across
            return float(np.clip(v_c + (float(np.sign(d0)) * edge - d0) / T,
                                 -p.v_wire_max, p.v_wire_max)), v_c
        if abs(d0) < edge or np.sign(d0) == -direction:
            goal = direction * edge           # inside, or heading across zero: land beyond the band
        else:
            goal = float(np.sign(d0)) * edge  # moving away but would still be inside: step out
        jump = (goal - d0) / T                # wire speed with the catheter held still
        if abs(jump) <= p.v_wire_max:
            return float(jump), 0.0
        near = float(np.sign(d0)) * edge      # out of reach: wait just outside the near edge
        return float(np.clip((near - d0) / T, -p.v_wire_max, p.v_wire_max)), 0.0

    def _normalize(self, v_w: float, w_w: float, v_c: float, w_c: float) -> np.ndarray:
        p = self.p
        phys = np.array([np.clip(v_w, -p.v_wire_max, p.v_wire_max), np.clip(w_w, -p.w_wire_max, p.w_wire_max),
                         np.clip(v_c, -p.v_cath_max, p.v_cath_max), np.clip(w_c, -p.w_cath_max, p.w_cath_max)])
        self.last_physical = phys
        return np.clip(phys / self.limits, -1.0, 1.0).astype(np.float32)

    def _wire_in_wrong_vessel(self, r: Readout) -> bool:
        v = r.vessel_wire
        if v == "aorta":
            return False
        if v in ("lsa", "lcca", "bct"):
            return self.anatomy.mouths[v].depth(r.wire_tip)[0] > 4.0
        return True

    def _cath_in_wrong_vessel(self, r: Readout) -> bool:
        v = r.vessel_cath
        if v in ("lsa", "lcca"):
            return self.anatomy.mouths[v].depth(r.cath_tip)[0] > self.p.wrong_mm
        return v.startswith("other") or v == "rsa"

    # -- the policy ----------------------------------------------------------------------
    def act(self, env=None) -> np.ndarray:
        """Normalized action for the current simulator state (advances the stage machine)."""
        env = self._env_or_given(env)
        if self.anatomy is None:
            self.reset(env)
        r = read(env, self.anatomy, self.hook_len)
        self._referee = self._referee_state(env)
        self._note_shell(env)
        self.steps += 1
        v_w, w_w, v_c, w_c = self._policy(r)
        v_w, v_c = self._cross(r, v_w, v_c)
        self.readout = r
        self.last = dict(stage=self.stage, substage=self.substage, tries=self.tries, d=r.d,
                         s_cath=r.s_cath, s_base=r.s_base, phi_up=r.phi_up,
                         depth_bct_cath=r.depth_bct_cath, wire_branch=r.wire_branch,
                         wire_branch_depth=r.wire_branch_depth, depth_rcca_cath=r.depth_rcca_cath,
                         target_depth=r.target_depth)
        return self._normalize(v_w, w_w, v_c, w_c)

    def _referee_state(self, env):
        """The env's referee state (``env.monitors["phase_detector"].state``), None without one."""
        monitors = getattr(env, "monitors", None) or {}
        detector = monitors.get("phase_detector") if hasattr(monitors, "get") else None
        return None if detector is None else detector.state

    def _note_shell(self, env) -> None:
        """Count consecutive steps in which the wire was told to push and did not (the shell's force
        limit: a tip pressed against the wall keeps its beam loaded, so it must back off first)."""
        intervention, _ = scene_of(env)
        executed = getattr(intervention, "last_action", None)
        commanded = self.last_physical[0] if self.steps > 0 else 0.0
        if executed is None or commanded <= 1.0:
            self._wire_blocked = 0
            return
        done = float(np.asarray(executed, float).reshape(-1)[0])
        self._wire_blocked = self._wire_blocked + 1 if done < 0.25 * commanded else 0

    def _policy(self, r: Readout) -> Tuple[float, float, float, float]:
        """The stage machine's action, or zeros once the script has given up (``failed``)."""
        if self.failed:
            return 0.0, 0.0, 0.0, 0.0
        out = self._stages(r)
        if self.failed:                       # given up inside a nested call
            return 0.0, 0.0, 0.0, 0.0
        why = self._give_up_reason()
        if why:
            self.failed, self.failure_reason = True, why
            self.log.append((f"failed:{why}", self.steps))
            return 0.0, 0.0, 0.0, 0.0
        return out

    def _give_up_reason(self) -> str:
        """Why the script gives up now ("" to go on): the catheter lost too often, too many hook
        placements, or the current stage past its step budget (``ScriptParams``, "giving up")."""
        p = self.p
        if self.success or self.stage in ("trim", "done"):
            return ""
        if self.rewires >= p.give_up_rewires:
            return f"rewires:{self.rewires}"
        if self.tries >= p.give_up_tries:
            return f"tries:{self.tries}"
        budget = (p.stage_budgets or {}).get(self.stage)
        if budget is not None and self.steps - self._stage_entered >= int(budget):
            return f"budget:{self.stage}"
        return ""

    def _stages(self, r: Readout) -> Tuple[float, float, float, float]:
        stage = self.stage
        if stage == "approach":
            return self._approach(r)
        if stage == "place":
            return self._place(r)
        if stage == "turn":
            return self._turn(r)
        if stage == "pull":
            return self._pull(r)
        if stage == "seat":
            return self._seat(r)
        if stage == "wire":
            return self._wire(r)
        if stage == "ica":
            return self._ica(r)
        if stage == "follow":
            return self._follow(r)
        if stage == "trim":
            return self._trim(r)
        return 0.0, 0.0, 0.0, 0.0

    # approach: wire ahead, both round the arch --------------------------------------------
    def _approach(self, r: Readout):
        p, m = self.p, self.anatomy.bct_mouth
        if self.substage == "redirect":
            # the wire took a side vessel: catheter waits, wire back out turning a quarter
            back = self._redirect_from - r.L[WIRE]
            if back < p.redirect_back_mm or self._wire_in_wrong_vessel(r):
                w_w = p.redirect_turn_rad / max(p.redirect_back_mm / 20.0, self.dt) if back < p.redirect_back_mm else 0.0
                return -20.0, float(w_w), 0.0, 0.0
            self._go("approach", "", r.L[CATH])
        if self._wire_in_wrong_vessel(r) and r.d > 0:
            self._redirect_from = r.L[WIRE]
            self._go("approach", "redirect", r.L[CATH])
            return -20.0, 0.0, 0.0, 0.0
        if r.s_cath >= m.s_parent + p.past_mouth_mm and r.vessel_cath == "aorta":
            self._go("place", "retract", r.L[CATH])
            return self._place(r)
        v_c = p.v_approach if (r.d >= p.lead_start_mm or r.L[CATH] > 1.0) else 0.0
        v_w = self._keep_d(r, v_c, p.lead_mm) if r.L[CATH] > 1.0 or r.d >= p.lead_start_mm else p.v_wire_max
        return float(np.clip(v_w, -p.v_wire_max, p.v_wire_max)), 0.0, v_c, 0.0

    # place: wire behind the hook, hook down, its base pulled to the mouth + offset -----------
    def _place(self, r: Readout):
        p, m = self.p, self.anatomy.bct_mouth
        if self.substage == "retract":
            if r.d > -p.behind_mm:
                return -p.v_retract, 0.0, -p.v_cath_retract, 0.0
            self._over = False
            self._go("place", "", r.L[CATH])
        if self.substage == "rewire":         # the catheter came out of the trunk: wire behind the hook
            if r.d > -p.behind_mm:
                return -p.v_retract, 0.0, 0.0, 0.0
            self._over = False
            if p.fast_recover and abs(r.s_cath - m.s_parent) < p.fast_recover_mm:
                # its tip is still at the mouth: turn the hook up where it is and pull it in again
                self._go("turn", "", r.L[CATH])
                return self._turn(r)
            self._go("place", "", r.L[CATH])
        if self.substage == "lower":
            # hook down again (and out of any vessel) before the next place
            w_c = float(np.clip(p.k_rot * r.phi_down, -p.w_cath_max, p.w_cath_max))
            done = abs(r.phi_down) < np.radians(p.place_down_deg) or self._elapsed() > p.lower_timeout
            v_c = p.v_free if self._cath_in_wrong_vessel(r) or r.cath_in_bct else 0.0
            if done and v_c == 0.0:
                self._over = False
                self._go("place", "", r.L[CATH])
            else:
                return self._keep_d(r, v_c, -p.behind_mm), 0.0, v_c, w_c
        err = m.s_parent + self.offset - r.s_base
        if not self._over:
            if abs(err + p.over_mm) < p.over_tol_mm:
                self._over = True
            else:
                err += p.over_mm
        e_aim = wrap(r.phi_up - np.radians(p.place_aim_deg))
        w_c = float(np.clip(p.k_rot * e_aim, -p.w_cath_max, p.w_cath_max))
        placed = self._over and abs(err) < p.place_tol_mm and abs(e_aim) < np.radians(p.place_down_deg)
        if placed or self._elapsed() > p.place_timeout:
            self._go("turn", "", r.L[CATH])
            return self._turn(r)
        v_c = float(np.clip(p.k_place * err, -p.v_place, p.v_place))
        return self._keep_d(r, v_c, -p.behind_mm), 0.0, v_c, w_c

    # turn: hook up, standing still ------------------------------------------------------------
    def _turn_rate(self, e: float) -> float:
        """Catheter turn rate towards a hook angle error ``e``: full rate far away, slow near (the
        shaft winds up and the hook then whips round, overshooting by up to 50 degrees)."""
        p = self.p
        cap = p.w_cath_max if abs(e) > np.radians(p.turn_slow_deg) else p.w_turn_near
        return float(np.clip(p.k_rot * e, -cap, cap))

    def _turn(self, r: Readout):
        p = self.p
        e = wrap(r.phi_up - self.aim)
        self._hold = self._hold + 1 if abs(e) < np.radians(p.turn_tol_deg) else 0
        timed_out = self._elapsed() > p.turn_timeout
        if timed_out and self._hold < p.turn_hold and p.retry_on_turn_timeout and self.tries + 1 < len(p.offsets_mm):
            self._next_try(r)                 # the hook would not settle up here: place it again
            return self._place(r)
        if (self._hold >= p.turn_hold or timed_out) and r.d <= -p.behind_mm + 2.0:
            if timed_out and self._hold < p.turn_hold:
                # the hook settles off the aim (wound-up shaft): pull aiming elsewhere (on 20 seeds,
                # aiming 15 deg off from the start fixed most of these and broke most of the others)
                if p.adapt_aim_deg > 0:
                    lim = np.radians(p.adapt_aim_deg)
                    self._aim_override = float(np.clip(r.phi_up, -lim, lim))
            self._go("pull", "", r.L[CATH])
            return self._pull(r)
        return self._keep_d(r, 0.0, -p.behind_mm), 0.0, 0.0, self._turn_rate(e)

    # pull: slowly, the hook's tip slides into the trunk --------------------------------------
    def _pull(self, r: Readout):
        p = self.p
        if self.substage == "free":       # tip went up another vessel: push it back out
            if self._cath_in_wrong_vessel(r) and self._elapsed() < 30:
                return self._keep_d(r, p.v_free, -p.behind_mm), 0.0, p.v_free, 0.0
            self._next_try(r)
            return self._place(r)
        if r.cath_in_bct and r.depth_bct_cath >= p.engage_mm:
            self._go("seat", "", r.L[CATH])
            return self._seat(r)
        if self._cath_in_wrong_vessel(r):
            self._go("pull", "free", r.L[CATH])
            return self._pull(r)
        if self._from_L - r.L[CATH] >= p.pull_max_mm:
            self._next_try(r)
            return self._place(r)
        w_c = self._turn_rate(wrap(r.phi_up - self.aim))
        v_c = -p.v_pull
        return self._keep_d(r, v_c, -p.behind_mm), 0.0, v_c, w_c

    # seat: a little more pull drives the tip deeper ----------------------------------------------
    def _seat(self, r: Readout):
        p = self.p
        if not r.cath_in_bct and r.depth_bct_cath < 0.0:
            self._next_try(r)
            return self._place(r)
        deep = r.cath_in_bct and r.depth_bct_cath >= p.seat_mm
        self._hold = self._hold + 1 if deep else 0
        if self._hold >= p.seat_hold or self._from_L - r.L[CATH] >= p.seat_pull_max_mm:
            self._go("wire", "out", r.L[CATH])
            return self._wire(r)
        v_c = 0.0 if deep else -p.v_seat
        return self._keep_d(r, v_c, -p.behind_mm), 0.0, v_c, 0.0

    # wire: out, aimed, bursts through the trunk's end, up the carotid -------------------------
    def _reseat(self, r: Readout) -> float:
        """Catheter speed while the wire works: a slow pull if its tip is slipping out of the
        trunk; never a push (pushing the trailing catheter only moves it along the wire, and
        pushing both bows the shared shaft)."""
        p = self.p
        if r.cath_in_bct and r.depth_bct_cath < p.reseat_mm:
            return -p.v_reseat
        return 0.0

    def _lost(self, r: Readout) -> bool:
        """Catheter tip well out of the trunk while the wire works (``lost_mm``; during the bursts
        ``fork_lost_mm``: they dip it to -1..-5 mm and the wire's breakthrough, or the wire backing
        off, pulls it back along the chain, so not every dip counts)."""
        bursting = self.stage == "wire" and self.substage in ("aim", "fork", "backoff", "rsaback")
        limit = self.p.fork_lost_mm if bursting else self.p.lost_mm
        return (not r.cath_in_bct and r.vessel_cath != "rcca") and r.depth_bct_cath < limit

    def _recover(self, r: Readout):
        """The catheter slipped out of the trunk: wire back behind the hook, then turn, pull and seat
        again (counts as a try)."""
        self.tries += 1
        self.rewires += 1
        self._anchor = np.nan
        self._go("place", "rewire", r.L[CATH])
        return self._place(r)

    def _ica_task(self, r: Readout) -> bool:
        """The target lies past the carotid's ICA/ECA split (the ``two_devices_ica`` config)."""
        an = self.anatomy
        return bool(np.isfinite(an.split_mm) and np.isfinite(r.target_depth) and r.target_depth > an.split_mm)

    def _turn_to(self, ang: float) -> float:
        return float(np.clip(self.p.aim_k * ang, -self.p.w_wire_max, self.p.w_wire_max))

    def _wire(self, r: Readout):
        p, an = self.p, self.anatomy
        L = r.L[CATH]
        if self.substage != "out" and self._lost(r):
            return self._recover(r)
        if self.substage == "out":
            if r.d >= p.out_lead_mm - 1.0:
                self._fork_tries = 0
                self._go("wire", "aim" if p.aim else "fork", L)
            else:
                return min(p.v_wire_out, max(p.k_d * (p.out_lead_mm - r.d), 4.0)), 0.0, self._reseat(r), 0.0
        if self.substage == "rsaback":
            # into the subclavian: back into the trunk (turning if there is no aiming), then again;
            # a tip that swings into the carotid on the way back is taken (seed 1005 did, 15-30 mm)
            if r.wire_branch == "rcca" and r.wire_branch_depth > p.in_rcca_mm:
                self._go("wire", "up", L)
                return self._wire(r)
            back = r.wire_branch == "bct" and r.depth_bct_wire < p.rsa_retreat_to_mm
            if not back and self._elapsed() < 25:
                w = 0.0 if p.aim else p.rsa_retreat_turn * self._spin
                return -p.v_backoff, w, self._reseat(r), 0.0
            self._go("wire", "aim" if p.aim else "fork", L)
        if self.substage == "aim":
            ang = aim_angle(r.wire, an.lat_trunk)
            self.last["aim_deg"] = np.nan if ang is None else float(np.degrees(ang))
            aimed = ang is not None and abs(ang) < np.radians(p.aim_tol_deg)
            if ang is None or aimed or self._elapsed() >= p.aim_steps:
                self.log.append((f"aim:{'ok' if aimed else 'skip'}", self.steps))
                self._go("wire", "fork", L)
            else:
                return 0.0, self._turn_to(ang), self._reseat(r), 0.0
        if self.substage == "backoff":
            if self._elapsed() < p.fork_backoff_steps:
                return -p.v_backoff, 0.0, self._reseat(r), 0.0
            self._go("wire", "fork", L)
        if self.substage == "fork":
            if self._fork_since != self._since:          # a new burst: fresh history
                self._fork_since, self._fork_hist, self._fork_dbc0 = self._since, [], r.depth_bct_cath
            depth = r.wire_branch_depth
            if r.wire_branch == "rcca" and depth > p.in_rcca_mm:
                self._go("wire", "up", L)
                return self._wire(r)
            if r.wire_branch in ("bct", "aorta"):
                progress = r.depth_bct_wire
            else:
                progress = an.bct_beyond_mouth + (depth if np.isfinite(depth) else 0.0)
            hist = self._fork_hist + [progress]
            self._fork_hist = hist[-(p.fork_stall_steps + 1):]
            n = p.fork_stall_steps
            buckle = len(hist) >= 3 and hist[-1] < hist[-3] - p.fork_retreat_mm
            stall = len(hist) > n and max(hist[-n:]) - hist[-n - 1] < 1.0
            cath_drop = (self._fork_dbc0 - r.depth_bct_cath) > p.fork_cath_drop_mm
            rsa = r.wire_branch == "rsa" and depth > p.rsa_back_mm
            why = ("rsa" if rsa else "buckle" if buckle else "stall" if stall else "cath_drop" if cath_drop
                   else "blocked" if self._wire_blocked >= p.blocked_steps else "")
            if why:
                self._wire_blocked = 0
                self._fork_tries += 1
                self._spin = -self._spin
                self.log.append((f"fork:{why}", self.steps))
                if self._fork_tries > p.fork_max_tries:
                    return self._recover(r)
                if why == "rsa":
                    self._go("wire", "rsaback", L)
                    return -p.v_backoff, (0.0 if p.aim else p.rsa_retreat_turn * self._spin), self._reseat(r), 0.0
                self._go("wire", "backoff", L)
                return -p.v_backoff, 0.0, self._reseat(r), 0.0
            return p.v_fork, p.w_fork * self._spin, self._reseat(r), 0.0
        if self.substage == "up":
            ica = self._ica_task(r)
            goal = p.up_goal_mm if ica else r.target_depth + p.wire_extra_mm
            depth = r.wire_in_rcca
            if depth >= goal:
                self._go("ica" if ica else "follow", "", L)
                return self._policy(r)
            back_at_fork = (r.wire_branch == "rsa" and r.wire_branch_depth > 3.0) or \
                (r.wire_branch == "bct" and r.depth_bct_wire < an.bct_beyond_mouth - 2.0) or \
                r.wire_branch == "aorta"
            if back_at_fork:
                self._go("wire", "aim" if p.aim else "fork", L)
                return self._wire(r)
            return p.v_rcca, 0.0, self._reseat(r), 0.0
        return 0.0, 0.0, 0.0, 0.0

    # ica: on up the carotid, aimed through the ICA/ECA split, anchored past it -------------------
    def _ica(self, r: Readout):
        p, an = self.p, self.anatomy
        L = r.L[CATH]
        label, side, s = r.wire_carotid, r.wire_carotid_depth, r.wire_chain
        if self._lost(r):
            return self._recover(r)
        anchor = max(an.split_mm + p.ica_anchor_mm, r.target_depth)
        if self.substage == "backoff":
            target_s = {"eca": an.split_mm - p.eca_back_to_mm, "down": an.down_mm - 8.0}.get(
                self._ica_why, self._ica_back_to)
            if s > target_s and self._elapsed() < 30 and label in ("cca", "ica", "eca", "down"):
                return -p.v_ica_backoff, 0.0, self._reseat(r), 0.0
            self._go("ica", "", L)
        if label == "ica" and s >= anchor:
            self._go("follow", "", L)
            return self._follow(r)
        if label not in ("cca", "ica", "eca", "down"):      # fell back out of the carotid
            self._go("wire", "aim" if p.aim else "fork", L)
            return self._wire(r)
        if self._elapsed() > p.ica_timeout:
            self.log.append(("ica:timeout", self.steps))
            return self._recover(r)
        why = ""
        if label == "eca" and side > p.eca_back_mm:
            why = "eca"
        elif label == "down" and side > p.v7_back_mm:
            why = "down"
        if self._ica_pushed:                                 # steps spent aiming do not count as a stall
            self._ica_hist = (self._ica_hist + [s])[-(p.ica_stall_steps + 1):]
        hist = self._ica_hist
        if not why and ((len(hist) > p.ica_stall_steps and max(hist) - hist[0] < 2.0)
                        or self._wire_blocked >= p.blocked_steps):
            why = "stall"
            self._ica_back_to = s - 8.0
        if why:
            self._wire_blocked = 0
            self._ica_hist = []
            self._spin = -self._spin
            self._ica_why = why
            self.log.append((f"ica:{why}", self.steps))
            self._go("ica", "backoff", L)
            return -p.v_ica_backoff, 0.0, self._reseat(r), 0.0
        v_c = self._reseat(r)
        if p.aim:
            target = None
            if an.lat_down_away is not None and an.down_mm - 12.0 <= s < an.down_mm + 4.0:
                target = an.lat_down_away
            elif an.lat_ica is not None and an.split_mm - p.aim_from_mm <= s < an.split_mm + 4.0:
                target = an.lat_ica
            if target is not None:
                ang = aim_angle(r.wire, target)
                self.last["aim_deg"] = np.nan if ang is None else float(np.degrees(ang))
                if ang is not None:
                    aligned = abs(ang) < np.radians(p.aim_tol_deg)
                    self._ica_aiming = 0 if aligned else self._ica_aiming + 1
                    v = p.aim_v if (aligned or self._ica_aiming > p.aim_steps) else 0.0
                    self._ica_pushed = v > 0.0
                    return v, self._turn_to(ang), v_c, 0.0
        self._ica_pushed = True
        return p.v_ica, 0.0, v_c, 0.0

    # follow: catheter over the anchored wire -------------------------------------------------
    def _cath_depth(self, r: Readout) -> float:
        """Catheter tip along trunk + carotid chain, mm from the carotid's opening (negative in the
        trunk, -inf elsewhere)."""
        if r.cath_carotid in ("cca", "ica", "eca", "down"):
            return r.cath_chain
        if r.vessel_cath == "bct":
            return r.depth_bct_cath - self.anatomy.bct_beyond_mouth
        return -np.inf

    def _cath_goal(self, r: Readout) -> Tuple[float, float]:
        """(the catheter's goal depth along the carotid chain, the least depth that counts as in)."""
        p, an = self.p, self.anatomy
        cath_min = (an.split_mm + p.cath_ica_mm) if self._ica_task(r) else p.cath_min_mm
        goal = max(cath_min + 2.0, min(r.target_depth - p.cath_goal_offset_mm, r.target_depth - p.d_lo - 2.0))
        return float(goal), float(cath_min)

    def _wire_slipped(self, r: Readout) -> bool:
        """The wire's anchor is gone: (ICA) out of the ICA or less than 5 mm past the split,
        (carotid) less than ``in_rcca_mm`` into the carotid."""
        if self._ica_task(r):
            return not (r.wire_carotid == "ica" and r.wire_chain >= self.anatomy.split_mm + 5.0)
        return r.wire_in_rcca < self.p.in_rcca_mm

    def _follow(self, r: Readout):
        """Catheter over the wire: it moves 1:1 along the wire as pushed while the wire's tip holds
        (its base is not fed: a wedged tip would only bow the chain); it catches at the trunk's end
        and at the split, where push-pause pulses get it on; a slip is braked."""
        p, an = self.p, self.anatomy
        L = r.L[CATH]
        if self._lost(r) and r.vessel_cath != "rcca":
            return self._recover(r)
        depth_c = self._cath_depth(r)
        prev, self._prev_depth_c = self._prev_depth_c, depth_c
        moved = depth_c - prev if np.isfinite(depth_c) and np.isfinite(prev) else 0.0
        sw = r.wire_chain if r.wire_carotid in ("cca", "ica") else -np.inf
        if not np.isfinite(self._anchor):                 # entering (or re-entering) the follow
            self._anchor = sw if np.isfinite(sw) else r.target_depth
            self._f_hist, self._f_mode, self._f_mode_since = [], "steady", self.steps
        goal_c, _ = self._cath_goal(r)
        if self._wire_slipped(r):
            self.log.append(("follow:reanchor", self.steps))
            self._anchor = np.nan
            if self._ica_task(r):
                self._go("ica", "", L)
                return self._ica(r)
            self._go("wire", "aim" if p.aim else "fork", L)
            return self._wire(r)
        if depth_c >= goal_c - 2.0 or self._elapsed() > p.follow_timeout:
            self._go("trim", "", L)
            return self._trim(r)
        if self.substage == "brake":
            if self._elapsed() < p.brake_min_steps or (self._elapsed() < p.brake_steps and moved > 0.3):
                return 0.0, 0.0, -p.v_brake, 0.0
            self._go("follow", "", L)
        in_carotid = r.cath_carotid in ("cca", "ica")
        near_end = r.cath_in_bct and r.depth_bct_cath >= p.slow_from_mm
        near_split = self._ica_task(r) and r.cath_carotid == "cca" and \
            r.cath_chain >= an.split_mm - p.slow_below_split_mm
        if moved > p.slip_mm and (near_end or in_carotid):
            self._go("follow", "brake", L)
            return 0.0, 0.0, -p.v_brake, 0.0
        hist = (self._f_hist + [depth_c])[-(p.follow_stall_steps + 1):]
        self._f_hist = hist
        stalled = len(hist) > p.follow_stall_steps and np.isfinite(hist[0]) and np.isfinite(hist[-1]) \
            and max(hist) - hist[0] < 1.0
        if stalled and self._f_mode == "steady":
            self._f_mode, self._f_mode_since, self._f_hist = "pulse", self.steps, []
            self.log.append(("follow:pulse", self.steps))
        elif self._f_mode == "pulse" and len(hist) > 3 and hist[-1] - hist[0] > 3.0:
            self._f_mode = "steady"
            self.log.append(("follow:steady", self.steps))
        slow = near_end or near_split or r.cath_carotid == "ica" or (in_carotid and depth_c >= goal_c - 10.0)
        v_c = p.v_follow_slow if slow else p.v_follow
        if self._f_mode == "pulse":
            k = (self.steps - self._f_mode_since) % (p.pulse_push_steps + p.pulse_pause_steps)
            v_c = p.v_follow if k < p.pulse_push_steps else 0.0
        # the wire's base is left alone: feeding it while its tip is wedged only bows the chain and
        # pulls the catheter back (seeds 1000/1001 of the first acceptance run); a tip that slips
        # below the split is sent up again by the ica stage (``_wire_slipped``)
        v_w = 0.0
        if r.d < p.d_lo:
            v_w = v_c + p.k_d * (p.d_lo - r.d)
        elif r.d > p.d_hi and np.isfinite(sw) and sw > self._anchor + 5.0:
            v_w = -3.0
        return v_w, 0.0, v_c, 0.0

    # trim: wire to the target, catheter to its goal, lead in range, held ----------------------
    def _trim(self, r: Readout):
        p = self.p
        L = r.L[CATH]
        if self._lost(r) and r.vessel_cath != "rcca":
            return self._recover(r)
        depth_c = self._cath_depth(r)
        goal_c, cath_min = self._cath_goal(r)
        depth_w = r.wire_chain if r.wire_carotid in ("cca", "ica") else -np.inf
        v_w = 0.0
        if np.isfinite(depth_w):
            v_w = float(np.clip(p.k_trim_wire * (r.target_depth - depth_w), -p.v_trim_wire, p.v_trim_wire))
            if abs(r.target_depth - depth_w) < p.wire_near_mm:
                v_w = float(np.clip(v_w, -p.v_trim_wire_near, p.v_trim_wire_near))   # gently: it drags the catheter
            if v_w > 0.0 and not (np.isfinite(depth_c) and depth_c >= goal_c - 10.0):
                v_w = 0.0                                   # catheter first: pushing a wedged wire bows the chain
        elif r.vessel_wire != "aorta":
            v_w = -p.v_trim_wire / 2.0                  # up a wrong branch: come back
        ref = self._referee
        cath_in = float(ref.depth_rcca_c) if ref is not None else float(r.cath_chain)
        wire_dist = float(ref.dist_to_target_w) if ref is not None else float(np.linalg.norm(r.wire_tip - r.target))
        v_c = 0.0
        if r.cath_carotid in ("cca", "ica") and np.isfinite(depth_c):
            if depth_c < cath_min + 1.0 or depth_c < goal_c - 2.0:
                v_c = p.v_trim_cath[1]
            elif depth_c > goal_c + 2.0:
                v_c = float(max(p.k_trim_cath * (goal_c - depth_c), p.v_trim_cath[0]))
        elif r.cath_in_bct:
            v_c = p.v_follow_slow                       # not in the carotid yet: push on slowly
        if r.d < p.d_lo:
            v_w = max(v_w, v_c + p.k_d * (p.d_lo - r.d) * 0.5)
        elif r.d > p.d_hi:
            v_c = max(v_c, 2.0)
        m = p.success_margin_mm
        ok = (wire_dist <= p.hand_tol_mm - m and cath_in >= cath_min + m
              and p.d_range[0] + m <= r.d <= p.d_range[1] - m)
        # the referee's own test, no margin: once it holds, anything we do can only break it
        holding = (wire_dist <= p.hand_tol_mm and cath_in >= cath_min
                   and p.d_range[0] <= r.d <= p.d_range[1]) or (ref is not None and ref.platform_steps > 0)
        self._count = self._count + 1 if (ok or holding) else 0
        if self._count >= p.success_hold:
            self.success = True
            self._go("done", "", L)
            return 0.0, 0.0, 0.0, 0.0
        if ok or holding:   # hold still while the criterion is being held
            return 0.0, 0.0, 0.0, 0.0
        return v_w, 0.0, v_c, 0.0

    # -- suggestion for P1 (residual box / observation) --------------------------------------
    def nominal_for_phase(self, env, phase: Optional[int] = None) -> np.ndarray:
        """Stateless suggestion for the hook-setting phase P1: wire back behind the hook, the hook's
        base to the trunk's mouth + first offset (hook down while it moves), then the hook turned up.
        Zeros in every other phase. Does not advance the stage machine. Called as
        ``(env, phase)`` or, by ``NominalAdvisor``, as ``(phase_state)`` (then the intervention given
        at construction is read)."""
        if phase is None:
            state, env = env, None
            phase = int(getattr(state, "phase", -1))
        if int(phase) != 1:
            return np.zeros(4, np.float32)
        env = self._env_or_given(env)
        if self.anatomy is None or self._intervention is None:
            self.reset(env)
        p, m = self.p, self.anatomy.bct_mouth
        r = read(env, self.anatomy, self.hook_len)
        if r.d > -p.behind_mm:
            v_w, w_w, v_c, w_c = -p.v_retract, 0.0, -p.v_cath_retract, 0.0
        else:
            err = m.s_parent + p.offsets_mm[0] - r.s_base
            if abs(err) > 3.0:
                v_c = float(np.clip(p.k_place * err, -p.v_place, p.v_place))
                w_c = float(np.clip(p.k_rot * r.phi_down, -p.w_cath_max, p.w_cath_max))
            else:
                v_c = 0.0
                w_c = float(np.clip(p.k_rot * wrap(r.phi_up - np.radians(p.aims_deg[0])), -p.w_cath_max, p.w_cath_max))
            v_w, w_w = self._keep_d(r, v_c, -p.behind_mm), 0.0
        saved = getattr(self, "last_physical", None)
        v_w, v_c = self._cross(r, v_w, v_c)
        out = self._normalize(v_w, w_w, v_c, w_c)
        if saved is not None:
            self.last_physical = saved
        return out
