# The two-device scripted controller as it was for the first training round (commit e908bee,
# 2026-10-31; recorded runs/demos_round1/ on configs/two_devices_round1.yaml with stEVE's default,
# heavy device masses). Kept verbatim next to script.py, which was retuned on 2026-11-06 for the
# real-mass setting; the numbers below were measured on the heavy, stiff devices.
"""Scripted controller for the two-device task (guidewire + the lab's hook catheter): the scripted
episodes the training starts from (demonstrations), the script's suggestion for the hook-setting
phase, and a reference for evaluation (todo_list/two_devices_training.md, WP-B).

Origin: the lab's ``reference/surgery-all-in-one/19-phantom-sofa/arch_catheter.py`` (``Mouth``,
``Stage``, ``Params``, ``Procedure``, ``split``), carried over to our env and extended past the lab's
end point (catheter seated, wire up the vessel) to the task's: wire tip at the target 35-55 mm into
the right common carotid, catheter >= 15 mm into it, wire ahead by 15-45 mm. The geometry is our
own numpy (no ``tree.py`` / ``vendor/pnav``). It reads 3D truth from the simulator every step: both
instruments' polylines out of ``simulation.dof_positions`` (vessel frame, tip first; the points
``fluoroscopy.instrument_trackings3d`` rotates into the image frame; the shorter instrument's tip is
counted back from the leading one), ``inserted_lengths``, ``rotations``, and the env's referee
(``env.monitors["phase_detector"]``) for the success test when there is one.

Stages (``stage`` / ``substage``) and the referee's phases they pass through:

=========  ==================================================================  ==============
stage      what                                                                phases
=========  ==================================================================  ==============
approach   both round the arch, wire ``lead_mm`` (40) ahead, until the catheter  P0 -> P1
           tip is ``past_mouth_mm`` (33) past the trunk's mouth along the aorta
place      wire back behind the hook (d = -37); hook down; its base pulled to   P1 -> P2
           the mouth + offset (4 mm; 0, 8, -4, 12 on later tries), reached from
           beyond (the lab: placed by pushing, the shaft springs forward)
turn       hook turned up to the trunk's axis, standing still; if it does not   P2
           settle within 40 steps: hook down, place again (next offset / aim)
pull       3 mm/s pull until the tip slides into the trunk (>= 2 mm)            P2
seat       pull on until 20 mm deep                                            P2 -> P3
wire       wire out 16 mm ahead; both up the trunk (32 steps or the catheter     P3 -> P4
           26 mm in); the wire probes the trunk's end turning slowly
           (0.6 rad/s), backs off from the right subclavian or when the shell
           holds its push, and goes up the carotid to target - 5
follow     catheter over the wire: 8 mm/s in the trunk, 3 mm/s near its end, a  P4 -> success
           hard pull while it slips into the carotid, then trim: wire to the
           target, catheter >= 15 mm in, wire 15-45 mm ahead, held 8 steps
=========  ==================================================================  ==============

A pull that finds nothing or goes up another vessel starts the next try; a catheter that comes out
of the trunk while the wire works sends the wire back behind the hook and repeats turn / pull /
seat (a try).

What the simulator does that shaped it (measured 2026-10-31 in this env; no click, default masses):

* Pushing a seated hook bows its shaft in the ascending aorta and pulls the tip *out*; pulling
  drives the tip deeper (a reverse curve). So the catheter is seated by pulling and pushed only with
  the wire well ahead of it (16 mm: the wire's stiff part then lies in the hook; 8 mm lost the
  catheter twice as often). Pulling it later, with the wire through the hook, only pulls it out.
* The wire pushed against the trunk's end levers the hook out of the trunk; the J tip goes into the
  carotid only after enough push and some turning (aiming the J by its measured bend did not work:
  the bend hardly follows the turning there). The carotid and the subclavian leave through a common
  bulb (radius ~5.6 mm for ~8 mm): a tip there is counted in neither.
* At the trunk's end the pushed catheter catches, the push is stored in the bowed shaft, and then
  the tip slips 15-35 mm into the carotid within a few steps and drags the wire on by 10-30 mm;
  pulling it back is slow (50 mm of pull moved the tip 2-5 mm). The wire therefore waits 5 mm short
  of the target and the push slows down near the trunk's end. Pushing the catheter while the wire
  goes up the carotid bowed it out of the trunk, so it waits.
* How the hook settles in the turn decides much of the rest: where the turn timed out, pulling it in
  anyway seated it badly (slow seat, levered out later); placing it again does better.
* The safety shell (``actions.TwoDeviceGuarded``) holds a push back when that tip's contact force
  is over its limit, and lets the two tips cross the same-length band only by speeding up the device
  moving that way: the controller holds the catheter still for the crossing step and lets the wire
  jump (``_cross``), and backs the wire off when the shell keeps holding its push back.

Result (``configs/two_devices_round1.yaml`` with MaxSteps 600 and the wire's force limit 1.0 N, one
run per seed): 14/20 reached the referee's success on seeds 1000-1019 (the recorded demonstrations,
361-516 steps, median 434) and 16-17/20 on seeds 0-19. The failures: the wire stuck at the trunk's end
(it goes into the bulb and the subclavian side again and again, 5/20) and one trim that never met the
target. Runs repeat closely but not always bit for bit (SOFA's collision detection runs on two
threads): the same seeds gave 15/20 and 14/20 in two runs.
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
    ``rcca_rest``, arc length 0 at the trunk's end = the carotid's opening)."""

    def __init__(self, vessel_tree) -> None:
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
    """Speeds in mm/s and rad/s, lengths in mm, times in control steps."""

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
    # turn: the hook up to the trunk's axis
    k_rot: float = 1.0
    aims_deg: Tuple[float, ...] = (0.0, -20.0, 20.0, -35.0, 35.0)
    turn_tol_deg: float = 15.0
    turn_slow_deg: float = 60.0     # (slowing the turn near the aim, to stop the hook whipping past
    w_turn_near: float = float(np.pi / 2)   # it, made the seat worse: 4/20 against 12/20)
    turn_hold: int = 3
    turn_timeout: int = 40
    retry_on_turn_timeout: bool = True    # turn timed out: hook down and place again (next offset/aim);
                                    # pulling it in off its aim got it levered out later (15/20 against 13/20)
    adapt_aim_deg: float = 30.0     # turn timed out: pull aiming where the hook settled, within +-this
                                    # (13/20 against 12/20 fixed; 20 deg: 9/20; a fixed -15: 12/20)
    # pull / seat: a slow pull slides the hook's tip into the trunk
    v_pull: float = 3.0
    pull_max_mm: float = 18.0
    engage_mm: float = 2.0
    wrong_mm: float = 3.0
    v_free: float = 3.5             # under the shell's 3.75 mm/s cap for a catheter pushed in a branch ahead of the wire
    lower_timeout: int = 25
    v_seat: float = 3.0
    seat_mm: float = 20.0           # pulled this deep before the wire goes out (12 lost the catheter in 3/10)
    seat_hold: int = 2
    seat_pull_max_mm: float = 25.0
    # wire: out of the catheter, both up the trunk, the wire probes the fork, into the carotid
    out_lead_mm: float = 16.0       # wire ahead of the catheter while both climb the trunk (its stiff
                                    # part then straightens the hook: 8 lost the catheter twice as often)
    in_rcca_mm: float = 6.0         # wire this deep in the carotid (fork resolved): it is in
    v_wire_out: float = 30.0
    v_climb: float = 8.0
    climb_depth_mm: float = 26.0
    climb_timeout: int = 32         # (20, or a slower climb, lost the catheter far more often)
    reseat_mm: float = 8.0          # catheter tip shallower than this in the trunk: pull to re-seat
    v_reseat: float = 3.0
    v_trunk: float = 8.0
    park_mm: float = 30.0
    v_fork: float = 6.0
    w_fork: float = 0.6
    fork_lead_mm: float = 20.0
    rsa_back_mm: float = 7.0        # wire this deep in the right subclavian: back off (the referee counts 8)
    backoff_steps: int = 5
    v_backoff: float = 15.0
    v_rcca: float = 10.0
    up_slow_mm: float = 0.0         # the wire goes up the carotid at v_fork until this deep (15: no better)
    wire_extra_mm: float = -5.0     # wire tip goal while the catheter follows: target + this (it gets dragged on)
    cath_climb_after_mm: float = float("inf")   # wire this deep in the carotid: the catheter climbs the trunk
                                    # while the wire goes up (off: pushing it then bowed it out of the trunk)
    wire_max_mm: float = 58.0       # never past this in the carotid (its first fork is 64 mm in)
    stall_steps: int = 15           # wire pushed this long without its tip gaining 2 mm: back off
    lost_mm: float = -2.0           # catheter tip this far out of the trunk while the wire works: lost
    fast_recover: bool = True       # lost near the mouth: wire back, turn up and pull again (no new place)
    fast_recover_mm: float = 15.0
    # follow: catheter over the wire into the carotid, wire kept near the target; the pushed
    # catheter coasts on after the push stops (its shaft was compressed), so the push stops early
    v_follow: float = 8.0
    v_follow_slow: float = 3.0
    slow_from_mm: float = 34.0      # catheter tip this deep in the trunk: slow push towards its end
    follow_wire_share: float = 0.5
    k_wire: float = 1.0
    follow_wire_over_mm: float = 3.0
    follow_wire_cap: Tuple[float, float] = (-6.0, 15.0)
    slip_mm: float = 1.5            # catheter tip moving this much in one step on a slow push: it slips
    v_brake: float = 15.0           # pull this fast while it slips
    brake_min_steps: int = 3
    brake_steps: int = 8
    brake_wire_d_mm: float = 25.0   # while braking, also pull the wire back if it leads by more than this
    cath_goal_offset_mm: float = 22.0   # catheter goal = target depth - this, within the next two
    cath_goal_range: Tuple[float, float] = (16.5, 30.0)
    k_trim_cath: float = 1.0
    v_trim_cath: Tuple[float, float] = (-15.0, 2.0)
    k_trim_wire: float = 1.5
    v_trim_wire: float = 10.0
    wire_near_mm: float = 6.0       # wire tip this close to the target depth: trim it slowly
    v_trim_wire_near: float = 3.0
    cath_push_below_mm: float = 12.0  # trim pushes the catheter only below this (it coasts on to >= 15)
    d_lo: float = 17.0
    d_hi: float = 43.0
    success_hold: int = 8
    success_margin_mm: float = 0.5  # the hand-made success test keeps this much inside the referee's limits
    hand_tol_mm: float = 5.0
    cath_min_mm: float = 15.0
    d_range: Tuple[float, float] = (15.0, 45.0)


STAGES = ("approach", "place", "turn", "pull", "seat", "wire", "follow", "done")


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
        self.success = False
        self.log: List[Tuple[str, int]] = [("approach", 0)]
        self.last: Dict[str, float] = {}
        self._since = 0
        self._from_L = 0.0
        self._over = False
        self._hold = 0
        self._spin = 1.0
        self._count = 0
        self._redirect_from = 0.0
        self._wire_hist: List[float] = []
        self._prev_depth_c = np.nan
        self._aim_override: Optional[float] = None
        self._wire_blocked = 0
        self._referee = None
        self.last_physical = np.zeros(4)
        self.readout: Optional[Readout] = None

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
        if stage == "follow":
            return self._follow(r)
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

    # wire: out, both up the trunk, the wire through the fork into the carotid ------------------
    def _reseat(self, r: Readout) -> float:
        """Catheter speed while the wire works: a slow pull if its tip is slipping out of the trunk
        (pulling the hook drives its tip in; pushing it would bow the shaft and pull the tip out)."""
        p = self.p
        if r.cath_in_bct and r.depth_bct_cath < p.reseat_mm:
            return -p.v_reseat
        return 0.0

    def _lost(self, r: Readout) -> bool:
        """Catheter tip out of the trunk again while the wire works and before the wire is anchored in
        the carotid (once it is, the catheter is brought back along it by the follow stage)."""
        out = (not r.cath_in_bct and r.vessel_cath != "rcca") and r.depth_bct_cath < self.p.lost_mm
        return out and r.wire_in_rcca < self.p.cath_climb_after_mm

    def _recover(self, r: Readout):
        """The catheter slipped out of the trunk: wire back behind the hook, then turn, pull and seat
        again (counts as a try)."""
        self.tries += 1
        self._go("place", "rewire", r.L[CATH])
        return self._place(r)

    def _wire(self, r: Readout):
        p = self.p
        if self.substage not in ("out",) and self._lost(r):
            return self._recover(r)
        if self.substage == "out":
            if r.d >= p.out_lead_mm - 1.0:
                self._go("wire", "climb", r.L[CATH])
            else:
                return min(p.v_wire_out, max(p.k_d * (p.out_lead_mm - r.d), 4.0)), 0.0, self._reseat(r), 0.0
        if self.substage == "climb":
            done = ((r.cath_in_bct and r.depth_bct_cath >= p.climb_depth_mm) or r.vessel_cath == "rcca"
                    or r.wire_branch in ("rcca", "rsa") or self._elapsed() > p.climb_timeout)
            if done:
                self._go("wire", "fork", r.L[CATH])
            else:
                v_c = p.v_climb if (r.cath_in_bct and r.depth_bct_cath >= p.reseat_mm) else -p.v_reseat
                return self._keep_d(r, v_c, p.out_lead_mm), 0.0, v_c, 0.0
        if self.substage == "backoff":
            if self._elapsed() < p.backoff_steps:
                return -p.v_backoff, 0.0, self._reseat(r), 0.0
            self._go("wire", "fork", r.L[CATH])
        if self.substage == "fork":
            if self._elapsed() == 0:
                self._wire_hist = []
            depth = r.wire_branch_depth
            if r.wire_branch == "rcca" and depth > p.in_rcca_mm:
                self._go("wire", "up", r.L[CATH])
            elif r.wire_branch == "rsa" and depth > p.rsa_back_mm:
                self._spin = -self._spin
                self._go("wire", "backoff", r.L[CATH])
                return -p.v_backoff, 0.0, self._reseat(r), 0.0
            else:
                progress = r.depth_bct_wire if r.wire_branch == "bct" else 0.0
                hist = getattr(self, "_wire_hist", [])
                hist = (hist + [progress])[-(p.stall_steps + 1):]
                self._wire_hist = hist
                if (len(hist) > p.stall_steps and max(hist) - hist[0] < 2.0 and r.d > p.fork_lead_mm) or \
                        self._wire_blocked >= p.blocked_steps:
                    self._wire_blocked = 0
                    self._wire_hist = []
                    self._spin = -self._spin
                    self._go("wire", "backoff", r.L[CATH])
                    return -p.v_backoff, 0.0, self._reseat(r), 0.0
                if r.wire_branch not in ("bct", "rcca", "rsa"):
                    return p.v_trunk, 0.0, self._reseat(r), 0.0     # still on its way out
                if r.depth_bct_wire < p.park_mm or r.d < p.fork_lead_mm:
                    return p.v_trunk, 0.0, self._reseat(r), 0.0
                return p.v_fork, p.w_fork * self._spin, self._reseat(r), 0.0
        if self.substage == "up":
            goal = min(r.target_depth + p.wire_extra_mm, p.wire_max_mm)
            depth = r.wire_in_rcca
            if depth >= goal:
                self._go("follow", "", r.L[CATH])
                return self._follow(r)
            back_at_fork = (r.wire_branch == "rsa" and r.wire_branch_depth > 3.0) or \
                (r.wire_branch == "bct" and r.depth_bct_wire < self.anatomy.bct_beyond_mouth - 2.0) or \
                r.wire_branch not in ("rcca", "rsa", "bct")
            if back_at_fork:                  # slipped back to the fork (an unresolved tip in the
                self._go("wire", "fork", r.L[CATH])   # bulb where both start is not)
                return p.v_fork, 0.0, self._reseat(r), 0.0
            if not np.isfinite(depth):
                depth = 0.0
            v_c = self._reseat(r)
            if depth >= p.cath_climb_after_mm and r.cath_in_bct and r.depth_bct_cath < p.slow_from_mm:
                v_c = p.v_follow              # the wire anchors in the carotid: catheter up the trunk meanwhile
            v_w = (p.v_rcca if depth >= p.up_slow_mm else p.v_fork) + max(v_c, 0.0) * p.follow_wire_share
            if r.d < p.d_lo:
                v_w = max(v_w, v_c + p.k_d * (p.d_lo - r.d))
            return v_w, 0.0, v_c, 0.0
        return 0.0, 0.0, 0.0, 0.0

    # follow: catheter over the wire into the carotid -----------------------------------------
    def _wire_to(self, r: Readout, goal: float, base: float, cap: float) -> float:
        depth = r.wire_in_rcca
        if not np.isfinite(depth):
            return base
        return float(np.clip(base + self.p.k_wire * (goal - depth), -cap, cap))

    def _cath_route_depth(self, r: Readout) -> float:
        """Catheter tip along trunk + carotid, mm from the carotid's opening (negative in the trunk)."""
        if r.vessel_cath == "rcca":
            return r.depth_rcca_cath
        if r.vessel_cath == "bct":
            return r.depth_bct_cath - self.anatomy.bct_beyond_mouth
        return -np.inf

    def _follow(self, r: Readout):
        """Catheter over the wire through the trunk into the carotid. In the trunk it moves as
        pushed; where the trunk ends it catches, the pushed shaft bows out in the aorta and then the
        tip slips 15-35 mm into the carotid within a few steps, dragging the wire on (measured). So:
        push fast in the trunk, slowly near its end, pull hard while it slips, then trim: the wire
        back to the target (that also drags the catheter back) and the catheter pulled towards its
        goal; never pushed again unless it is short of 15 mm."""
        p = self.p
        if self._lost(r):
            return self._recover(r)
        depth_c = self._cath_route_depth(r)
        prev = self._prev_depth_c
        self._prev_depth_c = depth_c
        moved = depth_c - prev if np.isfinite(depth_c) and np.isfinite(prev) else 0.0
        near_end = r.cath_in_bct and r.depth_bct_cath >= p.slow_from_mm
        if self.substage == "":
            slipping = (r.vessel_cath == "rcca" and depth_c > 2.0) or (moved > p.slip_mm and (near_end or r.vessel_cath == "rcca"))
            if slipping:
                self._go("follow", "brake", r.L[CATH])
            else:
                v_c = p.v_follow if (r.cath_in_bct and not near_end) else p.v_follow_slow
                goal_w = min(r.target_depth + p.wire_extra_mm, p.wire_max_mm)
                v_w = self._wire_to(r, goal_w, p.follow_wire_share * v_c, 15.0)
                v_w = float(np.clip(v_w, *p.follow_wire_cap))
                if r.d < p.d_lo:
                    v_w = max(v_w, v_c + p.k_d * (p.d_lo - r.d))
                if r.wire_in_rcca >= p.wire_max_mm:
                    v_w = min(v_w, 0.0)
                return v_w, 0.0, v_c, 0.0
        if self.substage == "brake":
            if self._elapsed() < p.brake_min_steps or (self._elapsed() < p.brake_steps and moved > 0.3):
                v_w = 0.0 if r.d >= p.d_lo else p.k_d * (p.d_lo - r.d)
                if r.d > p.brake_wire_d_mm:
                    v_w = -p.v_trim_wire
                return v_w, 0.0, -p.v_brake, 0.0
            self._go("follow", "trim", r.L[CATH])
        # trim: wire to the target, catheter towards its goal, wire 15-45 mm ahead; then hold 8 steps
        goal_c = float(np.clip(r.target_depth - p.cath_goal_offset_mm, *p.cath_goal_range))
        depth_w = r.wire_in_rcca
        v_w = 0.0
        if np.isfinite(depth_w):
            v_w = float(np.clip(p.k_trim_wire * (r.target_depth - depth_w), -p.v_trim_wire, p.v_trim_wire))
        elif r.vessel_wire != "aorta":
            v_w = -p.v_trim_wire / 2.0                  # past the carotid's first fork: come back
        ref = self._referee
        cath_in = float(ref.depth_rcca_c) if ref is not None else r.depth_rcca_cath
        wire_dist = float(ref.dist_to_target_w) if ref is not None else float(np.linalg.norm(r.wire_tip - r.target))
        if np.isfinite(depth_w) and abs(r.target_depth - depth_w) < p.wire_near_mm:
            v_w = float(np.clip(v_w, -p.v_trim_wire_near, p.v_trim_wire_near))   # gently: it drags the catheter
        v_c = 0.0
        if r.vessel_cath == "rcca" and np.isfinite(depth_c):
            if cath_in < p.cath_push_below_mm:
                v_c = p.v_trim_cath[1]                  # it coasts on a few mm after the push stops
            elif depth_c > goal_c + 2.0:
                v_c = float(max(p.k_trim_cath * (goal_c - depth_c), p.v_trim_cath[0]))
        elif r.cath_in_bct:
            v_c = p.v_follow_slow                       # not in the carotid yet: slow push, brake on a slip
            if moved > p.slip_mm:
                self._go("follow", "brake", r.L[CATH])
                return 0.0, 0.0, -p.v_brake, 0.0
        if r.d < p.d_lo:
            v_w = max(v_w, v_c + p.k_d * (p.d_lo - r.d) * 0.5)
        elif r.d > p.d_hi:
            v_c = max(v_c, 2.0)
        m = p.success_margin_mm
        ok = (wire_dist <= p.hand_tol_mm - m and cath_in >= p.cath_min_mm + m
              and p.d_range[0] + m <= r.d <= p.d_range[1] - m)
        # the referee's own test, no margin: once it holds, anything we do can only break it
        holding = (wire_dist <= p.hand_tol_mm and cath_in >= p.cath_min_mm
                   and p.d_range[0] <= r.d <= p.d_range[1]) or (ref is not None and ref.platform_steps > 0)
        self._count = self._count + 1 if (ok or holding) else 0
        if self._count >= p.success_hold:
            self.success = True
            self._go("done", "", r.L[CATH])
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
