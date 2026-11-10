"""The referee of the two-device task: which of the five phases the episode is in, what happened
this step, success and hard failure (todo_list/two_devices_training.md, sections 1-2).

``PhaseDetector`` is a scene monitor (``kind="monitor"``): the builder makes one, steps it right
after the simulator (before observations, rewards, terminals) and hands the same instance to every
part that asks for ``phase_detector``. It reads the 3D truth (both tips, the catheter's tip
direction, insertion lengths); its ``state`` is for the referee's clients (rewards, terminals,
truncation, labels, the safety shell, the script) and never for the policy's observation, except
the phase itself (``PhaseOneHot``).

Phases (``state.phase``) and how they change; every change needs its condition on two consecutive
steps (``consecutive_steps``), the conditions are checked in the order listed, the first that
holds is the candidate:

======  ===========================  =======================================================
phase   what                         leaves for
======  ===========================  =======================================================
P0 = 0  over the arch                P1: catheter tip past O by >= 30 mm along the aorta
P1 = 1  hook set                     P0: catheter tip back below O by > 30 mm;
                                     P2: d <= -35 (wire behind the hook) and hook up
                                     (catheter tip direction within 60 deg of +z)
P2 = 2  select the opening           P3: catheter tip >= 8 mm into the trunk;
                                     P0: catheter tip back below O by > 60 mm;
                                     P1: hook straightened (-10 < d < 0: the wire tip
                                     inside the catheter, near its tip, and not past
                                     the trunk's mouth)
P3 = 3  wire out                     P2: catheter out of the opening (< 4 mm into the trunk);
                                     P4: wire >= 15 mm into the right carotid and d >= 10
P4 = 4  catheter follows             P2: catheter out of the opening;
                                     P3: wire back to < 10 mm into the right carotid
======  ===========================  =======================================================

d = L_wire - L_cath (insertion lengths; positive: the wire leads). Depths are measured from the
vessel's opening (``route.py``): the trunk's is its mouth on the aortic wall, the right carotid's
the trunk's end.

Success (``state.success``): wire tip <= 5 mm from the target, catheter tip >= 15 mm into the right
carotid and 15 <= d <= 45, all three for 8 steps in a row. Hard failure (``state.hard_failure``):
either tip outside the vessel (more than ``outside_margin_mm`` beyond every centerline tube), or
both tips at the same insertion length (|d| < ``same_length_mm`` with both inserted). A step with
a simulation error (NaN, stEVE resets the instruments) changes nothing and is flagged
(``state.sim_error``). With the wall monitor in the config (two_device/wall.py, ``WallBreach``), a
centerline crossing the vessel mesh ends the episode first, as a dropped step, so the outside-the-
tubes hard failure is a backstop that normally never fires (user decision 2026-11-09).

The two progress rewards measure in P0 with the arch ruler (``PhaseState.progress_paths``): the
path left to the hook point H = O + 30 mm along the aorta (where P1 starts), plus the fixed path
from H on; going past H in P0 neither earns nor costs (P0 has to push the catheter past O, which
the plain path would charge). From P1 on they use the plain path along the tree.

Events (``state.events``, counted on the step they happen only):

  - ``wrong_vessel``: a tip >= 8 mm into the left carotid, left subclavian or right subclavian
    (again after it came back to < 4 mm);
  - ``dislodged``: the catheter left the opening (the change P3/P4 -> P2);
  - ``bare_advance``: the shell stopped a catheter push with its tip in a branch ahead of the wire
    (reported by ``TwoDeviceGuarded``);
  - ``wire_lead_arch``: the wire leads by > 100 mm with its tip in the aorta;
  - ``wire_past_fork``: the wire tip past the carotid's first fork (64 mm in).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from eve.info import Info

from ..registry import register
from .route import ROUTE_CHAIN, Route, arclen, unit

WIRE, CATH = 0, 1
N_PHASES = 5
PHASE_NAMES = ("arch", "hook", "select", "wire_out", "follow")
EVENT_NAMES = ("wrong_vessel", "dislodged", "bare_advance", "wire_lead_arch", "wire_past_fork")
WRONG_VESSELS = ("lcca", "lsa", "rsa")
DEFAULT_CUTOFF = 5          # phase >= cutoff truncates; 5 = never (only success ends it)


# --------------------------------------------------------------------------- devices ----
def instrument_polylines(dofs_tip_first: np.ndarray, inserted: Sequence[float]) -> List[np.ndarray]:
    """Each instrument's deployed part, tip first, out of the combined beam (tip first).

    The instruments are coaxial: one that is x mm shorter than the leading one ends x mm down the
    beam, counted back from the leading tip (as the lab's ``arch_catheter.split``; the node
    polyline can be a few mm shorter than the inserted length, so counting from the base would
    misplace the shorter tip). The first point is the interpolated tip, then the beam's nodes."""
    dofs = np.asarray(dofs_tip_first, dtype=np.float64)
    cum = arclen(dofs)
    lead = float(max(inserted))
    out = []
    for length in inserted:
        back = min(max(lead - float(length), 0.0), float(cum[-1]))
        tip = np.array([np.interp(back, cum, dofs[:, k]) for k in range(3)])
        rest = dofs[cum > back + 1e-9]
        out.append(np.vstack([tip[None], rest]))
    return out


def point_back(polyline: np.ndarray, distance_mm: float) -> np.ndarray:
    """The point ``distance_mm`` back from the first point along a polyline (clamped)."""
    cum = arclen(polyline)
    return np.array([np.interp(distance_mm, cum, polyline[:, k]) for k in range(3)])


@dataclass
class Devices:
    """Both instruments this step, vessel frame (3D truth)."""

    polylines: List[np.ndarray]          # tip first
    inserted: np.ndarray                 # [wire, catheter] mm
    rotations: np.ndarray                # rad

    @property
    def tips(self) -> List[np.ndarray]:
        return [p[0] for p in self.polylines]

    @property
    def d(self) -> float:
        return float(self.inserted[WIRE] - self.inserted[CATH])

    def tip_direction(self, instrument: int, back_mm: float = 3.0) -> np.ndarray:
        line = self.polylines[instrument]
        return unit(line[0] - point_back(line, back_mm))


_DEVICES_CACHE: "weakref.WeakKeyDictionary[Any, Tuple[Any, Any, Devices]]" = \
    weakref.WeakKeyDictionary()


def devices_of(simulation) -> Devices:
    """``Devices`` of the simulation's current step, computed once per step. stEVE's
    ``SofaBeamAdapter`` makes new ``dof_positions`` and ``inserted_lengths`` arrays every step, so
    those objects key the step (the cache holds them, so their ids cannot be reused meanwhile)."""
    dofs = simulation.dof_positions
    inserted = simulation.inserted_lengths
    try:
        cached = _DEVICES_CACHE.get(simulation)
    except TypeError:                     # not weak-referenceable: no cache
        cached = None
    if cached is not None and cached[0] is dofs and cached[1] is inserted:
        return cached[2]
    inserted_arr = np.asarray(inserted, dtype=np.float64).reshape(-1)
    devices = Devices(instrument_polylines(dofs, inserted_arr), inserted_arr,
                      np.asarray(simulation.rotations, dtype=np.float64).reshape(-1))
    try:
        _DEVICES_CACHE[simulation] = (dofs, inserted, devices)
    except TypeError:
        pass
    return devices


def target_in_vessel_frame(target, fluoroscopy) -> np.ndarray:
    """The target in vessel coordinates (stEVE keeps it in the tracking frame)."""
    if getattr(target, "coordinates_vessel", None) is not None:
        return np.asarray(target.coordinates_vessel, dtype=np.float64).reshape(3)
    from eve.util.coordtransform import tracking3d_to_vessel_cs
    return np.asarray(tracking3d_to_vessel_cs(np.asarray(target.coordinates3d, dtype=np.float64),
                                              fluoroscopy.image_rot_zx, fluoroscopy.image_center),
                      dtype=np.float64).reshape(3)


# ----------------------------------------------------------------------------- state ----
@dataclass
class PhaseState:
    """What the referee knows after a step. The first ten fields are the plan's interface
    (section 2); the rest is extra, for rewards, labels, the shell and logs."""

    phase: int = 0                                   # 0-4
    d: float = 0.0                                   # L_wire - L_cath, mm
    L_wire: float = 0.0
    L_cath: float = 0.0
    depth_bct_c: float = -np.inf                     # catheter tip, mm into the trunk (signed)
    depth_rcca_c: float = -np.inf                    # catheter tip, mm into the right carotid
    depth_rcca_w: float = -np.inf                    # wire tip, mm into the right carotid
    hook_up: bool = False
    events: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in EVENT_NAMES})
    steps_in_phase: int = 0
    # --- extra ---
    max_phase: int = 0
    step: int = 0                                    # steps since reset
    previous_phase: int = 0
    success: bool = False
    hard_failure: bool = False
    failure_reason: str = ""                         # "outside_wire" | "outside_cath" | "same_length"
    sim_error: bool = False
    platform_steps: int = 0                          # success conditions held this many steps
    wire_tip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cath_tip: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cath_tip_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hook_angle_deg: float = 180.0                    # catheter tip direction against +z
    branch_w: str = ""
    branch_c: str = ""
    clearance_w: float = 0.0                         # signed tube clearance of the tips, mm
    clearance_c: float = 0.0
    aorta_pos_c: float = -np.inf                     # catheter tip along the aorta from O, mm
    depth_bct_w: float = -np.inf
    s_w: float = 0.0                                 # route arc length of the tips
    s_c: float = 0.0
    path_to_target_w: float = np.inf
    path_to_target_c: float = np.inf
    path_to_F_c: float = np.inf
    dist_to_target_w: float = np.inf                 # straight line, mm
    depth_wrong: Dict[str, float] = field(default_factory=dict)   # max over tips, per wrong vessel
    event_totals: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in EVENT_NAMES})
    cutoff: int = DEFAULT_CUTOFF
    # the arch ruler (P0): the hook point H is O + 30 mm along the aorta, where P1 starts
    to_hook_w: float = 0.0                           # path left to H, 0 once past it, mm
    to_hook_c: float = 0.0
    hook_to_target: float = 0.0                      # fixed path from H to the target, mm
    hook_to_F: float = 0.0                           # fixed path from H to F, mm

    @property
    def leading_path(self) -> float:
        """Path to the target of the tip further along (the smaller of the two)."""
        return min(self.path_to_target_w, self.path_to_target_c)

    def progress_paths(self) -> Tuple[float, float, float]:
        """The rulers of the two progress rewards: (wire to target, catheter to target, catheter
        to F). In P0 the way to the hook point H plus the fixed way on from H (going past H in
        P0 neither earns nor costs); from P1 on the plain path along the tree. The two agree at H,
        where P1 starts."""
        if self.phase == 0:
            return (self.to_hook_w + self.hook_to_target, self.to_hook_c + self.hook_to_target,
                    self.to_hook_c + self.hook_to_F)
        return (self.path_to_target_w, self.path_to_target_c, self.path_to_F_c)

    def info(self) -> Dict[str, Any]:
        return {"phase": int(self.phase), "max_phase": int(self.max_phase),
                "events": dict(self.events), "is_success": bool(self.success),
                "hard_failure": bool(self.hard_failure), "failure_reason": self.failure_reason,
                "d": float(self.d), "cutoff": int(self.cutoff), "sim_error": bool(self.sim_error)}


@dataclass(frozen=True)
class Thresholds:
    """The plan's numbers (section 1), mm and degrees."""

    past_O: float = 30.0              # P0 -> P1
    hook_behind: float = -35.0        # P1 -> P2: d <= this
    hook_up_deg: float = 60.0
    back_P1: float = -30.0            # P1 -> P0: catheter along the aorta from O below this
    engaged: float = 8.0              # P2 -> P3
    back_P2: float = -60.0            # P2 -> P0
    straightened: float = -10.0       # P2 -> P1: d above this
    dislodged: float = 4.0            # P3/P4 -> P2: catheter into the trunk below this
    wire_in_rcca: float = 15.0        # P3 -> P4 ...
    wire_lead: float = 10.0           # ... and d >= this
    wire_back: float = 10.0           # P4 -> P3: wire into the right carotid below this
    target_mm: float = 5.0            # success
    cath_in_rcca: float = 15.0
    lead_min: float = 15.0
    lead_max: float = 45.0
    wrong_in: float = 8.0             # events
    wrong_out: float = 4.0
    lead_arch: float = 100.0
    lead_arch_out: float = 90.0
    fork: float = 64.0
    fork_out: float = 60.0


@register("monitor")
class PhaseDetector:
    """The referee (module docstring). ``state`` is replaced every step (``PhaseState``)."""

    def __init__(self, intervention, route_chain: Sequence[str] = ROUTE_CHAIN,
                 consecutive_steps: int = 2, platform_hold_steps: int = 8,
                 tip_direction_mm: float = 3.0, outside_margin_mm: float = 6.0,
                 same_length_mm: float = 0.5, same_length_min_inserted_mm: float = 2.0,
                 catheter_goal_depth_mm: float = 15.0, thresholds: Optional[Dict[str, float]] = None,
                 name: str = "phase_detector") -> None:
        self.name = name
        self.intervention = intervention
        self.route_chain = tuple(route_chain)
        self.consecutive_steps = int(consecutive_steps)
        self.platform_hold_steps = int(platform_hold_steps)
        self.tip_direction_mm = float(tip_direction_mm)
        self.outside_margin_mm = float(outside_margin_mm)
        self.same_length_mm = float(same_length_mm)
        self.same_length_min_inserted_mm = float(same_length_min_inserted_mm)
        self.catheter_goal_depth_mm = float(catheter_goal_depth_mm)
        self.t = Thresholds(**(thresholds or {}))
        self.cutoff = DEFAULT_CUTOFF
        self.route: Optional[Route] = None
        self.state = PhaseState()
        self.pending_events: Dict[str, int] = {k: 0 for k in EVENT_NAMES}
        self._candidate: Tuple[int, int] = (-1, 0)       # (target phase, steps it held)
        self._armed: Dict[str, bool] = {}
        self.devices: Optional[Devices] = None
        self.hook_point = None
        self.hook_to_target = self.hook_to_F = 0.0

    # ------------------------------------------------------------------ interface ----
    def set_cutoff(self, cutoff: int) -> None:
        self.cutoff = int(cutoff)
        self.state.cutoff = self.cutoff

    def report(self, event: str, count: int = 1) -> None:
        """An event seen outside the referee (the shell); counted in the next ``step``."""
        if event not in self.pending_events:
            raise KeyError(f"unknown event {event!r}; known: {EVENT_NAMES}")
        self.pending_events[event] += int(count)

    def reset(self, episode_nr: int = 0) -> None:
        self.route = Route.of(self.intervention.vessel_tree, self.route_chain)
        self.route.set_target(target_in_vessel_frame(self.intervention.target,
                                                      self.intervention.fluoroscopy),
                              self.catheter_goal_depth_mm)
        self.hook_point = self.route.aorta_location(self.route.s_O + self.t.past_O)
        self.hook_to_target = self.route.path_to_target(None, self.hook_point)
        self.hook_to_F = self.route.path_to_F(None, self.hook_point)
        self.pending_events = {k: 0 for k in EVENT_NAMES}
        self._candidate = (-1, 0)
        self._armed = {}
        measured = self._measure()
        self.state = replace(measured, phase=0, max_phase=0, previous_phase=0, steps_in_phase=0,
                             step=0, cutoff=self.cutoff)
        self._update_arming(self.state)
        restored = getattr(self.intervention, "snapshot_referee_state", None)
        if restored is not None:                     # the episode starts from a snapshot (snapshot.py)
            from .snapshot import apply_referee_state
            apply_referee_state(self, restored)
            self.intervention.snapshot_referee_state = None

    def step(self) -> None:
        previous = self.state
        events = {k: 0 for k in EVENT_NAMES}
        for k, v in self.pending_events.items():
            events[k] += v
        self.pending_events = {k: 0 for k in EVENT_NAMES}
        if bool(getattr(self.intervention.simulation, "simulation_error", False)):
            # stEVE has reset the instruments to the insertion point; nothing here is meaningful
            self.state = replace(previous, step=previous.step + 1, sim_error=True, events=events,
                                 event_totals=_add(previous.event_totals, events),
                                 steps_in_phase=previous.steps_in_phase + 1)
            return
        self.advance(self._measure(), events)

    def advance(self, s: PhaseState, events: Optional[Dict[str, int]] = None) -> PhaseState:
        """One referee step on a measured state (the geometry fields of ``s``; phase, counters
        and outcome come from here). ``step`` = measure + this; tests call it with made-up
        states."""
        previous = self.state
        events = {k: 0 for k in EVENT_NAMES} if events is None else events
        phase, steps_in_phase = previous.phase, previous.steps_in_phase + 1
        target = self._candidate_target(previous.phase, s)
        if target is None:
            self._candidate = (-1, 0)
        else:
            held = self._candidate[1] + 1 if self._candidate[0] == target else 1
            self._candidate = (target, held)
            if held >= self.consecutive_steps:
                if previous.phase in (3, 4) and target == 2:
                    events["dislodged"] += 1
                phase, steps_in_phase = target, 0
                self._candidate = (-1, 0)
        self._count_events(s, events)
        platform = (s.dist_to_target_w <= self.t.target_mm and s.depth_rcca_c >= self.t.cath_in_rcca
                    and self.t.lead_min <= s.d <= self.t.lead_max)
        platform_steps = previous.platform_steps + 1 if platform else 0
        failure = ""
        if s.clearance_w < -self.outside_margin_mm:
            failure = "outside_wire"
        elif s.clearance_c < -self.outside_margin_mm:
            failure = "outside_cath"
        elif abs(s.d) < self.same_length_mm and \
                min(s.L_wire, s.L_cath) >= self.same_length_min_inserted_mm:
            failure = "same_length"
        self.state = replace(
            s, phase=phase, previous_phase=previous.phase, steps_in_phase=steps_in_phase,
            max_phase=max(previous.max_phase, phase), step=previous.step + 1, events=events,
            event_totals=_add(previous.event_totals, events), platform_steps=platform_steps,
            success=platform_steps >= self.platform_hold_steps, hard_failure=bool(failure),
            failure_reason=failure, sim_error=False, cutoff=self.cutoff)
        return self.state

    # ------------------------------------------------------------------- internals ----
    def _measure(self) -> PhaseState:
        route = self.route
        devices = devices_of(self.intervention.simulation)
        self.devices = devices
        wire_tip, cath_tip = devices.tips
        lw, lc = route.locate(wire_tip), route.locate(cath_tip)
        direction = devices.tip_direction(CATH, self.tip_direction_mm)
        angle = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0)))) \
            if np.linalg.norm(direction) > 0 else 180.0
        rcca = self.route_chain[2]
        bct = self.route_chain[1]
        wrong = {}
        for vessel in WRONG_VESSELS:
            if vessel in route.graph.index:
                wrong[vessel] = max(route.depth(None, vessel, lw), route.depth(None, vessel, lc))
        return PhaseState(
            d=devices.d, L_wire=float(devices.inserted[WIRE]), L_cath=float(devices.inserted[CATH]),
            depth_bct_c=route.depth(None, bct, lc), depth_rcca_c=route.depth(None, rcca, lc),
            depth_rcca_w=route.depth(None, rcca, lw), hook_up=angle < self.t.hook_up_deg,
            wire_tip=np.asarray(wire_tip, dtype=np.float64), cath_tip=np.asarray(cath_tip, dtype=np.float64),
            cath_tip_dir=direction, hook_angle_deg=angle, branch_w=lw.branch, branch_c=lc.branch,
            clearance_w=lw.clearance, clearance_c=lc.clearance,
            aorta_pos_c=route.aorta_position(None, lc), depth_bct_w=route.depth(None, bct, lw),
            s_w=route.s(None, lw), s_c=route.s(None, lc),
            path_to_target_w=route.path_to_target(None, lw),
            path_to_target_c=route.path_to_target(None, lc), path_to_F_c=route.path_to_F(None, lc),
            dist_to_target_w=float(np.linalg.norm(np.asarray(wire_tip) - route.target)),
            depth_wrong=wrong, to_hook_w=self._to_hook(lw), to_hook_c=self._to_hook(lc),
            hook_to_target=self.hook_to_target, hook_to_F=self.hook_to_F)

    def _to_hook(self, loc) -> float:
        """Path left to the hook point H (O + 30 mm along the aorta); 0 once past it (on the aorta
        beyond H, or in a vessel leaving the aorta beyond H)."""
        if self.route.aorta_position(None, loc) >= self.t.past_O:
            return 0.0
        return self.route.path(loc, self.hook_point)

    def _candidate_target(self, phase: int, s: PhaseState) -> Optional[int]:
        t = self.t
        if phase == 0:
            rules = [(1, s.aorta_pos_c >= t.past_O)]
        elif phase == 1:
            rules = [(0, s.aorta_pos_c < t.back_P1), (2, s.d <= t.hook_behind and s.hook_up)]
        elif phase == 2:
            # the hook is straightened only by a wire still inside the catheter (d < 0) and not yet
            # past the trunk's mouth; a P2 reached by the catheter slipping out of the opening after
            # the wire came out stays P2 until the catheter sits >= 8 mm in again (-> P3)
            straightened = t.straightened < s.d < 0.0 and s.depth_bct_w <= 0.0
            rules = [(3, s.depth_bct_c >= t.engaged), (0, s.aorta_pos_c < t.back_P2),
                     (1, straightened)]
        elif phase == 3:
            rules = [(2, s.depth_bct_c < t.dislodged),
                     (4, s.depth_rcca_w >= t.wire_in_rcca and s.d >= t.wire_lead)]
        else:
            rules = [(2, s.depth_bct_c < t.dislodged), (3, s.depth_rcca_w < t.wire_back)]
        for target, holds in rules:
            if holds:
                return target
        return None

    def _update_arming(self, s: PhaseState) -> None:
        t = self.t
        for vessel, depth in s.depth_wrong.items():
            if depth < t.wrong_out:
                self._armed[f"wrong:{vessel}"] = True
        if s.d < t.lead_arch_out or s.branch_w != self.route_chain[0]:
            self._armed["lead_arch"] = True
        if s.depth_rcca_w < t.fork_out:
            self._armed["fork"] = True

    def _count_events(self, s: PhaseState, events: Dict[str, int]) -> None:
        t = self.t
        for vessel, depth in s.depth_wrong.items():
            key = f"wrong:{vessel}"
            if depth >= t.wrong_in and self._armed.get(key, True):
                events["wrong_vessel"] += 1
                self._armed[key] = False
        if s.d > t.lead_arch and s.branch_w == self.route_chain[0] and self._armed.get("lead_arch", True):
            events["wire_lead_arch"] += 1
            self._armed["lead_arch"] = False
        if s.depth_rcca_w >= t.fork and self._armed.get("fork", True):
            events["wire_past_fork"] += 1
            self._armed["fork"] = False
        self._update_arming(s)


def _add(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in set(a) | set(b)}


@register("info")
class PhaseInfo(Info):
    """The referee's step summary in ``info``: ``phase``, ``max_phase``, ``events``,
    ``is_success``, ``hard_failure``, ``failure_reason``, ``d``, ``cutoff``, ``sim_error``."""

    def __init__(self, phase_detector: PhaseDetector, name: str = "phase") -> None:
        super().__init__(name)
        self.phase_detector = phase_detector

    @property
    def info(self) -> Dict[str, Any]:
        return self.phase_detector.state.info()
