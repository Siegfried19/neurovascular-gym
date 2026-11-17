"""The referee of the two-device task: which of the five phases the episode is in, what happened
this step, success and hard failure (todo_list/archive/two_devices_training.md, sections 1-2).

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

The two tiers of the finish (round two, user decisions 2026-11-09; ``Thresholds``). The split is
where ``split_branch`` (``v8``, the external carotid) leaves the route: the ICA begins there, F (the
catheter's goal) is ``catheter_goal_past_split_mm`` (5) past it.

  - First tier (``tier1``, an event counted once per episode, ``state.tier1_done``; the episode goes
    on): wire tip within ``target_mm`` (5) of the target, catheter tip >= ``tier1_cath_depth`` (60 mm)
    into the right carotid and not yet past the split, both for ``platform_hold_steps`` (8) steps in
    a row.
  - Second tier (``state.success``, the episode ends): wire tip within ``target_mm`` of the target and
    catheter tip >= ``tier2_cath_past_split`` (5 mm) past the split, both for 8 steps in a row. The
    wire's lead is no condition (the geometry of the target window and F keeps the catheter on the
    wire's stiff part; the lead at the end is reported for the statistics).

Hard failure (``state.hard_failure``): a tip outside the vessel (more than ``outside_margin_mm``
beyond every centerline tube). A step with a simulation error (NaN, stEVE resets the instruments)
changes nothing and is flagged (``state.sim_error``). With the wall monitor in the config
(two_device/wall.py, ``WallBreach``), a centerline crossing the vessel mesh ends the episode first,
as a dropped step, so the outside-the-tubes hard failure is a backstop that normally never fires.

The progress rewards use the plain path along the tree the whole way (``PhaseState.progress_paths``;
the arch ruler to the hook point H = O + 30 mm is kept in ``to_hook_*`` only as the switched-off
reserve of the reward formula, user decision 2026-11-09 18:49). The arch ruler holds for a tip only
until that tip has reached H or gone past the trunk's mouth in this episode (``hook_done_*``); from
then on that tip reads the plain path (fix of 2026-11-10: without the switch the ruler measured the
way back to H for a tip up the trunk, so every mm up the trunk and the carotid counted as a loss).

Events (``state.events``, counted on the step they happen only; each needs the tip to come all the
way out before it can count again):

  - ``wrong_vessel``: a tip >= ``wrong_in`` (8) mm into the left carotid, left subclavian, right
    subclavian or, past the split, the split's other branch (``split_branch``: the external carotid on
    the ICA route; user 2026-11-11) (again only after it came back out to < ``wrong_out`` (0) mm);
  - ``dislodged``: the catheter, once >= ``seated`` (8) mm into the trunk, came back out of its
    mouth (< ``unseated`` (0) mm); counted again only after it sat >= 8 mm in again;
  - ``tier1``: the first tier (above), once.

``progress_ratio``: how much of the task's two paths is done, 0 at the insertion point and 1 with
the wire at the target and the catheter at F (the nearer tip's path to the target plus the catheter's
path to F, over the same two paths from the insertion point); the start curriculum's grid.

``landmark_grid``: the start curriculum's other grid (``training.start_curriculum.grid: landmarks``,
user decisions 2026-11-10 14:56 and 22:24): milestones generated from the route's structure, no
vessel name written in (``PhaseDetector.landmarks``), in the order of the procedure: the catheter
at the first branch's mouth along the aorta, the catheter seated ``ENTER_MM`` (8) into that branch;
the wire past every later place on the route where a vessel leaves it (``ENTER_MM`` into a vessel
the route turns into, past the mouth of a side vessel, ``ENTER_MM`` past the last junction before
the target, i.e. into the target vessel), the wire in the target ball; the catheter ``ENTER_MM`` into
every later vessel the route turns into, at the first tier's depth, at F. The grid is 1 + how many
milestones hold now, counted from the first one in order (``landmark_grids`` = milestones + 1). On
this route: nine milestones, grids 1-10.
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
ENTER_MM = 8.0          # "in" a vessel: this far past its mouth (the referee's events count the same 8)
N_PHASES = 5
PHASE_NAMES = ("arch", "hook", "select", "wire_out", "follow")
EVENT_NAMES = ("wrong_vessel", "dislodged", "tier1")
WRONG_VESSELS = ("lcca", "lsa", "rsa")   # plus the split's other branch (``split_branch``) when set
# what the referee carries from step to step (the rest of ``PhaseState`` is measured afresh each step)
HISTORY_FIELDS = ("phase", "events", "steps_in_phase", "max_phase", "step", "previous_phase", "success",
                  "hard_failure", "failure_reason", "sim_error", "platform_steps", "event_totals", "cutoff",
                  "tier1_done", "tier1_steps")
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
    depth_ica_c: float = -np.inf                     # catheter tip, mm past the ICA/ECA split (signed)
    depth_ica_w: float = -np.inf                     # wire tip, mm past the split
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
    platform_steps: int = 0                          # second-tier conditions held this many steps
    tier1_steps: int = 0                             # first-tier conditions held this many steps
    tier1_done: bool = False                         # the first tier was paid this episode
    progress_ratio: float = 0.0                      # 0 at the insertion point, 1 = both at their goals
    landmark_grid: int = 0                           # 1 + milestones held in order (module docstring)
    landmark_grids: int = 0                          # how many grids the milestones make (+1)
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
    hook_done_w: bool = False                        # this tip reached H or went past the trunk's mouth
    hook_done_c: bool = False                        # this episode: the arch ruler is over for it
    hook_to_target: float = 0.0                      # fixed path from H to the target, mm
    hook_to_F: float = 0.0                           # fixed path from H to F, mm

    @property
    def leading_path(self) -> float:
        """Path to the target of the tip further along (the smaller of the two)."""
        return min(self.path_to_target_w, self.path_to_target_c)

    def progress_paths(self) -> Tuple[float, float, float]:
        """The rulers of the two progress rewards: (wire to target, catheter to target, catheter
        to F), the plain path along the tree the whole way."""
        return (self.path_to_target_w, self.path_to_target_c, self.path_to_F_c)

    def hook_paths(self) -> Tuple[float, float, float]:
        """The reserve arch ruler (switched off by default): the way left to the hook point H plus
        the fixed way on from H, flat on the aorta past H; a tip whose arch ruler is over
        (``hook_done_*``: it reached H or went past the trunk's mouth this episode) reads the plain
        path from then on. Both read the same at H, so reaching H and pulling back into the trunk
        is continuous; a tip that enters the trunk without reaching H jumps once, at the mouth."""
        wire = self.path_to_target_w if self.hook_done_w else self.to_hook_w + self.hook_to_target
        cath = self.path_to_target_c if self.hook_done_c else self.to_hook_c + self.hook_to_target
        to_F = self.path_to_F_c if self.hook_done_c else self.to_hook_c + self.hook_to_F
        return wire, cath, to_F

    def info(self) -> Dict[str, Any]:
        return {"phase": int(self.phase), "max_phase": int(self.max_phase),
                "events": dict(self.events), "is_success": bool(self.success),
                "hard_failure": bool(self.hard_failure), "failure_reason": self.failure_reason,
                "d": float(self.d), "cutoff": int(self.cutoff), "sim_error": bool(self.sim_error),
                "tier1": bool(self.tier1_done), "depth_ica_c": float(self.depth_ica_c),
                "depth_ica_w": float(self.depth_ica_w), "progress_ratio": float(self.progress_ratio),
                "landmark_grid": int(self.landmark_grid), "landmark_grids": int(self.landmark_grids)}


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
    target_mm: float = 5.0            # the target ball (both tiers)
    tier1_cath_depth: float = 60.0    # first tier: catheter this far into the right carotid, not past the split
    tier2_cath_past_split: float = 5.0   # second tier (success): catheter this far past the split
    wrong_in: float = 8.0             # events: a wrong vessel entered this deep ...
    wrong_out: float = 0.0            # ... counts again only after coming back out to below this
    seated: float = 8.0               # dislodged: catheter this deep in the trunk arms the event ...
    unseated: float = 0.0             # ... fired when it comes back out to below this


@register("monitor")
class PhaseDetector:
    """The referee (module docstring). ``state`` is replaced every step (``PhaseState``)."""

    def __init__(self, intervention, route_chain: Sequence[str] = ROUTE_CHAIN,
                 consecutive_steps: int = 2, platform_hold_steps: int = 8,
                 tip_direction_mm: float = 3.0, outside_margin_mm: float = 6.0,
                 split_branch: Optional[str] = "v8", catheter_goal_past_split_mm: float = 5.0,
                 catheter_goal_depth_mm: Optional[float] = None,
                 thresholds: Optional[Dict[str, float]] = None, name: str = "phase_detector") -> None:
        """``split_branch``: the side vessel whose junction is the ICA/ECA split (``v8``); F is
        ``catheter_goal_past_split_mm`` past it. ``split_branch: null`` with ``catheter_goal_depth_mm``
        puts F that far into the right carotid instead (no split: the tiers' split conditions never
        hold, the first round's carotid task)."""
        self.name = name
        self.intervention = intervention
        self.route_chain = tuple(route_chain)
        self.consecutive_steps = int(consecutive_steps)
        self.platform_hold_steps = int(platform_hold_steps)
        self.tip_direction_mm = float(tip_direction_mm)
        self.outside_margin_mm = float(outside_margin_mm)
        self.split_branch = split_branch
        # the wrong vessels: the arch's other branches and, past the split, its other branch (the ECA
        # on the ICA route; user 2026-11-11). The ICA and ECA lumens overlap for the first ~6 mm past
        # the split; ICA lumen points read at most 6.7 mm into the ECA there (WORK_LOG 2026-11-11
        # 19:20), under the 8 mm entry, so the same threshold counts no ICA tip as an ECA entry
        self.wrong_vessels = WRONG_VESSELS + ((split_branch,) if split_branch is not None else ())
        self.catheter_goal_past_split_mm = float(catheter_goal_past_split_mm)
        self.catheter_goal_depth_mm = None if catheter_goal_depth_mm is None else float(catheter_goal_depth_mm)
        if split_branch is None and self.catheter_goal_depth_mm is None:
            raise ValueError("PhaseDetector: without split_branch, catheter_goal_depth_mm places F")
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
        self.s_split: Optional[float] = None
        self.total_path = 1.0                            # both paths from the insertion point, mm
        self.landmarks: List[Tuple[str, int, Optional[float]]] = []

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
        target = target_in_vessel_frame(self.intervention.target, self.intervention.fluoroscopy)
        if self.split_branch is not None:
            self.s_split = self.route.s_of_fork(self.split_branch)
            self.route.set_target(target, catheter_goal_s=self.s_split + self.catheter_goal_past_split_mm)
        else:
            self.s_split = None
            self.route.set_target(target, self.catheter_goal_depth_mm)
        self.hook_point = self.route.aorta_location(self.route.s_O + self.t.past_O)
        self.hook_to_target = self.route.path_to_target(None, self.hook_point)
        self.hook_to_F = self.route.path_to_F(None, self.hook_point)
        start = self.route.locate(self.route.insertion)
        self.total_path = max(self.route.path_to_target(None, start) + self.route.path_to_F(None, start), 1e-6)
        self.landmarks = self.make_landmarks()
        self.pending_events = {k: 0 for k in EVENT_NAMES}
        self._candidate = (-1, 0)
        self._armed = {}
        measured = self._measure(carry=False)
        self.state = replace(measured, phase=0, max_phase=0, previous_phase=0, steps_in_phase=0,
                             step=0, cutoff=self.cutoff)
        self._update_arming(self.state)
        restored = getattr(self.intervention, "snapshot_referee_state", None)
        if restored is not None:                     # the episode starts from a snapshot (snapshot.py)
            from .snapshot import apply_referee_state
            apply_referee_state(self, restored)
            self.intervention.snapshot_referee_state = None
            # keep the saved history (phase, events, counters), measure the rest again: the
            # snapshot may come from an episode with another target (the paths, the progress)
            saved = self.state
            self.state = replace(self._measure(carry=True), **{f: getattr(saved, f) for f in HISTORY_FIELDS})

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
                phase, steps_in_phase = target, 0
                self._candidate = (-1, 0)
        self._count_events(s, events)
        t = self.t
        at_target = s.dist_to_target_w <= t.target_mm
        tier1 = at_target and s.depth_rcca_c >= t.tier1_cath_depth and s.depth_ica_c < 0.0
        tier1_steps = previous.tier1_steps + 1 if tier1 else 0
        tier1_done = previous.tier1_done
        if tier1_steps >= self.platform_hold_steps and not tier1_done:
            events["tier1"] += 1
            tier1_done = True
        platform = at_target and s.depth_ica_c >= t.tier2_cath_past_split
        platform_steps = previous.platform_steps + 1 if platform else 0
        failure = ""
        if s.clearance_w < -self.outside_margin_mm:
            failure = "outside_wire"
        elif s.clearance_c < -self.outside_margin_mm:
            failure = "outside_cath"
        self.state = replace(
            s, phase=phase, previous_phase=previous.phase, steps_in_phase=steps_in_phase,
            max_phase=max(previous.max_phase, phase), step=previous.step + 1, events=events,
            event_totals=_add(previous.event_totals, events), platform_steps=platform_steps,
            tier1_steps=tier1_steps, tier1_done=tier1_done,
            success=platform_steps >= self.platform_hold_steps, hard_failure=bool(failure),
            failure_reason=failure, sim_error=False, cutoff=self.cutoff)
        return self.state

    # ------------------------------------------------------------------- internals ----
    def _measure(self, carry: bool = True) -> PhaseState:
        """This step's measurements; ``carry``: keep this episode's arch-ruler switches (off at reset)."""
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
        for vessel in self.wrong_vessels:
            if vessel in route.graph.index:
                wrong[vessel] = max(route.depth(None, vessel, lw), route.depth(None, vessel, lc))
        if self.s_split is not None:
            depth_ica_c, depth_ica_w = route.depth_past(self.s_split, lc), route.depth_past(self.s_split, lw)
        else:
            depth_ica_c = depth_ica_w = -np.inf
        path_w, path_c, path_F = (route.path_to_target(None, lw), route.path_to_target(None, lc),
                                  route.path_to_F(None, lc))
        ratio = float(np.clip(1.0 - (min(path_w, path_c) + path_F) / self.total_path, 0.0, 1.0))
        landmark_grid = 1 + self.landmarks_held(lw, lc, wire_tip)
        return PhaseState(
            d=devices.d, L_wire=float(devices.inserted[WIRE]), L_cath=float(devices.inserted[CATH]),
            depth_bct_c=route.depth(None, bct, lc), depth_rcca_c=route.depth(None, rcca, lc),
            depth_rcca_w=route.depth(None, rcca, lw), depth_ica_c=depth_ica_c, depth_ica_w=depth_ica_w,
            progress_ratio=ratio, landmark_grid=landmark_grid, landmark_grids=len(self.landmarks) + 1,
            hook_up=angle < self.t.hook_up_deg,
            wire_tip=np.asarray(wire_tip, dtype=np.float64), cath_tip=np.asarray(cath_tip, dtype=np.float64),
            cath_tip_dir=direction, hook_angle_deg=angle, branch_w=lw.branch, branch_c=lc.branch,
            clearance_w=lw.clearance, clearance_c=lc.clearance,
            aorta_pos_c=route.aorta_position(None, lc), depth_bct_w=route.depth(None, bct, lw),
            s_w=route.s(None, lw), s_c=route.s(None, lc),
            path_to_target_w=path_w, path_to_target_c=path_c, path_to_F_c=path_F,
            dist_to_target_w=float(np.linalg.norm(np.asarray(wire_tip) - route.target)),
            depth_wrong=wrong, to_hook_w=self._to_hook(lw), to_hook_c=self._to_hook(lc),
            hook_done_w=(carry and self.state.hook_done_w) or self.hook_reached(lw),
            hook_done_c=(carry and self.state.hook_done_c) or self.hook_reached(lc),
            hook_to_target=self.hook_to_target, hook_to_F=self.hook_to_F)

    def make_landmarks(self) -> List[Tuple[str, int, Optional[float]]]:
        """The milestones of this route and target (module docstring), in order: (what, device,
        route arc length the tip must reach; None = the wire in the target ball)."""
        route, chain = self.route, self.route_chain
        s_first = route.offsets[chain[1]]                 # the first branch's mouth on the route
        s_target = route.s_target
        turns = {name for name, _m, _d in route.junctions}
        out: List[Tuple[str, int, Optional[float]]] = [
            (f"catheter at the {chain[1]} mouth", CATH, s_first),
            (f"catheter {ENTER_MM:g} mm into {chain[1]}", CATH, s_first + ENTER_MM)]
        later = [(name, s) for name, _m, _d, s in route.all_junctions if s_first + 1e-6 < s < s_target]
        for k, (name, s) in enumerate(later):
            if k == len(later) - 1:
                out.append((f"wire {ENTER_MM:g} mm past the {name} junction (target vessel)", WIRE, s + ENTER_MM))
            elif name in turns:
                out.append((f"wire {ENTER_MM:g} mm into {name}", WIRE, s + ENTER_MM))
            else:
                out.append((f"wire past the {name} mouth", WIRE, s))
        out.append(("wire in the target ball", WIRE, None))
        for name in chain[2:]:
            if name in turns:
                out.append((f"catheter {ENTER_MM:g} mm into {name}", CATH, route.offsets[name] + ENTER_MM))
        tier1 = route.s_rcca + self.t.tier1_cath_depth
        catheter_last = max(s for _w, device, s in out if device == CATH)
        if catheter_last < tier1 < route.s_F:
            out.append((f"catheter at the first tier's depth ({self.t.tier1_cath_depth:g} mm)", CATH, tier1))
        out.append(("catheter at F", CATH, route.s_F))
        return out

    def along(self, loc, s_mark: float) -> float:
        """Signed mm past route arc length ``s_mark`` for a tip at ``loc``: ``Route.depth_past``, except
        that the aorta beyond the first branch's junction (towards the root, where the hooked catheter
        swings out) is off the route for any mark past that junction."""
        route = self.route
        if loc.branch == self.route_chain[0] and s_mark > route.s_O + 1e-9 and loc.arc > route.s_O:
            return -route.graph.path(loc, route.location_at(s_mark))
        return route.depth_past(s_mark, loc)

    def landmarks_held(self, lw, lc, wire_tip) -> int:
        """How many milestones hold now, from the first one, in order."""
        held = 0
        for _what, device, s_mark in self.landmarks:
            if s_mark is None:
                ok = float(np.linalg.norm(np.asarray(wire_tip, dtype=np.float64) - self.route.target)) \
                    <= self.t.target_mm
            else:
                ok = self.along(lw if device == WIRE else lc, s_mark) >= 0.0
            if not ok:
                break
            held += 1
        return held

    def hook_reached(self, loc) -> bool:
        """Whether a tip here ends its arch ruler: at or past H (``_to_hook`` 0), or past the mouth
        of the route's first branch (the trunk, and everything up it)."""
        return self._to_hook(loc) == 0.0 or self.route.depth(None, self.route_chain[1], loc) > 0.0

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
        """Hysteresis: a wrong-vessel entry counts again only after the tip came back out
        (``wrong_out``); the catheter's dislodging only after it sat ``seated`` mm in again."""
        t = self.t
        for vessel, depth in s.depth_wrong.items():
            if depth < t.wrong_out:
                self._armed[f"wrong:{vessel}"] = True
        if s.depth_bct_c >= t.seated:
            self._armed["seated"] = True

    def _count_events(self, s: PhaseState, events: Dict[str, int]) -> None:
        t = self.t
        for vessel, depth in s.depth_wrong.items():
            key = f"wrong:{vessel}"
            if depth >= t.wrong_in and self._armed.get(key, True):
                events["wrong_vessel"] += 1
                self._armed[key] = False
        if s.depth_bct_c < t.unseated and self._armed.get("seated", False):
            events["dislodged"] += 1
            self._armed["seated"] = False
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
