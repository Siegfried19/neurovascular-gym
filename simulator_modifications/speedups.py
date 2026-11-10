"""Two SOFA speed-ups (docs/simulator_changes.md, problems 4 and 5), done the way the lab
does them (reference/surgery-all-in-one/51-legobot-ros2/legobot_sofa/legobot_sofa/twin.py:89-107,
148-156, measured there on SOFA 25.12).

Problem 4, collision tree depth 6 -> 16: SOFA sorts the vessel triangles into a tree of
boxes ``depth`` levels deep and compares triangles one by one only inside the smallest
boxes. stEVE's 6 levels leave the 58k-triangle phantom with ~900 triangles per box. A
deeper tree finds the same contacts with fewer tests; only the order of the contacts
changes, so the iterative solver lands on slightly different numbers (lab: tip median
0.003 mm, max 0.62 mm).

Problem 5, collision detection beside the free motion: ``FreeMotionAnimationLoop`` can
run the two on separate threads; they read and write different state, so results are
unchanged. Alone, the flag makes SOFA's task scheduler start one thread per core (most of
them spinning). MultiThreading's broad and narrow phases, added before the loop, set the
scheduler up first with ``collision_threads`` threads (they find the same contacts as the
serial phases).

Problem 16, the scene slows down as it runs: every SOFA step leaves something behind that
makes later steps slower (phantom, measured 2026-10-25: 0.033 -> 0.057 s per control step over
6000 steps without any reset; stEVE's per-episode reset does not clear it, so a training run
went from 0.24 to 0.48 s per step). The cause (found 2026-10-26): LCPConstraintSolver with
``initial_guess`` on (the default) looks up last step's contact forces in a table keyed by
constraint and contact id (a mesh element index); entries are only marked -1, never removed,
and the whole table is scanned every step (LCPConstraintSolver.cpp, keepContactForcesValue,
SOFA v25.12.00). A reset keeps the solver and its table; a rebuilt scene has a new solver.
Rebuilding the scene from scratch costs ~0.11 s, so by default the scene is rebuilt at every
reset (``rebuild_every_resets``; user decision 2026-10-27: keep ``initial_guess`` and rebuild).
Labels re-attach to the new scene (GuidewireEnv.reset).

Problem 20, the collision alarm distance (``LocalMinDistance.alarmDistance``, stEVE writes 0.5 mm):
at the start of every physics step each wire collision point is paired with the wall triangles
closer than this, and only those pairs become constraints ("stay 0.3 mm off this plane at the end
of the step"). A point that moves farther than this in one step towards a wall it was not paired
with passes through unopposed. ``alarm_distance`` makes it a parameter (default unchanged);
BeamAdapter's own examples use 2-3 mm. Measured 2026-10-28 on the phantom: 2 mm stops the one
exit the stock wire had in 18 episodes and costs ~30 % per step; it does not save a real-mass wire
at 6 ms (that needs 2 ms steps, see configs/phantom_real_mass.yaml).

This class carries the hollow-catheter fix (problem 2, ``hollow_catheter.py``); ``type:
SofaBeamAdapter`` resolves to its subclass in ``device_contact.py`` (problems 8 and 22, all off
by default).
"""

from __future__ import annotations

from . import UPSTREAM

if UPSTREAM:
    from .hollow_catheter import SofaBeamAdapter as HollowSofaBeamAdapter

    # not registered: ``type: SofaBeamAdapter`` is device_contact.SofaBeamAdapter, built on this one
    # (one registration only, so the class a config gets never depends on which module was imported last)
    class SofaBeamAdapter(HollowSofaBeamAdapter):
        """stEVE's ``SofaBeamAdapter`` with hollow catheters, the two speed-ups and a fresh scene per reset."""

        def __init__(self, friction: float = 0.1, dt_simulation: float = 0.006,
                     collision_tree_depth: int = 16, parallel_collision: bool = True,
                     collision_threads: int = 2, rebuild_every_resets: int = 1,
                     alarm_distance: float = 0.5) -> None:
            super().__init__(friction=friction, dt_simulation=dt_simulation)
            self.alarm_distance = float(alarm_distance)     # mm; stEVE's 0.5 unless a config says otherwise
            self.collision_tree_depth = int(collision_tree_depth)
            self.parallel_collision = bool(parallel_collision)
            self.collision_threads = int(collision_threads)
            self.rebuild_every_resets = int(rebuild_every_resets)   # 0: only when stEVE rebuilds
            self.resets = 0
            self.builds = 0

        def reset(self, *args, **kwargs):
            self.resets += 1
            if (self.root is not None and self.rebuild_every_resets > 0
                    and self.resets % self.rebuild_every_resets == 0):
                self._mesh_path = None      # a changed scene input makes stEVE rebuild everything
            super().reset(*args, **kwargs)

        def _basic_setup(self, friction: float):
            self.builds += 1
            self._root_components(friction)

        def _load_plugins(self):
            super()._load_plugins()
            if self.parallel_collision:
                self.root.addObject("RequiredPlugin", pluginName="MultiThreading")

        def _root_components(self, friction: float):
            # stEVE's _basic_setup (sofabeamadapter.py:229-249) with the two changes.
            if self.parallel_collision:
                self.root.addObject("ParallelBruteForceBroadPhase", nbThreads=self.collision_threads)
                self.root.addObject("ParallelBVHNarrowPhase", nbThreads=self.collision_threads)
                self.root.addObject("FreeMotionAnimationLoop",
                                    parallelCollisionDetectionAndFreeMotion=True)
                self.root.addObject("CollisionPipeline", draw="0",
                                    depth=str(self.collision_tree_depth), verbose="1")
            else:
                self.root.addObject("FreeMotionAnimationLoop")
                self.root.addObject("CollisionPipeline", draw="0",
                                    depth=str(self.collision_tree_depth), verbose="1")
                self.root.addObject("BruteForceBroadPhase")
                self.root.addObject("BVHNarrowPhase")
            self.root.addObject("LocalMinDistance", contactDistance=0.3, alarmDistance=self.alarm_distance,
                                angleCone=0.02, name="localmindistance")
            self.root.addObject("CollisionResponse", response="FrictionContactConstraint")
            self.root.addObject("LCPConstraintSolver", mu=friction, tolerance=1e-4, maxIt=2000,
                                name="LCP", build_lcp=False)
