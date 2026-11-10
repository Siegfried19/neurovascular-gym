"""The path length in the reward without a deep copy every step (docs/simulator_changes.md,
problem 18).

``BruteForceBFS._create_search_graph`` (stEVE ``eve/pathfinder/bruteforcebfs.py``) starts every
step from ``deepcopy(self._search_graph_base)``. That copies every branching point and, through
them, every branch with its coordinates, only to append "start" and "target" entries. With the
six-branch phantom it costs about 1 ms; with a centerline for every phantom vessel (75 branches)
about 7 ms per step. The search only needs its own lists, so this copies the lists and keeps
the branching points themselves; the paths and lengths are the same.
"""

from __future__ import annotations

from guidewire_rl.env.registry import register
from . import UPSTREAM

if UPSTREAM:
    import eve

    @register("pathfinder", "BruteForceBFS")
    class BruteForceBFS(eve.pathfinder.BruteForceBFS):
        def _create_search_graph(self, start_branch, target_branch):
            # stEVE's method with the deepcopy replaced by a copy of the lists
            search_graph = {node: list(nodes) for node, nodes in self._search_graph_base.items()}
            if start_branch == target_branch:
                search_graph["start"] = ["target"]
                return search_graph
            start_connections = []
            for branching_point in self.intervention.vessel_tree.branching_points:
                if start_branch in branching_point.connections:
                    start_connections.append(branching_point)
                if target_branch in branching_point.connections:
                    search_graph[branching_point].append("target")
            search_graph["start"] = start_connections
            return search_graph
