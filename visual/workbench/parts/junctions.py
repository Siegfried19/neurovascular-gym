"""Junction map layer (路口地图): a green ball at each of the six places on the route to the right
internal carotid where a vessel branches off (round-two design, block 1, WORK_LOG 2026-11-08 01:33):
the left subclavian's, the left common carotid's and the brachiocephalic trunk's mouths on the
aorta, the trunk's end, the side branch 64 mm up the carotid and the internal / external carotid
split, in that order.

Each point is where its branch leaves its parent, as ``two_device.route.VesselGraph`` computes it
(``mouth_arc``): for a branch off the aorta where its centerline leaves the aorta's tube, for the
others the branch's start. Read from ``rebuilt_cut.json`` (every BIRO phantom version shares its
frame), drawn only when the scene is that phantom (``body.fits``). The design gives the policy
these six, each with its position in the image and the direction of the branch to take; the
directions are not drawn here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np

from .body import REFERENCE, fits
from .frame import SceneInfo
from .scene import MM, Color

JUNCTION_COLOR: Color = (40, 200, 90)
RADIUS_MM = 2.0
# (branch whose mouth it is, name)
JUNCTIONS: Tuple[Tuple[str, str], ...] = (
    ("lsa", "左锁骨下口"), ("lcca", "左颈总口"), ("bct", "头臂干口"),
    ("rcca", "头臂干末端"), ("v7", "颈总 64 mm 侧支口"), ("v8", "颈内 / 颈外分叉"),
)


@lru_cache(maxsize=2)
def junction_points(spec_path: Path = REFERENCE) -> List[Tuple[str, np.ndarray]]:
    """[(name, point mm in the training frame), ...] in route order."""
    from guidewire_rl.env.two_device.route import VesselGraph

    spec = json.loads(Path(spec_path).read_text())
    branches = [SimpleNamespace(name=n, coordinates=b["coordinates"], radii=b["radii"])
                for n, b in spec["branches"].items()]
    graph = VesselGraph(branches, [(np.asarray(j["point"], dtype=np.float64), j["branches"])
                                   for j in spec["junctions"]])
    return [(label, graph.point(branch, float(graph.mouth_arc[graph.index[branch]])))
            for branch, label in JUNCTIONS]


class JunctionLayer:
    """The six junctions as green balls; nothing when the scene is not the BIRO phantom."""

    def __init__(self, server, path: str = "/junctions") -> None:
        self._server, self._path = server, path
        self._balls: List = []
        self._visible = True

    def set_scene(self, info: SceneInfo) -> None:
        for ball in self._balls:
            ball.remove()
        self._balls = []
        if not fits(info):
            return
        for i, (_, point) in enumerate(junction_points()):
            self._balls.append(self._server.scene.add_icosphere(
                f"{self._path}/{i}", radius=RADIUS_MM * MM, color=JUNCTION_COLOR,
                position=tuple(np.asarray(point, dtype=np.float64) * MM), cast_shadow=False,
                visible=self._visible))

    @property
    def shown(self) -> bool:
        return bool(self._balls)

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for ball in self._balls:
            ball.visible = self._visible
