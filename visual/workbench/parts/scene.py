"""3D scene layers (viser). Each layer owns its handles under one scene path.

Contract (same as MuscleMimic's viewer plugins):
    __init__(server, ...)      create handles
    set_scene(info)            static geometry; called again when the vessel changes
    update(frame, ...)         per-step data -> handles; never touches the simulator
    set_visible(bool)

Scene units are metres (viser's camera and grid defaults assume them); everything
arrives in mm and is scaled by ``MM``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frame import Frame, SceneInfo, fluoro_rotation, instrument_polylines, polyline_head, tracked_points_3d

MM = 1e-3
Color = Tuple[int, int, int]
VESSEL_COLOR: Color = (240, 178, 162)
CENTERLINE_COLOR: Color = (239, 159, 39)
AORTA_COLOR: Color = (150, 150, 145)
WIRE_COLOR: Color = (226, 75, 74)
CATHETER_COLOR: Color = (31, 119, 180)
TARGET_COLOR: Color = (239, 159, 39)
REACHED_COLOR: Color = (29, 158, 117)
INSERTION_COLOR: Color = (29, 158, 117)
TRAIL_COLOR: Color = (55, 138, 221)
OTHER_LINE_COLOR: Color = (136, 135, 128)
TUBE_COLOR: Color = (127, 119, 221)
TRACKED_COLOR: Color = (55, 138, 221)


def force_color(force_n: Optional[float], full_scale_n: float = 1.0) -> Color:
    """Wire colour from its tip contact force: red at 0 N, towards yellow-white at full scale."""
    if force_n is None or not np.isfinite(force_n):
        return WIRE_COLOR
    t = float(np.clip(force_n / full_scale_n, 0.0, 1.0))
    hot = np.array([255, 236, 120])
    return tuple(int(c) for c in np.round(np.array(WIRE_COLOR) * (1 - t) + hot * t))


class VesselLayer:
    def __init__(self, server, path: str = "/vessel", opacity: float = 0.35) -> None:
        self._server, self._path, self._opacity = server, path, opacity
        self._handle = None
        self._visible = True
        self._look: Optional[Tuple[Color, float]] = None
        self._outside = "double"     # the side of the faces that looks out of the vessel

    def set_scene(self, info: SceneInfo) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._outside = outside_side(info.vessel_vertices, info.vessel_faces)
        color, opacity = (VESSEL_COLOR, self._opacity) if self._look is None else self._look
        self._handle = self._server.scene.add_mesh_simple(
            self._path, vertices=info.vessel_vertices * MM, faces=info.vessel_faces,
            color=color, opacity=opacity, side="double" if self._look is None else self._outside,
            cast_shadow=False, receive_shadow=False, visible=self._visible)

    def set_opacity(self, opacity: float) -> None:
        self._opacity = float(opacity)
        if self._handle is not None and self._look is None:
            self._handle.opacity = self._opacity

    def set_look(self, look: Optional[Tuple[Color, float]]) -> None:
        """(colour, opacity) instead of the normal look (X-ray mode); None restores it. The
        special look draws only the outer surface, so a see-through vessel shades evenly."""
        self._look = look
        if self._handle is not None:
            self._handle.color = VESSEL_COLOR if look is None else look[0]
            self._handle.opacity = self._opacity if look is None else look[1]
            self._handle.side = "double" if look is None else self._outside

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible


def outside_side(vertices: np.ndarray, faces: np.ndarray) -> str:
    """"front" when a closed mesh's normals point out of it, "back" when they point in (signed volume)."""
    if len(faces) == 0:
        return "double"
    v = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    volume = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0
    return "front" if volume > 0 else "back"


class CenterlineLayer:
    """Centerlines of one group: the training's branches (``task=True``, with the insertion
    point) or every other vessel's."""

    def __init__(self, server, path: str = "/centerlines", task: bool = True) -> None:
        self._server, self._path, self._task = server, path, task
        self._handles = []
        self._visible = True

    def set_scene(self, info: SceneInfo) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for name, line in info.centerlines.items():
            if len(line) < 2 or info.is_task_branch(name) != self._task:
                continue
            if self._task:
                color, thickness = (AORTA_COLOR if name == "aorta" else CENTERLINE_COLOR), 0.7 * MM
            else:
                color, thickness = OTHER_LINE_COLOR, 0.45 * MM
            self._handles.append(self._server.scene.add_spline_catmull_rom(
                f"{self._path}/{name}", points=np.asarray(line) * MM, thickness=thickness,
                color=color, visible=self._visible))
        if not self._task:
            return
        p, d = info.insertion_position * MM, info.insertion_direction
        self._handles.append(self._server.scene.add_spline_catmull_rom(
            f"{self._path}/insertion", points=np.stack([p - d * 35 * MM, p]), thickness=1.6 * MM,
            color=INSERTION_COLOR, visible=self._visible))

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for h in self._handles:
            h.visible = self._visible


class WireLayer:
    """The inserted instruments as tubes, each with a tip ball.

    One instrument: the wire, ``diameter_mm`` thick, coloured by tip force. Two or more coaxial ones
    (guidewire in a catheter): the beam is split by inserted lengths (``instrument_polylines``); each
    is drawn at its outer diameter (at least ``MIN_DIAMETER_MM``), the thinnest (the wire) coloured by
    tip force and the others ``CATHETER_COLOR``. A thinner instrument is drawn only where it sticks
    out of the thicker one: inside it is hidden, as in reality, and two tubes on the same points
    would flicker.
    """

    MIN_DIAMETER_MM = 1.0

    def __init__(self, server, path: str = "/wire", diameter_mm: float = 1.8) -> None:
        self._server, self._path = server, path
        self._diameter_mm = diameter_mm
        self._splines: List = []                      # tube per instrument (None until first drawn)
        self._tube_shown: List[bool] = []             # this instrument's tube has points this step
        self._tips: List = []                         # (radius_mm, handle) per instrument
        self._tip_placed: List[bool] = []             # this instrument has a tip to show
        self._visible = True
        self.color_override: Optional[Color] = None   # X-ray mode draws the instruments dark

    def update(self, frame: Frame, force_n: Optional[float] = None,
               instruments: Optional[Sequence] = None) -> None:
        wire_color = self.color_override or force_color(force_n)
        inserted = frame.inserted_mm
        if instruments is not None and len(instruments) >= 2 and inserted is not None \
                and len(inserted) == len(instruments):
            lines = instrument_polylines(frame.wire_mm, inserted)
            outer = [max((s.diameter_outer_mm for s in inst.sections), default=0.0) for inst in instruments]
            thinnest = int(np.argmin(outer))
            parts = []
            for i, line in enumerate(lines):
                cover = max([float(inserted[j]) for j in range(len(lines)) if outer[j] > outer[i]], default=None)
                shown = line if cover is None else polyline_head(line, float(inserted[i]) - cover)
                color = wire_color if i == thinnest else (self.color_override or CATHETER_COLOR)
                diameter = max(outer[i], self.MIN_DIAMETER_MM)
                parts.append((shown, diameter, color, 0.6 * diameter))
        else:
            parts = [(np.asarray(frame.wire_mm, dtype=np.float64), self._diameter_mm, wire_color,
                      1.1 * self._diameter_mm)]
        self._tip_placed = []
        for i, (points, diameter, color, radius) in enumerate(parts):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3) * MM
            self._draw_tube(i, points, diameter * MM, color)
            tip = self._tip(i, radius)
            if len(points):
                tip.position = tuple(points[0])
                tip.color = color
            self._tip_placed.append(len(points) > 0)
            tip.visible = self._visible and self._tip_placed[-1]
        for _, handle in self._tips[len(parts):]:
            if handle is not None:
                handle.visible = False
        for i in range(len(parts), len(self._splines)):
            self._tube_shown[i] = False
            if self._splines[i] is not None:
                self._splines[i].visible = False

    def _draw_tube(self, i: int, points: np.ndarray, thickness: float, color: Color) -> None:
        """Tube ``i``, changed in place (removing and adding it every step makes it flash);
        hidden while it has fewer than two points."""
        while len(self._splines) <= i:
            self._splines.append(None)
            self._tube_shown.append(False)
        spline = self._splines[i]
        self._tube_shown[i] = len(points) >= 2
        if len(points) < 2:
            if spline is not None:
                spline.visible = False
            return
        segments = min(4 * len(points), 1200)
        if spline is None:
            self._splines[i] = self._server.scene.add_spline_catmull_rom(
                f"{self._path}/body{i}" if i else f"{self._path}/body", points=points, thickness=thickness,
                color=color, segments=segments, visible=self._visible)
            return
        spline.points, spline.segments = points, segments
        if spline.thickness != thickness:
            spline.thickness = thickness
        if tuple(spline.color) != tuple(color):
            spline.color = color
        if spline.visible != self._visible:
            spline.visible = self._visible

    def _tip(self, i: int, radius_mm: float):
        while len(self._tips) <= i:
            self._tips.append((None, None))
        radius, handle = self._tips[i]
        if handle is None or radius != radius_mm:
            if handle is not None:
                handle.remove()
            handle = self._server.scene.add_icosphere(f"{self._path}/tip{i}" if i else f"{self._path}/tip",
                                                     radius=radius_mm * MM, color=WIRE_COLOR, cast_shadow=False)
            self._tips[i] = (radius_mm, handle)
        return handle

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for spline, shown in zip(self._splines, self._tube_shown):
            if spline is not None:
                spline.visible = self._visible and shown
        for i, (_, handle) in enumerate(self._tips):
            if handle is not None:
                handle.visible = self._visible and i < len(self._tip_placed) and self._tip_placed[i]


class TrailLayer:
    """Where the tip has been this episode."""

    def __init__(self, server, path: str = "/trail", n_max: int = 2000) -> None:
        self._server, self._path, self._n_max = server, path, n_max
        self._handle = None
        self._visible = True
        self._shown = False

    def update(self, tips_mm: np.ndarray) -> None:
        """Changed in place (removing and adding it every step makes it flash)."""
        tips = np.asarray(tips_mm, dtype=np.float32)[-self._n_max:]
        self._shown = len(tips) >= 2
        if not self._shown:
            if self._handle is not None:
                self._handle.visible = False
            return
        segments = np.stack([tips[:-1], tips[1:]], axis=1) * MM
        if self._handle is None:
            self._handle = self._server.scene.add_line_segments(
                self._path, points=segments, colors=TRAIL_COLOR, thickness=2.0,
                thickness_units="screen", visible=self._visible)
            return
        colors = np.empty(segments.shape, np.uint8)
        colors[:] = TRAIL_COLOR
        self._handle.colors, self._handle.points = colors, segments
        if self._handle.visible != self._visible:
            self._handle.visible = self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible and self._shown


class TargetLayer:
    """The target point and its "reached" sphere."""

    def __init__(self, server, path: str = "/target") -> None:
        self._server, self._path = server, path
        self._radius = None
        self._sphere = None
        self._dot = server.scene.add_icosphere(f"{path}/point", radius=1.2 * MM, color=TARGET_COLOR,
                                               cast_shadow=False)
        self._visible = True

    def set_scene(self, info: SceneInfo) -> None:
        if self._sphere is not None:
            self._sphere.remove()
        self._sphere = self._server.scene.add_icosphere(
            f"{self._path}/reach", radius=info.target_threshold_mm * MM, color=TARGET_COLOR,
            opacity=0.3, cast_shadow=False, visible=self._visible)

    def update(self, frame: Frame) -> None:
        shown = self._visible and frame.target_mm is not None
        if frame.target_mm is not None:
            p = tuple(np.asarray(frame.target_mm, dtype=np.float64) * MM)
            self._dot.position = p
            if self._sphere is not None:
                self._sphere.position = p
                self._sphere.color = REACHED_COLOR if frame.success else TARGET_COLOR
        self._dot.visible = shown
        if self._sphere is not None:
            self._sphere.visible = shown

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self._dot.visible = self._visible
        if self._sphere is not None:
            self._sphere.visible = self._visible


class ForceArrowLayer:
    """Contact force at the tip as an arrow, once something supplies the vector.

    Reads ``frame.extras["force_vector_N"]`` (3 numbers, vessel frame). The
    ``distal_force`` label gives magnitudes only today, so this stays hidden
    until a label or the simulator reports the direction.
    """

    def __init__(self, server, path: str = "/force", mm_per_newton: float = 20.0) -> None:
        self._server, self._path, self._scale = server, path, mm_per_newton
        self._handle = None
        self._visible = True

    def update(self, frame: Frame) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        vector = frame.extras.get("force_vector_N")
        if vector is None or frame.tip_mm is None:
            return
        start = np.asarray(frame.tip_mm, dtype=np.float64)
        end = start + np.asarray(vector, dtype=np.float64) * self._scale
        self._handle = self._server.scene.add_line_segments(
            self._path, points=np.stack([start, end])[None] * MM, colors=(255, 236, 120),
            thickness=4.0, thickness_units="screen", visible=self._visible)

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible


def tube_mesh(points_mm: np.ndarray, radii_mm: np.ndarray, sides: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """A tube of varying radius around a polyline: (vertices mm, faces)."""
    p = np.asarray(points_mm, dtype=np.float64)
    r = np.asarray(radii_mm, dtype=np.float64).reshape(-1)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    p, r = p[keep], r[keep]
    if len(p) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    normal = np.cross(t[0], [0.0, 0.0, 1.0] if abs(t[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    normal /= np.linalg.norm(normal)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rings = []
    for i in range(len(p)):
        normal = normal - (normal @ t[i]) * t[i]                 # carry the frame along (no twist)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(t[i], normal)
        rings.append(p[i] + r[i] * (np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * binormal))
    vertices = np.concatenate(rings)
    faces = []
    for i in range(len(p) - 1):
        a, b = i * sides, (i + 1) * sides
        for j in range(sides):
            k = (j + 1) % sides
            faces += [[a + j, b + j, b + k], [a + j, b + k, a + k]]
    return vertices, np.asarray(faces, dtype=np.int32)


class RadiusTubeLayer:
    """Each centerline drawn as a tube of its recorded radius: the lumen the observation and
    reward assume, to hold against the collision mesh."""

    def __init__(self, server, path: str = "/radius_tubes", opacity: float = 0.35) -> None:
        self._server, self._path, self._opacity = server, path, opacity
        self._handle = None
        self._visible = True

    def set_scene(self, info: SceneInfo) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        vertices, faces, offset = [], [], 0
        for name, line in info.centerlines.items():
            radii = info.centerline_radii.get(name)
            if radii is None or len(radii) != len(line):
                continue
            v, f = tube_mesh(line, radii)
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        if not vertices:
            return
        self._handle = self._server.scene.add_mesh_simple(
            self._path, vertices=np.concatenate(vertices) * MM, faces=np.concatenate(faces), color=TUBE_COLOR,
            opacity=self._opacity, side="double", cast_shadow=False, receive_shadow=False, visible=self._visible)

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._handle is not None:
            self._handle.visible = self._visible


class TrackedPointsLayer:
    """The wire points behind the policy's tracked image points (tip red, the rest blue)."""

    def __init__(self, server, path: str = "/tracked", count: int = 5, spacing_mm: float = 1.0) -> None:
        self._count, self._spacing = count, spacing_mm
        self._dots = [server.scene.add_icosphere(f"{path}/{i}", radius=(1.3 if i == 0 else 0.9) * MM,
                                                 color=WIRE_COLOR if i == 0 else TRACKED_COLOR,
                                                 cast_shadow=False, visible=False) for i in range(count)]
        self._visible = False
        self._placed = 0          # how many dots have a position from the current frame

    def update(self, frame: Frame, info: SceneInfo) -> None:
        points = tracked_points_3d(frame.wire_mm, info.image_rot_zx_deg, self._spacing, self._count)
        self._placed = len(points) if len(frame.wire_mm) >= 2 else 0
        for i, dot in enumerate(self._dots):
            if i < self._placed:
                dot.position = tuple(points[i] * MM)
            dot.visible = self._visible and i < self._placed

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for i, dot in enumerate(self._dots):
            dot.visible = self._visible and i < self._placed


# ----------------------------------------------------------------- camera --
def scene_center_mm(info: SceneInfo) -> np.ndarray:
    """Middle of the training's branches (all centerlines when none are named)."""
    lines = [v for k, v in info.centerlines.items() if info.is_task_branch(k)] or list(info.centerlines.values())
    lines = np.concatenate(lines)
    return (lines.min(axis=0) + lines.max(axis=0)) / 2


def camera_pose(preset: str, info: SceneInfo, tip_mm: Optional[np.ndarray] = None,
                distance_mm: float = 300.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, look_at, up) in scene units for a named view.

    ``fluoro``: along the fluoroscope's projection axis, so the screen shows what the
    2D tracking sees (right = image x, up = image y). ``side``: from the +x side.
    ``tip``: fluoro direction, close to the tip.
    """
    rot = fluoro_rotation(info.image_rot_zx_deg)
    forward, up = rot.T @ np.array([0.0, 1.0, 0.0]), rot.T @ np.array([0.0, 0.0, 1.0])
    center = scene_center_mm(info)
    if preset == "side":
        forward, up = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    elif preset == "tip" and tip_mm is not None:
        center, distance_mm = np.asarray(tip_mm, dtype=np.float64), 90.0
    position = center - forward * distance_mm
    return position * MM, center * MM, up


def apply_camera(clients: Sequence, pose) -> None:
    position, look_at, up = pose
    for client in clients:
        with client.atomic():
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(look_at)
