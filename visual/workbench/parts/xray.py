"""X-ray mode: the main 3D view as the C-arm shows it.

The camera looks along the fluoroscope axis of the environment (``image_rot_zx``) with a
narrow field of view from far away, so the picture is nearly a parallel projection like
the policy's 2D image; the background turns into a grey intensifier field.

What shows is decided by the workbench from three switches (``visible_in_xray``):
  - without contrast only the instruments show, as in the real procedure;
  - "contrast / road map" adds the vessels as a faint dark shadow;
  - "what the policy gets" marks the tracked wire points and the target.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .frame import SceneInfo, fluoro_rotation, project_2d
from .scene import MM

FOV_DEG = 3.0
WIRE_DARK = (25, 25, 25)
CONTRAST_LOOK = ((35, 35, 35), 0.55)
LIGHT = ("studio", 0.7, 0.12)      # environment map, intensity normally and in X-ray mode (flat, no shine)
# per layer: shown in X-ray mode? (None = follow its own checkbox)
ALWAYS_HIDDEN = ("body", "centerlines", "other_centerlines", "radius_tubes", "trail", "force")
POLICY_ONLY = ("target", "tracked")


def visible_in_xray(layer: str, contrast: bool, policy_marks: bool) -> Optional[bool]:
    """In X-ray mode: True / False = shown / hidden whatever its checkbox says; None = its checkbox."""
    if layer in ALWAYS_HIDDEN:
        return False
    if layer in POLICY_ONLY:
        return policy_marks
    if layer == "vessel":
        return None if contrast else False
    return None


def field_image(height: int = 270, width: int = 480) -> np.ndarray:
    """Grey intensifier field, darker towards the edge (uint8 RGB). The viewer shows background
    images darker than their values (colour conversion), hence the light numbers."""
    y, x = np.mgrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    r = np.sqrt(x ** 2 + y ** 2)
    grey = 235 - 60 * np.clip(r - 0.55, 0, None) / 0.85
    return np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2)


def region_of_interest(info: SceneInfo) -> Tuple[np.ndarray, float]:
    """Centre (mm, vessel frame) and half-height (mm) of the image: the training branches
    (all centerlines when none are named), with a margin."""
    lines = [v for k, v in info.centerlines.items() if info.is_task_branch(k)] or list(info.centerlines.values())
    points = np.concatenate(lines)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    image = project_2d(points, info.image_rot_zx_deg)
    half = float((image.max(axis=0) - image.min(axis=0)).max()) / 2 + 20.0
    return center, half


def xray_camera(info: SceneInfo) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(position, look_at, up) in scene units and fov (rad). Near and far stay as they are: the
    background image is drawn at a fixed distance in front of the camera and a far-off near plane
    would clip it."""
    rot = fluoro_rotation(info.image_rot_zx_deg)
    forward, up = rot.T @ np.array([0.0, 1.0, 0.0]), rot.T @ np.array([0.0, 0.0, 1.0])
    center, half = region_of_interest(info)
    fov = np.radians(FOV_DEG)
    distance = half / np.tan(fov / 2)
    position = center - forward * distance
    return position * MM, center * MM, up, float(fov)


class XrayMode:
    """Turns the scene background and every client's camera to X-ray and back."""

    def __init__(self, server) -> None:
        self.server = server
        self.on = False
        self._saved: Dict[int, float] = {}   # client id -> fov before X-ray mode

    def enter(self, info: SceneInfo, clients: Sequence) -> None:
        self.on = True
        self.server.scene.set_background_image(field_image(), format="png")
        self.server.scene.configure_environment_map(LIGHT[0], environment_intensity=LIGHT[2])
        for client in clients:
            self.apply_camera(client, info)

    def apply_camera(self, client, info: SceneInfo) -> None:
        position, look_at, up, fov = xray_camera(info)
        try:
            if client.client_id not in self._saved:
                self._saved[client.client_id] = client.camera.fov
        except AssertionError:  # the client has not reported its camera yet
            pass
        with client.atomic():
            client.camera.fov = fov
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(look_at)

    def leave(self, clients: Sequence) -> None:
        self.on = False
        self.server.scene.set_background_image(None)
        self.server.scene.configure_environment_map(LIGHT[0], environment_intensity=LIGHT[1])
        for client in clients:
            saved = self._saved.pop(client.client_id, None)
            if saved is not None:
                client.camera.fov = saved
        self._saved.clear()
