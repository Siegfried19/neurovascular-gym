"""The parameter panel: the simulation numbers that matter most, read from the config, editable.

Each ``Param`` points at one place in the environment config. Shown values come from the
config, or the class default when the config does not set it (the defaults are checked
against stEVE / our classes in the tests). Edited values go back into the config as
overrides when the environment is rebuilt; changing the physics step also sets the force
label's expected substep count (``1 / image_frequency / dt``, truncated, as stEVE steps).
Add a row by adding a ``Param``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .i18n import Pair

Path_ = Tuple[Union[str, int], ...]


@dataclass(frozen=True)
class Param:
    key: str
    label: Pair
    unit: str
    path: Path_
    default: Any                 # class default when the config does not set it
    scale: float = 1.0           # shown = stored x scale
    step: float = 0.01
    vector: bool = False         # two numbers (C-arm angle)
    integer: bool = False
    view_only_too: bool = False  # also means something for a view-only phantom


SIM = ("env", "scene", "simulation")
WIRE = ("env", "scene", "instruments", 0)
PARAMS: List[Param] = [
    Param("friction", ("摩擦系数（导丝和管壁）", "Friction (wire on wall)"), "", SIM + ("friction",), 0.1, step=0.001),
    Param("dt", ("物理步", "Physics step"), "ms", SIM + ("dt_simulation",), 0.006, scale=1e3, step=0.5),
    Param("alarm", ("碰撞预判距离", "Collision alarm distance"), "mm", SIM + ("alarm_distance",), 0.5, step=0.1),
    Param("density", ("导丝杆身密度", "Wire shaft density"), "g/cm³", WIRE + ("mass_density",), 2.1e-5,
          scale=1e9, step=0.01),
    Param("flex_density", ("导丝软段密度", "Wire soft-section density"), "g/cm³", WIRE + ("flex_mass_density",),
          2.1e-5, scale=1e9, step=0.01),
    Param("carm", ("C 臂角度（image_rot_zx）", "C-arm angle (image_rot_zx)"), "°",
          ("env", "scene", "fluoroscopy", "image_rot_zx"), [20.0, 5.0], step=0.5, vector=True, view_only_too=True),
    Param("threshold", ("到达目标的距离", "Target reached within"), "mm", ("env", "scene", "target", "threshold"),
          5.0, step=0.5),
    Param("max_steps", ("每回合步数上限", "Steps per episode"), "", ("env", "truncation", 0, "max_steps"), 200,
          step=1, integer=True),
]


def _get(node: Any, path: Path_) -> Any:
    for key in path:
        if isinstance(key, int):
            if not isinstance(node, list) or key >= len(node):
                return None
            node = node[key]
        else:
            if not isinstance(node, Mapping) or key not in node:
                return None
            node = node[key]
    return node


def _set(node: Any, path: Path_, value: Any) -> None:
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def read(config: Mapping[str, Any], params: Sequence[Param] = PARAMS) -> Dict[str, Any]:
    """Stored (config-unit) value of every parameter."""
    out = {}
    for p in params:
        value = _get(config, p.path)
        out[p.key] = copy.deepcopy(p.default if value is None else value)
    return out


def shown(p: Param, stored: Any) -> Any:
    if p.vector:
        return tuple(float(v) * p.scale for v in stored)
    return int(stored) if p.integer else float(stored) * p.scale


def stored(p: Param, value: Any) -> Any:
    if p.vector:
        return [float(v) / p.scale for v in value]
    return int(round(float(value))) if p.integer else float(value) / p.scale


def changed(base: Mapping[str, Any], values: Mapping[str, Any], params: Sequence[Param] = PARAMS) -> List[str]:
    """Keys whose value differs from ``base`` (relative 1e-9)."""
    out = []
    for p in params:
        a, b = base[p.key], values[p.key]
        a, b = (list(a), list(b)) if p.vector else ([a], [b])
        if any(abs(float(x) - float(y)) > 1e-9 * max(1.0, abs(float(x))) for x, y in zip(a, b)):
            out.append(p.key)
    return out


def apply(config: Dict[str, Any], values: Mapping[str, Any], params: Sequence[Param] = PARAMS) -> Dict[str, Any]:
    """A copy of ``config`` with every parameter set to ``values`` (stored units)."""
    config = copy.deepcopy(config)
    for p in params:
        if p.key in values and _get(config, p.path[:-1]) is not None:
            _set(config, p.path, copy.deepcopy(values[p.key]))
    dt = _get(config, SIM + ("dt_simulation",)) or 0.006
    frequency = _get(config, ("env", "scene", "fluoroscopy", "image_frequency")) or 7.5
    for label in config.get("env", {}).get("labels") or []:
        if isinstance(label, dict) and label.get("expected_substeps") is not None:
            label["expected_substeps"] = int((1 / frequency) / dt)
    return config


def find(key: str, params: Sequence[Param] = PARAMS) -> Optional[Param]:
    return next((p for p in params if p.key == key), None)
