"""The instrument window: what each instrument looks like at rest, and its numbers.

``instrument_info`` reads an upstream stEVE ``Instrument`` (a straight ``base_section``
then a ``tip_section``): every section's length, diameters, stiffness and density, and
the rest shape (straight shaft, then the tip section's own shape, built the way stEVE
builds it). Everything else here is pure numpy / strings, so it is testable without
stEVE or viser.

Units: mm, MPa, and SOFA's mass density in t/mm^3 (shown as g/cm^3: x 1e9).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .i18n import tr

SHAFT_COLOR, TIP_COLOR, INK = "#888780", "#E24B4A", "currentColor"


@dataclass(eq=False)
class SectionInfo:
    role: str                    # "shaft" | "tip"
    length_mm: float
    diameter_outer_mm: float
    diameter_inner_mm: float
    young_modulus_mpa: float
    mass_density: float          # t/mm^3 (SOFA's N-mm-s units)

    @property
    def mass_g(self) -> float:
        area = np.pi / 4 * (self.diameter_outer_mm ** 2 - self.diameter_inner_mm ** 2)
        return float(self.mass_density * area * self.length_mm * 1e6)


@dataclass(eq=False)
class InstrumentInfo:
    name: str
    kind: str                                # stEVE class name, e.g. "Angled"
    sections: List[SectionInfo]              # base to tip
    rest_shape_mm: np.ndarray                # (N, 3) base -> tip; the shaft ends at the origin
    shape_params: Dict[str, float] = field(default_factory=dict)   # e.g. tip_radius, tip_angle_deg

    @property
    def length_mm(self) -> float:
        return float(sum(s.length_mm for s in self.sections))

    @property
    def mass_g(self) -> float:
        return float(sum(s.mass_g for s in self.sections))

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "shape_params": self.shape_params,
                "sections": [vars(s) for s in self.sections],
                "rest_shape_mm": np.asarray(self.rest_shape_mm).round(4).tolist()}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "InstrumentInfo":
        return cls(name=data["name"], kind=data["kind"], shape_params=dict(data["shape_params"]),
                   sections=[SectionInfo(**s) for s in data["sections"]],
                   rest_shape_mm=np.asarray(data["rest_shape_mm"], dtype=np.float64).reshape(-1, 3))


# ----------------------------------------------------------- read stEVE --
def _section(role: str, section) -> SectionInfo:
    return SectionInfo(role=role, length_mm=float(section.length),
                       diameter_outer_mm=float(section.diameter_outer),
                       diameter_inner_mm=float(section.diameter_inner),
                       young_modulus_mpa=float(section.young_modulus),
                       mass_density=float(section.mass_density))


def _tip_shape(section) -> np.ndarray:
    """The tip section's rest shape from the origin along +x, as stEVE builds it."""
    if hasattr(section, "mesh_elements"):
        from eve.intervention.instrument.util.meshing import create_shape_point_cloud
        return np.asarray(create_shape_point_cloud(section.mesh_elements)[0], dtype=np.float64)
    if hasattr(section, "spire_radius"):
        r, h = float(section.spire_radius), float(section.spire_height)
        t = np.linspace(0.0, np.radians(float(section.spire_angle_deg)), 200)
        return np.stack([r * np.sin(t), r * (1 - np.cos(t)), h * t / (2 * np.pi)], axis=1)
    return np.array([[0.0, 0.0, 0.0], [float(section.length), 0.0, 0.0]])


def instrument_info(instrument) -> Optional[InstrumentInfo]:
    """Upstream stEVE instrument -> InstrumentInfo; None for anything without sections."""
    tip = getattr(instrument, "tip_section", None)
    if tip is None:
        return None
    base = getattr(instrument, "base_section", None)
    sections = ([_section("shaft", base)] if base is not None else []) + [_section("tip", tip)]
    shape = _tip_shape(tip)
    if base is not None:
        shape = np.vstack([[[-float(base.length), 0.0, 0.0]], shape])
    params = {k: float(getattr(instrument, k)) for k in ("tip_radius", "tip_angle_deg")
              if isinstance(getattr(instrument, k, None), (int, float))}
    return InstrumentInfo(name=str(getattr(instrument, "name", "")), kind=type(instrument).__name__,
                          sections=sections, rest_shape_mm=shape, shape_params=params)


# ------------------------------------------------------------- drawing --
def _plane(points: np.ndarray) -> np.ndarray:
    """2D coordinates in the plane the shape bends in (x stays along the shaft)."""
    p = np.asarray(points, dtype=np.float64)
    off = p[:, 1:] - p[:, 1:].mean(axis=0)
    if np.allclose(off, 0):
        return p[:, :2]
    _, _, vt = np.linalg.svd(off, full_matrices=False)
    return np.stack([p[:, 0], p[:, 1:] @ vt[0]], axis=1)


def _cumlen(points: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def instrument_svg(info: InstrumentInfo, lang: str = "zh", *, distal_mm: Optional[float] = None,
                   height: int = 200) -> str:
    """The distal end at rest, to scale (line width ~ diameter), plus the whole length as a bar."""
    shape = np.asarray(info.rest_shape_mm, dtype=np.float64)
    s = _cumlen(shape)
    total = s[-1]
    tip_len = info.sections[-1].length_mm
    distal = float(distal_mm if distal_mm is not None else min(total, tip_len + 25.0))
    keep = s >= total - distal - 1e-9
    start = np.interp(total - distal, s, np.arange(len(s)))
    i0 = int(np.floor(start))
    pts = np.vstack([shape[i0] + (shape[min(i0 + 1, len(shape) - 1)] - shape[i0]) * (start - i0), shape[keep]])
    sp = _cumlen(pts) + (total - distal)
    xy = _plane(pts)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = max(float((hi - lo).max()), 10.0)
    pad = 0.12 * span
    lo, hi = lo - pad, hi + pad
    w_mm, h_mm = hi - lo
    k = 100.0 / max(w_mm, h_mm)

    def path(points: np.ndarray, color: str, width_mm: float) -> str:
        if len(points) < 2:
            return ""
        d = " ".join(f"{(x - lo[0]) * k:.2f},{(hi[1] - y) * k:.2f}" for x, y in points)
        return (f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{max(width_mm * k, 0.8):.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')

    boundary = total - tip_len
    shaft_part = xy[sp <= boundary + 1e-9]
    tip_part = xy[sp >= boundary - 1e-9]
    body = [path(shaft_part, SHAFT_COLOR, info.sections[0].diameter_outer_mm) if len(info.sections) > 1 else "",
            path(tip_part, TIP_COLOR, info.sections[-1].diameter_outer_mm)]
    # scale bar, 10 mm
    bar = 10.0 * k
    vb_w, vb_h = w_mm * k, h_mm * k
    body.append(f'<line x1="2" y1="{vb_h - 2:.1f}" x2="{2 + bar:.1f}" y2="{vb_h - 2:.1f}" stroke="{INK}" '
                f'stroke-width=".8" opacity=".7"/><text x="2" y="{vb_h - 4:.1f}" font-size="4" fill="{INK}" '
                f'opacity=".7">10 mm</text>')
    distal_svg = (f'<svg viewBox="0 0 {vb_w:.1f} {vb_h:.1f}" width="100%" height="{height}" '
                  f'preserveAspectRatio="xMidYMid meet" style="background:rgba(128,128,128,.08);'
                  f'border-radius:6px">{"".join(body)}</svg>')

    # whole length as a bar: shaft grey, tip section red, to scale
    parts = []
    x = 0.0
    for sec in info.sections:
        width = 100.0 * sec.length_mm / total
        color = TIP_COLOR if sec.role == "tip" else SHAFT_COLOR
        parts.append(f'<rect x="{x:.2f}" y="1" width="{max(width, 0.4):.2f}" height="4" fill="{color}"/>')
        x += width
    whole_svg = (f'<svg viewBox="0 0 100 6" width="100%" height="14" preserveAspectRatio="none">'
                 f'{"".join(parts)}</svg>')
    caption = tr("instrument_distal", lang, mm=f"{distal:.0f}")
    whole = tr("instrument_whole", lang, mm=f"{total:.0f}")
    return (f'<div style="font-size:11px;opacity:.6;margin:2px 0">{html.escape(caption)}</div>{distal_svg}'
            f'<div style="font-size:11px;opacity:.6;margin:6px 0 2px">{html.escape(whole)}</div>{whole_svg}')


_TABLE = 'style="width:100%;border-collapse:collapse;font-size:12px"'
_CELL = 'style="padding:2px 4px;white-space:nowrap"'


def _grams(g: float) -> str:
    return f"{g / 1000:,.1f} kg" if g >= 1000 else f"{g:.2f} g"


def _g_per_cm3(mass_density: float) -> str:
    value = mass_density * 1e9                     # t/mm^3 -> g/cm^3
    return f"{value:,.0f}" if value >= 100 else f"{value:.2f}"


def instrument_table_html(info: InstrumentInfo, lang: str = "zh") -> str:
    rows = [(tr("inst_kind", lang), f"{html.escape(info.kind)} · {html.escape(info.name)}"),
            (tr("inst_length", lang), f"{info.length_mm:.1f} mm"),
            (tr("inst_mass", lang), _grams(info.mass_g))]
    if "tip_radius" in info.shape_params and "tip_angle_deg" in info.shape_params:
        rows.append((tr("inst_bend", lang), tr("inst_bend_value", lang, deg=f"{info.shape_params['tip_angle_deg']:.0f}",
                                                 r=f"{info.shape_params['tip_radius']:.1f}")))
    head = (f'<tr style="opacity:.6"><td {_CELL}></td><td {_CELL} align="right">{tr("inst_length_short", lang)} mm</td>'
            f'<td {_CELL} align="right">{tr("inst_diameter", lang)} mm</td><td {_CELL} align="right">E GPa</td>'
            f'<td {_CELL} align="right">{tr("inst_density", lang)} g/cm³</td></tr>')
    sec_rows = []
    for sec in info.sections:
        name = tr("inst_shaft" if sec.role == "shaft" else "inst_tip", lang)
        sec_rows.append(
            f'<tr><td {_CELL}>{name}</td><td {_CELL} align="right">{sec.length_mm:.1f}</td>'
            f'<td {_CELL} align="right">{sec.diameter_outer_mm:.2f}</td>'
            f'<td {_CELL} align="right">{sec.young_modulus_mpa / 1e3:.1f}</td>'
            f'<td {_CELL} align="right">{_g_per_cm3(sec.mass_density)}</td></tr>')
    top = "".join(f'<tr><td {_CELL} style="opacity:.65;padding:2px 4px">{k}</td><td {_CELL}>{v}</td></tr>'
                  for k, v in rows)
    return (f"<table {_TABLE}>{top}</table><table {_TABLE} style='margin-top:6px'>{head}{''.join(sec_rows)}</table>"
            f'<div style="font-size:11px;opacity:.6;margin-top:4px">{html.escape(tr("inst_units_hint", lang))}</div>')


def instruments_html(infos: List[InstrumentInfo], lang: str = "zh") -> str:
    if not infos:
        return f'<div style="opacity:.6">{tr("no_instruments", lang)}</div>'
    blocks = []
    for info in infos:
        blocks.append(instrument_svg(info, lang) + instrument_table_html(info, lang))
    return '<hr style="opacity:.2">'.join(blocks)
