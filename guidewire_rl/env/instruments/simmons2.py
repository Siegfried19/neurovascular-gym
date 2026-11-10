"""Simmons 2 catheter (Cordis 5F Sidewinder II / Simmons 2) with its head traced from a real one.

The head's rest shape comes from ``data/simmons.csv``: Qiu 2024's trace of a photographed Cordis 5F
Simmons 2 (Zenodo 10.5281/zenodo.11203294, ``2.5D Aortic Arch/Herniation/simmons.csv``, kept unchanged;
provenance, licence and checksums in ``data/README.md``). The trace runs from a piece of straight shaft
(first row) round the reversed curve to the catheter tip (last row, the file's own ``Tip`` landmark).

How the head is rebuilt (``head_pieces``), all in the trace's plane:

1. Points in mm (the file's columns are µm), y flipped so the plane is right-handed.
2. Tangent direction along the trace, smoothed with a Gaussian of ``SMOOTH_MM`` over arc length.
3. The head starts where the shaft stops being straight: the last turning point of the tangent
   before the reversed curve (the shaft in the photo wanders by a few degrees; from here on the
   tangent turns one way only). That is 51.6 mm from the first row and leaves 114 mm of head.
4. The head is cut into pieces of ``PIECE_MM`` and each piece becomes one circular arc that turns by
   exactly what the smoothed tangent turns over it (a straight piece if it turns less than
   ``STRAIGHT_BELOW_DEG``). The pieces start along +x, the direction the shaft is built in.

stEVE's ``MeshSection`` then lays the arcs out (``ArcMeshElement`` / ``StraightMeshElement``) and
BeamAdapter reads that polyline as the head's rest shape. Against the trace's own points the result
is off by 0.05 mm on average and 0.2 mm at most (``tests/test_instruments.py``).

The catheter is built as stEVE builds its ``Simmons4Bends``: a straight shaft first, then the head as a
mesh section (docs/simulator_changes.md, problem 17: a straight section anywhere but first gets its
rest shape displaced). The inner diameter reaches BeamAdapter through the hollow-catheter fix
(simulator_modifications/hollow_catheter.py).

Material defaults (5F, user's table 2026-10-30): outer 1.67 mm, inner 1.22 mm, E 1.78e3 MPa, shaft
500 mm, density 1.2e-9 t/mm^3. The density is an assumed polymer value (about 1.2 g/cm^3), not a
measurement; shaft and head share one material.

Upstream stEVE only (``eve.intervention.instrument``). ``type: Simmons2`` under ``instruments:``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from eve.intervention.instrument.instrument import (ArcMeshElement, Instrument, MeshSection,
                                                    SpireSection, StraightMeshElement,
                                                    StraightSection)
from scipy.ndimage import gaussian_filter1d

from ..registry import register

TRACE_PATH = Path(__file__).with_name("data") / "simmons.csv"
TRACE_MD5 = "6978de93499064bc16f3c3eb75378b62"

SAMPLE_MM = 0.05            # the trace's own spacing; it is resampled at this spacing
SMOOTH_MM = 0.5             # Gaussian sigma for the tangent direction
SEARCH_BEFORE_MM = 85.0     # the reversed curve is past this arc length from the first row
PIECE_MM = 2.0              # length of one arc piece
STRAIGHT_BELOW_DEG = 0.05   # a piece turning less than this is written as a straight piece


def load_trace(path: Union[str, Path] = TRACE_PATH) -> np.ndarray:
    """The traced centreline, (n, 2) in mm, from the first row (shaft) to the tip; y flipped."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    xy = np.array([[float(row[4]), float(row[5])] for row in rows]) / 1000.0
    xy[:, 1] *= -1.0
    return xy


def arc_length(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _resampled(trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = arc_length(trace)
    grid = np.arange(0.0, s[-1] + 1e-9, SAMPLE_MM)
    return grid, np.c_[np.interp(grid, s, trace[:, 0]), np.interp(grid, s, trace[:, 1])]


def _tangent_angle(points: np.ndarray, sigma_mm: float) -> np.ndarray:
    smooth = gaussian_filter1d(points, sigma_mm / SAMPLE_MM, axis=0, mode="nearest")
    tangent = np.gradient(smooth, axis=0)
    return np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))


def head_start(trace: np.ndarray) -> float:
    """Arc length (mm from the first row) where the head starts (module docstring, step 3)."""
    grid, points = _resampled(trace)
    turn = _tangent_angle(points, 2.0)            # coarser smoothing: only to find the start
    turn = (turn - turn[0]) * np.sign(turn[int(round(100.0 / SAMPLE_MM))] - turn[0])
    end = int(round(SEARCH_BEFORE_MM / SAMPLE_MM))
    lows = [i for i in range(1, end) if turn[i] <= turn[i - 1] and turn[i] <= turn[i + 1]]
    return float(grid[lows[-1]])


@lru_cache(maxsize=None)
def head_pieces(path: str = str(TRACE_PATH), piece_mm: float = PIECE_MM) -> Tuple[Tuple[float, float], ...]:
    """The head as ((length mm, turn deg), ...), from the shaft end to the tip."""
    trace = load_trace(path)
    grid, points = _resampled(trace)
    angle = _tangent_angle(points, SMOOTH_MM)
    start = int(round(head_start(trace) / SAMPLE_MM))
    head_s = grid[start:] - grid[start]
    head_angle = angle[start:]
    n = max(1, int(round(head_s[-1] / piece_mm)))
    length = head_s[-1] / n
    bounds = np.interp(np.arange(n + 1) * length, head_s, head_angle)
    return tuple((float(length), float(np.degrees(turn))) for turn in np.diff(bounds))


def mesh_elements(pieces, resolution: float = 0.1) -> List[Union[StraightMeshElement, ArcMeshElement]]:
    elements = []
    for length, turn_deg in pieces:
        if abs(turn_deg) < STRAIGHT_BELOW_DEG:
            elements.append(StraightMeshElement(length))
        else:
            radius = length / np.radians(abs(turn_deg))
            elements.append(ArcMeshElement(radius, turn_deg, 0.0, resolution))
    return elements


def head_trace_in_mesh_frame(path: Union[str, Path] = TRACE_PATH) -> np.ndarray:
    """The trace's head points (raw, not smoothed) moved into the frame the mesh is built in:
    head start at the origin, the head's first direction along +x, plane z = 0. (n, 3) mm."""
    trace = load_trace(path)
    grid, points = _resampled(trace)
    angle = _tangent_angle(points, SMOOTH_MM)
    s0 = head_start(trace)
    i0 = int(round(s0 / SAMPLE_MM))
    origin, phi = points[i0], angle[i0]
    raw = trace[arc_length(trace) >= s0 - 1e-9]
    c, s = np.cos(-phi), np.sin(-phi)
    local = (raw - origin) @ np.array([[c, -s], [s, c]]).T
    return np.c_[local, np.zeros(len(local))]


@register("instrument")
@dataclass
class Simmons2(Instrument):
    """5F Simmons 2: a straight shaft and the traced head (module docstring)."""

    name: str = "simmons2"
    velocity_limit: Tuple[float, float] = (50, 3.14)

    diameter_outer: float = 1.67
    diameter_inner: float = 1.22
    young_modulus: float = 1.78e3
    mass_density: float = 1.2e-9
    poisson_ratio: float = 0.49
    collis_edges_per_mm: float = 2
    visu_edges_per_mm: float = 0.5

    base_length: float = 500
    base_collis_edges_per_mm: float = 0.1
    base_visu_edges_per_mm: float = 0.5

    piece_mm: float = PIECE_MM
    arc_mesh_resolution: float = 0.1
    color: Tuple[float, float, float] = (30, 144, 255)

    tip_section: Union[StraightSection, SpireSection, MeshSection] = field(
        init=False, repr=False, default=None)
    base_section: Optional[StraightSection] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        material = dict(diameter_outer=self.diameter_outer, diameter_inner=self.diameter_inner,
                        poisson_ratio=self.poisson_ratio, young_modulus=self.young_modulus,
                        mass_density=self.mass_density)
        self.base_section = StraightSection(
            length=self.base_length, visu_edges_per_mm=self.base_visu_edges_per_mm,
            collis_edges_per_mm=self.base_collis_edges_per_mm, **material)
        self.tip_section = MeshSection(
            visu_edges_per_mm=self.visu_edges_per_mm, collis_edges_per_mm=self.collis_edges_per_mm,
            mesh_elements=mesh_elements(head_pieces(str(TRACE_PATH), float(self.piece_mm)),
                                        self.arc_mesh_resolution),
            **material)
