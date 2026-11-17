"""Cordis Tempo Aqua Sidewinder II, 5F × 125 cm (catalogue SRD6109): the lab's Simmons 2 catheter.

What the package label says (photo, user 2026-11-12): angiographic catheter, SIDEWINDER II TEMPO AQUA,
5F (1.65 mm) × 125 cm, guidewire up to .038″ (0.97 mm), open end, 0 side holes, 1200 psi, "SPECIAL".
"Sidewinder II" is Cordis' name for the Simmons 2 curve. The label gives no inner diameter, stiffness
or curve dimensions.

Built as ``Simmons2`` (the head traced from Qiu 2024's photographed Cordis Tempo 5F SIM 2, catalogue
451531H0, a standard Tempo, 100 cm), with the label's outer diameter:

  - outer 1.65 mm (label); inner 1.22 mm (the user's table of 2026-10-30; the label only says a
    0.97 mm wire passes);
  - bending stiffness 486 N mm^2, Qiu 2024's measurement of that Tempo, so E = 486 / I(1.65, 1.22)
    ~ 1.91 GPa. That the Tempo Aqua (hydrophilic coat, 125 cm, "SPECIAL") has the same curve and
    stiffness as the measured standard Tempo is an inference, not a measurement;
  - density 1.2e-9 t/mm^3 (plastic, an estimate), as the training's catheter;
  - push / turn limits 25 mm/s, 90 deg/s, as the training's catheter.

``type: CordisSidewinder2`` under ``instruments:``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..registry import register
from .simmons2 import Simmons2

MEASURED_EI_NMM2 = 486.0                    # Qiu 2024, Cordis 5F Simmons 2 Tempo
OUTER_MM, INNER_MM = 1.65, 1.22


def tube_second_moment(outer_mm: float, inner_mm: float) -> float:
    """Area moment of a round tube, mm^4."""
    return float(np.pi / 64.0 * (outer_mm ** 4 - inner_mm ** 4))


@register("instrument")
@dataclass
class CordisSidewinder2(Simmons2):
    """Simmons 2 head (traced), the SRD6109 label's size, the measured Tempo stiffness."""

    name: str = "catheter"
    velocity_limit: Tuple[float, float] = (25.0, 1.5707963)

    diameter_outer: float = OUTER_MM
    diameter_inner: float = INNER_MM
    young_modulus: float = MEASURED_EI_NMM2 / tube_second_moment(OUTER_MM, INNER_MM)
    mass_density: float = 1.2e-9
