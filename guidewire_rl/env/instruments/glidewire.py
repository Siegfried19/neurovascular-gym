"""Terumo Glidewire GS3508: 0.035″ × 180 cm, stiff shaft, angled tip (the guidewire the training uses now,
and the one the lab uses: user 2026-11-12, a DOTmed listing of GS3508 "0.035 x 180cm, Angled, Stiff").

The numbers are the ones the two-device configs write out by hand under ``instruments:``
(``configs/two_devices_round1.yaml``, masses from ``two_devices_real_mass.yaml``), chosen 2026-10-28
(WORK_LOG 10-28 21:05); ``tests/test_instrument_catalog.py`` keeps the two equal. The configs are not
switched to this class yet (configs change together at round two).

  - shaft: 0.89 mm, E 24.9 GPa, i.e. bending stiffness 767 N mm^2 = Qiu 2023's measured stiff
    Glidewire shaft (7.67 N cm^2, PMC10760252, Table 1);
  - soft tip: 30 mm, 0.89 mm, E 0.325 GPa, i.e. 10 N mm^2 = the same table's distal value (0.10 N cm^2);
  - tip bend: radius 12.1 mm, 72 degrees (stEVE's ``Angled`` default shape);
  - density 6.45e-9 t/mm^3 (nitinol) on both sections.

``type: GlidewireStiff`` under ``instruments:``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from eve.intervention.instrument import Angled

from ..registry import register


@register("instrument")
@dataclass
class GlidewireStiff(Angled):
    """0.035″ stiff angled Glidewire with the measured stiffness (module docstring)."""

    name: str = "guidewire"
    velocity_limit: Tuple[float, float] = (40.0, 3.1415926)
    length: float = 450.0
    tip_radius: float = 12.1
    tip_angle_deg: float = 72.0

    diameter_outer: float = 0.89
    young_modulus: float = 24.9e3
    mass_density: float = 6.45e-9

    flex_length: float = 30.0
    flex_diameter_outer: float = 0.89
    flex_young_modulus: float = 325.0
    flex_mass_density: float = 6.45e-9
