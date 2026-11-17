"""AngioDynamics Accu-Vu Omni Flush, 5F × 100 cm, 21 radiopaque markers (catalogue 13709703).

What the package label says (photo, user 2026-11-12; the FDA device record H787137097030 agrees):
5F (1.8 mm) × 100 cm, non-braided, 21 radiopaque markers 1 cm apart, 6 side holes, guidewire up to
0.035″ (0.89 mm), 22 mL/s, 1050 psi. The maker's stated uses: flush aortography, run-off studies,
crossing the aortic bifurcation. No curve drawing, inner diameter or stiffness was found
(WORK_LOG 2026-11-12 21:30).

So the head and the material are ESTIMATES (user 2026-11-12: build it on the estimates for now):

  - head: straight ``lead`` 10 mm, then an arc of radius 7.5 mm turning 180 degrees (a loop
    ~15 mm across, after the only number found, for another product: the 4F Soft-Vu Omni Flush,
    "top of the shape to the tip end approximately 15 mm"), then a straight ``tip`` of 5 mm;
  - outer 1.8 mm (label); inner 0.97 mm (0.038″, a lumen that passes the label's 0.035″ wire);
  - E 1.78 GPa, the measured Cordis Tempo's (Qiu 2024: 486 N mm^2 at 1.67 / 1.22 mm), taken as the
    same nylon; with this tube's thicker wall that is ~840 N mm^2;
  - density 1.2e-9 t/mm^3, push / turn limits 25 mm/s, 90 deg/s, as the training's catheter.

Side holes and markers are not modelled. Built as the lab's ``HookCatheter`` (shaft first, the head
one mesh section). Replace the head with a trace once the catheter is photographed flat.
``type: AccuVuOmniFlush`` under ``instruments:``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from eve.intervention.instrument.instrument import (ArcMeshElement, Instrument, MeshSection,
                                                    SpireSection, StraightMeshElement,
                                                    StraightSection)

from ..registry import register


@register("instrument")
@dataclass
class AccuVuOmniFlush(Instrument):
    """5F flush catheter with an estimated 180-degree loop at the tip (module docstring)."""

    name: str = "catheter"
    velocity_limit: Tuple[float, float] = (25.0, 1.5707963)
    length: float = 450.0

    lead: float = 10.0
    radius: float = 7.5
    angle_deg: float = 180.0
    tip: float = 5.0

    diameter_outer: float = 1.8
    diameter_inner: float = 0.97
    young_modulus: float = 1.78e3
    mass_density: float = 1.2e-9
    poisson_ratio: float = 0.49
    collis_edges_per_mm: float = 1.0
    visu_edges_per_mm: float = 0.5
    base_collis_edges_per_mm: float = 0.1
    arc_mesh_resolution: float = 0.1
    color: Tuple[float, float, float] = (30, 144, 255)

    tip_section: Union[StraightSection, SpireSection, MeshSection] = field(
        init=False, repr=False, default=None)
    base_section: Optional[StraightSection] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        material = dict(diameter_outer=self.diameter_outer, diameter_inner=self.diameter_inner,
                        poisson_ratio=self.poisson_ratio, young_modulus=self.young_modulus,
                        mass_density=self.mass_density, visu_edges_per_mm=self.visu_edges_per_mm)
        self.tip_section = MeshSection(
            collis_edges_per_mm=self.collis_edges_per_mm,
            mesh_elements=[StraightMeshElement(self.lead),
                           ArcMeshElement(self.radius, self.angle_deg, 0.0, self.arc_mesh_resolution),
                           StraightMeshElement(self.tip)],
            **material)
        shaft = self.length - self.tip_section.length
        if shaft <= 0:
            raise ValueError(f"length {self.length} mm leaves no shaft behind a "
                             f"{self.tip_section.length:.1f} mm head")
        self.base_section = StraightSection(
            length=shaft, collis_edges_per_mm=self.base_collis_edges_per_mm, **material)
