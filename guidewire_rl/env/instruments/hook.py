"""The lab's hooked catheter (``reference/surgery-all-in-one/19-phantom-sofa/arch_catheter.py``).

The lab's numbers (its ``HOOK`` dict and ``hook_instrument()``), kept as they are:

  - head (``MeshSection``, 1 collision edge per mm): straight ``lead`` 20 mm, an arc of radius
    ``radius`` 8 mm turning ``angle_deg`` 130 degrees, straight ``tip`` 8 mm (46.2 mm in all; the
    hook proper, arc + tip, is 26.2 mm, the lab's ``hook_len``);
  - shaft (``StraightSection``, 0.1 collision edges per mm) makes up the rest of ``length`` (450 mm);
  - both: outer 2.0 mm, inner 1.0 mm, E 4 GPa (4e3 MPa), Poisson 0.49, density 5e-6 (stEVE's
    default for its catheters; the lab uses it too, and at the real plastic density two devices
    in one scene go numerically unstable: todo_list/two_devices_training.md section 0).

They are simulation parameters, not a product (the lab's own words). The shaft comes first and the
head is one mesh section, as stEVE builds its ``Simmons4Bends`` (docs/simulator_changes.md,
problem 17). The inner diameter reaches BeamAdapter through the hollow-catheter fix
(``simulator_modifications/hollow_catheter.py``).

Upstream stEVE only (``eve.intervention.instrument``). ``type: HookCatheter`` under ``instruments:``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
from eve.intervention.instrument.instrument import (ArcMeshElement, Instrument, MeshSection,
                                                    SpireSection, StraightMeshElement,
                                                    StraightSection)

from ..registry import register

# arch_catheter.py, line 32 (HOOK) and hook_instrument(length=450, v_max=35)
LAB_HOOK = dict(lead=20.0, radius=8.0, angle_deg=130.0, tip=8.0, od=2.0, id=1.0, young=4e3,
                collis_edges_per_mm=1.0)


def hook_length(radius: float, angle_deg: float, tip: float) -> float:
    """Arc + straight tip, mm: how far behind the catheter tip the hook starts."""
    return float(radius * np.radians(angle_deg) + tip)


@register("instrument")
@dataclass
class HookCatheter(Instrument):
    """The lab's 2 mm catheter with a 130-degree hook at the tip (module docstring)."""

    name: str = "catheter"
    velocity_limit: Tuple[float, float] = (35.0, 3.14)
    length: float = 450.0

    lead: float = LAB_HOOK["lead"]
    radius: float = LAB_HOOK["radius"]
    angle_deg: float = LAB_HOOK["angle_deg"]
    tip: float = LAB_HOOK["tip"]

    diameter_outer: float = LAB_HOOK["od"]
    diameter_inner: float = LAB_HOOK["id"]
    young_modulus: float = LAB_HOOK["young"]
    mass_density: float = 0.000005
    poisson_ratio: float = 0.49
    collis_edges_per_mm: float = LAB_HOOK["collis_edges_per_mm"]
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

    @property
    def hook_length(self) -> float:
        return hook_length(self.radius, self.angle_deg, self.tip)
