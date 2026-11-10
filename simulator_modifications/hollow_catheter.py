"""Catheters as hollow tubes (docs/simulator_changes.md, problem 2).

stEVE builds each rod section with only ``radius = diameter_outer / 2``.
BeamAdapter's ``innerRadius`` defaults to 0, so every catheter bent like a
solid rod: second moment pi/64 * OD^4 instead of pi/64 * (OD^4 - ID^4), i.e.
1.3x (5F) to 2.8x (8F sheath) too stiff. Guidewires (inner diameter 0) are
unchanged.

The fix is the lab's (reference/surgery-all-in-one/51-legobot-ros2/legobot_sofa/
legobot_sofa/twin.py:138-147): let stEVE build the instruments, then give each
section its inner radius before SOFA initialises the scene.

Not registered itself: ``type: SofaBeamAdapter`` resolves to ``device_contact.SofaBeamAdapter``,
which builds on ``speedups.SofaBeamAdapter``, which builds on this class.
"""

from __future__ import annotations

from . import UPSTREAM

if UPSTREAM:
    import eve

    class SofaBeamAdapter(eve.intervention.simulation.SofaBeamAdapter):
        """stEVE's ``SofaBeamAdapter`` with hollow instrument sections."""

        def _add_instruments(self, instruments, insertion_point, insertion_direction):
            super()._add_instruments(instruments, insertion_point, insertion_direction)
            for instrument in instruments:
                node = self.root.getChild("topolines_" + instrument.name)
                for prefix, section in (("base_section_", instrument.base_section),
                                        ("tip_section_", instrument.tip_section)):
                    inner = float(getattr(section, "diameter_inner", 0.0) or 0.0) \
                        if section is not None else 0.0
                    if inner > 0.0:
                        node.getObject(prefix + instrument.name).innerRadius.value = inner / 2.0
