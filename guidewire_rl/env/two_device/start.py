"""How an episode starts (``env.start``; stEVE calls it after the intervention's reset, before the
observations, rewards and the referee are reset, so none of this counts as an episode step).

``WireFirst``: the devices back at the insertion point (stEVE's ``InsertionPoint``), then the wire
alone pushed in to ``wire_mm`` (40) at full speed (5.28 mm a step: stEVE runs whole substeps, 33 of 4 ms),
the last step slowed to end there, 8 steps; the catheter stays at 0. Why (docs/simulator_changes.md,
problem 23, 2026-11-04): the insertion point is on the centerline of the descending aorta (radius
about 11.2 mm) and the device shafts are held there;
the hook reaches about 19 mm to one side, so a hook out of the catheter in the first 40 mm cannot
open and presses on the wall (3-7 N, 9 N at an impact; random pushes, 124 of 535 steps with the
catheter tip that far in had more than 2 N). Started over the wire, the hook comes out on the
wire's shaft (the wire's last 30 mm are its soft part) and stays straight. Keeping the wire ahead
after that is the policy's business (no shell rule; user decision 2026-11-04): with random actions
from this opening (the same 10 x 300 steps), 12 of 325 steps with the catheter tip that far in had
more than 2 N, each after the wire had been pulled back to 2.5-11 mm ahead (the hook again on the
soft part, or out). Whether the hook is straight on the shaft is inferred from the forces, not
looked at. The executed action is zeroed afterwards, so the episode's
first observation shows no action taken yet.
"""

from __future__ import annotations

import numpy as np
from eve.start import Start

from ..registry import register
from .phases import WIRE


@register("start")
class WireFirst(Start):
    def __init__(self, intervention, wire_mm: float = 40.0, max_steps: int = 20) -> None:
        self.intervention = intervention
        self.wire_mm = float(wire_mm)
        self.max_steps = int(max_steps)

    def reset(self, episode_nr: int = 0) -> None:
        intervention = self.intervention
        intervention.reset_instruments()
        limits = np.abs(np.asarray(intervention.velocity_limits, dtype=np.float64))
        step_s = 1.0 / float(intervention.fluoroscopy.image_frequency)
        normalized = bool(getattr(intervention, "normalize_action", False))
        simulation = getattr(intervention, "simulation", None)
        dt = getattr(simulation, "dt_simulation", None)
        if dt:                                   # stEVE runs whole substeps: 33 x 4 ms of 133.3 ms
            step_s = int(step_s / float(dt)) * float(dt)
        for _ in range(self.max_steps):
            remaining = self.wire_mm - float(np.asarray(intervention.instrument_lengths_inserted)[WIRE])
            if remaining <= 1e-6 or getattr(simulation, "simulation_error", False):
                break
            speed = min(limits[WIRE, 0], remaining / step_s)
            action = np.zeros_like(limits)
            action[WIRE, 0] = speed / limits[WIRE, 0] if normalized else speed
            intervention.step(action)
        intervention.last_action = np.zeros_like(np.asarray(intervention.last_action, dtype=np.float64))
