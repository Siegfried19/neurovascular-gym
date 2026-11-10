"""Everything we change in the simulator (see README.md here and docs/simulator_changes.md).

The Python fixes to stEVE live in this package. They are registered under
stEVE's own class names, so a config that says ``type: SofaBeamAdapter`` or
``type: CenterlineRandom`` gets the fixed version. The BeamAdapter fix is a
patch plus build script in ``beamadapter/``.

Only active with upstream stEVE (workspace ``stEVE/``, SOFA 25.12, conda env
``guidewire_rl``), recognised by its ``eve.intervention.instrument`` module.
With Wilson's eve (old env, G13 reproduction) nothing is registered and
stEVE's classes are used unchanged.
"""

import eve

UPSTREAM = hasattr(eve.intervention, "instrument")
