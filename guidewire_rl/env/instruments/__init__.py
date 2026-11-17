"""Our own instruments for stEVE scenes (upstream stEVE only), one file per device.

Real devices (``catalog.py`` lists them by name, with where each number comes from):

- ``GlidewireStiff`` (``glidewire.py``): Terumo Glidewire GS3508 0.035″ stiff angled, the lab's wire;
  measured stiffness, the numbers the training configs write out by hand.
- ``CordisSidewinder2`` (``cordis_sidewinder2.py``): Cordis Tempo Aqua Sidewinder II 5F (SRD6109),
  the Simmons 2 head below with the label's size.
- ``AccuVuOmniFlush`` (``accuvu_omni_flush.py``): AngioDynamics Accu-Vu Omni Flush 5F; curve and
  material estimated.

Shapes and stand-ins:

- ``Simmons2`` (``simmons2.py``): 5F Simmons 2, head traced from a real one (``data/``).
- ``HookCatheter`` (``hook.py``): the lab's hooked catheter (the training's).

``free_space.py`` (``FreeSpace``) pushes instruments into empty space, no vessel.
Listed in a config as ``instruments: [{type: GlidewireStiff}, {type: CordisSidewinder2}]``.
"""

from .accuvu_omni_flush import AccuVuOmniFlush
from .cordis_sidewinder2 import CordisSidewinder2
from .glidewire import GlidewireStiff
from .hook import HookCatheter
from .simmons2 import Simmons2

__all__ = ["AccuVuOmniFlush", "CordisSidewinder2", "GlidewireStiff", "HookCatheter", "Simmons2"]
