"""Our own instruments for stEVE scenes (upstream stEVE only).

``Simmons2`` (``simmons2.py``): 5F Simmons 2 catheter, head traced from a real one (``data/``).
Listed in a config as ``instruments: [..., {type: Simmons2}]``.
"""

from .simmons2 import Simmons2

__all__ = ["Simmons2"]
