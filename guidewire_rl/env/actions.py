"""Policy action → stEVE device command."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .registry import register


@register("action")
class LinearToDeviceLimits:
    """Normalized ``[-1, 1]`` per axis, mapped linearly onto the device's velocity
    limits (G13: insertion ±50 mm/s, rotation ±3.14 rad/s).

    Same arithmetic as the reference ``StEVEEnv._denormalize_action``.
    """

    def space(self, raw_low: np.ndarray, raw_high: np.ndarray) -> gym.spaces.Box:
        shape = np.asarray(raw_low).reshape(-1).shape
        return gym.spaces.Box(low=-np.ones(shape, dtype=np.float32),
                              high=np.ones(shape, dtype=np.float32), dtype=np.float32)

    def to_raw(self, action: np.ndarray, raw_low: np.ndarray,
               raw_high: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != raw_low.shape:
            raise ValueError(f"Expected action shape {raw_low.shape}, got {normalized.shape}.")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Action contains NaN or infinity")
        normalized = np.clip(normalized, -1.0, 1.0)
        return (raw_low + 0.5 * (normalized + 1.0) * (raw_high - raw_low)).astype(np.float32)
