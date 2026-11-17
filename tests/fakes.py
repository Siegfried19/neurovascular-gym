"""A tiny stand-in for ``GuidewireEnv`` (no SOFA) so the training loop can be
tested in seconds: a point is pushed toward a goal in 2-D."""

from __future__ import annotations

import os
import time

import gymnasium as gym
import numpy as np


class FakeEnv:
    label_names = ["distal_force"]

    def __init__(self, max_episode_steps: int = 15, error_at_calls=(), crash_at_calls=(),
                 hang_at_calls=(), delay_at_calls=None, stall_step=None) -> None:
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.error_at_calls = set(error_at_calls)
        self.crash_at_calls = set(crash_at_calls)     # the process dies, like a SOFA segfault
        self.hang_at_calls = set(hang_at_calls)
        self.delay_at_calls = dict(delay_at_calls or {})   # call number -> seconds
        self.calls = 0
        self.stall_step = stall_step          # with end_when_stalled() asked for: the episode ends there
        self._stall_requested = self._stall_rule = False

    def end_when_stalled(self) -> bool:       # as GuidewireEnv: for the next episode only
        self._stall_requested = True
        return True

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.goal]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(-0.5, 0.5, 2)
        self.goal = rng.uniform(-0.5, 0.5, 2)
        self.t = 0
        self._stall_rule, self._stall_requested = self._stall_requested, False
        return self._obs(), {"is_success": False, "simulation_error": False}

    def step(self, action):
        self.calls += 1
        if self.calls in self.error_at_calls:
            raise RuntimeError("fake simulator failure")
        if self.calls in self.crash_at_calls:
            os._exit(1)
        if self.calls in self.hang_at_calls:
            time.sleep(3600)
        if self.calls in self.delay_at_calls:
            time.sleep(self.delay_at_calls[self.calls])
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        before = np.linalg.norm(self.goal - self.pos)
        self.pos = np.clip(self.pos + 0.2 * action, -1, 1)
        self.t += 1
        distance = np.linalg.norm(self.goal - self.pos)
        terminated = bool(distance < 0.1)
        truncated = self.t >= self.max_episode_steps
        reward = float(before - distance) + float(terminated)
        info = {"is_success": terminated, "simulation_error": False,
                "labels": {"distal_force": float(np.abs(action).sum())}}
        if self._stall_rule and self.stall_step is not None and self.t >= self.stall_step and not terminated:
            truncated, info["stalled"] = True, True
        return self._obs(), reward, terminated, truncated, info

    def close(self) -> None:
        pass
