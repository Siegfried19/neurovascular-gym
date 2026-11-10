"""Gymnasium view of a stEVE env assembled by ``builder.build_env``.

Reset/step semantics follow the reference ``StEVEEnv`` + ``GeometryForceEnv``:
the observation is every configured term flattened in YAML order (then
optionally clipped), the action is normalized, and labels are reported in
``info["labels"]``.

``monitors`` are the env's shared per-step parts (``env.monitors["phase_detector"]``;
builder.py). ``set_cutoff(k)`` hands a cutoff to every part that takes one (the
two-device task's ``PhaseCutoff``); ``reset(options={"cutoff": k})`` does the same
before the reset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


class GuidewireEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, *, eve_env, simulation, observation_terms: Sequence[str],
                 labels: Sequence[Any], action_map, observation_clip: Optional[float],
                 max_episode_steps: Optional[int], render_mode: Optional[str] = None,
                 monitors: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._env = eve_env
        self.simulation = simulation
        self.intervention = eve_env.intervention
        self.labels = list(labels)
        self.monitors: Dict[str, Any] = dict(monitors or {})
        self._action_map = action_map
        self.observation_clip = observation_clip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate label names: {names}")

        terms = self._env.observation.observations
        self.observation_layout: List[Tuple[str, int]] = []
        self.observation_names: List[str] = []
        for name in observation_terms:
            term = terms[name]
            size = int(np.prod(term.space.shape))
            self.observation_layout.append((name, size))
            per_dim = getattr(term, "feature_names", None)
            self.observation_names.extend(per_dim if per_dim else
                                          [f"{name}[{i}]" for i in range(size)])
        dim = sum(size for _, size in self.observation_layout)
        bound = np.inf if observation_clip is None else float(observation_clip)
        self.observation_space = gym.spaces.Box(
            low=np.full(dim, -bound, dtype=np.float32),
            high=np.full(dim, bound, dtype=np.float32), dtype=np.float32)

        self._raw_shape = self._env.action_space.shape
        self._raw_low = np.asarray(self._env.action_space.low, dtype=np.float32).reshape(-1)
        self._raw_high = np.asarray(self._env.action_space.high, dtype=np.float32).reshape(-1)
        self.action_space = action_map.space(self._raw_low, self._raw_high)

    @property
    def label_names(self) -> List[str]:
        return [label.name for label in self.labels]

    @property
    def raw_action_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._raw_low.copy(), self._raw_high.copy()

    def term(self, name: str):
        """The stEVE observation object behind one observation term."""
        return self._env.observation.observations[name]

    @property
    def action_map(self):
        return self._action_map

    def _parts(self):
        """Every MDP part, flattened out of stEVE's combinations."""
        def flat(part, children):
            nested = getattr(part, children, None)
            if nested is None:
                yield part
            else:
                for child in nested:
                    yield from flat(child, children)
        yield from self.monitors.values()
        yield from flat(self._env.reward, "rewards")
        yield from flat(self._env.terminal, "terminals")
        yield from flat(self._env.truncation, "truncations")
        yield from flat(self._env.info, "infos")
        yield self._action_map

    def set_cutoff(self, cutoff: int) -> None:
        """Hand the episode cutoff to every part that takes one (two-device ``PhaseCutoff``)."""
        takers = [part for part in self._parts() if callable(getattr(part, "set_cutoff", None))]
        if not takers:
            raise AttributeError("no part of this env takes a cutoff (no PhaseCutoff in env.truncation)")
        for part in takers:
            part.set_cutoff(int(cutoff))

    def _flatten(self, observation: Dict[str, Any]) -> np.ndarray:
        flat = np.concatenate([np.asarray(observation[name], dtype=np.float32).reshape(-1)
                               for name, _ in self.observation_layout])
        if not np.all(np.isfinite(flat)):
            raise FloatingPointError("stEVE returned a non-finite observation")
        if self.observation_clip is not None:
            # Numerical round-off in the episode normalizer can slightly exceed bounds.
            flat = np.clip(flat, -self.observation_clip, self.observation_clip)
        return flat.astype(np.float32, copy=False)

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # MonoPlaneStatic does not forward its reset seed to the simulation;
            # seed its generator so the initial wire rotation is reproducible
            # (same workaround as the reference StEVEEnv).
            self.simulation._rng = np.random.default_rng(seed)  # pylint: disable=protected-access
        self.simulation.simulation_error = False
        if options and options.get("cutoff") is not None:
            self.set_cutoff(int(options["cutoff"]))
            options = {k: v for k, v in options.items() if k != "cutoff"} or None
        observation, info = self._env.reset(seed=seed, options=options)
        for label in self.labels:        # every reset: the simulator may have rebuilt the scene
            label.attach(self)
        out = dict(info)
        out["is_success"] = bool(out.get("is_success", False))
        out["simulation_error"] = bool(self.simulation.simulation_error)
        return self._flatten(observation), out

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw = self._action_map.to_raw(action, self._raw_low, self._raw_high).reshape(self._raw_shape)
        for label in self.labels:
            label.begin_step()
        try:
            observation, reward, terminated, truncated, info = self._env.step(raw)
            values, details = {}, {}
            for label in self.labels:
                values[label.name], details[label.name] = label.finish_step()
        except BaseException:
            for label in self.labels:
                label.abort_step()
            raise
        reward = float(reward)
        if not np.isfinite(reward):
            raise FloatingPointError("stEVE returned a non-finite reward")
        out = dict(info)
        out["is_success"] = bool(out.get("is_success", terminated))
        out["simulation_error"] = bool(self.simulation.simulation_error)
        out["raw_action"] = raw.reshape(-1).copy()
        out["labels"] = values
        out["label_details"] = details
        return self._flatten(observation), reward, bool(terminated), bool(truncated), out

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "human":
            return None
        return self._env.render()

    def close(self) -> None:
        # Close components separately so a failed visual reset does not mask
        # the original error (same as the reference StEVEEnv.close).
        self.intervention.close()
        visualisation = self._env.visualisation
        if getattr(visualisation, "_pygame", None) is not None:
            visualisation.close()
