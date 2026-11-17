"""Gymnasium view of a stEVE env assembled by ``builder.build_env``.

Reset/step semantics follow the reference ``StEVEEnv`` + ``GeometryForceEnv``:
the observation is every configured term flattened in YAML order (then
optionally clipped), the action is normalized, and labels are reported in
``info["labels"]``.

``monitors`` are the env's shared per-step parts (``env.monitors["phase_detector"]``;
builder.py). ``set_cutoff(k)`` hands a cutoff to every part that takes one (the
two-device task's ``PhaseCutoff``); ``reset(options={"cutoff": k})`` does the same
before the reset.

``shadow_expert`` (``env.shadow_expert``, two_device/shadow.py; None = off): a controller reset
after every reset and asked, before every step's action is applied, what it would do; its answer is
reported in ``info["expert_action"]`` (and ``expert_stage`` ...), never executed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


def advance_step_limit(truncation: Any, steps: int) -> List[Any]:
    """Count ``steps`` already taken on every ``MaxSteps`` of a truncation (inside a ``Combination``
    too) and only those: an episode restored from a snapshot then has the limit minus the saved
    state's steps left (``env.start: {type: FromSnapshot, count_saved_steps: true}``). Called right
    after the reset (stEVE resets the truncation after the start). Returns the ``MaxSteps`` found."""
    from eve.truncation import Combination, MaxSteps
    found, todo = [], [truncation]
    while todo:
        part = todo.pop()
        if isinstance(part, MaxSteps):
            found.append(part)
        elif isinstance(part, Combination):
            todo.extend(part.truncations)
    for part in found:
        for _ in range(max(0, int(steps))):
            part.step()
    return found


class GuidewireEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, *, eve_env, simulation, observation_terms: Sequence[str],
                 labels: Sequence[Any], action_map, observation_clip: Optional[float],
                 max_episode_steps: Optional[int], render_mode: Optional[str] = None,
                 monitors: Optional[Dict[str, Any]] = None, reward_formula=None,
                 shadow_expert=None, randomizer=None) -> None:
        super().__init__()
        self.shadow_expert = shadow_expert
        self.randomizer = randomizer             # env.randomize (randomization.py); None = off
        self._env = eve_env
        self.simulation = simulation
        self.intervention = eve_env.intervention
        self.labels = list(labels)
        self.monitors: Dict[str, Any] = dict(monitors or {})
        # round two: the reward is one formula of the step's raw materials, computed here after the
        # labels (so this step's forces are in); the stEVE reward slot is then ZeroReward
        # (two_device/reward_formula.py). ``info["reward_materials"]`` carries the materials.
        self.reward_formula = reward_formula
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

        # the smoothness material (reward_formula ``action_change``): the newest executed action in
        # the observation's last-actions term, compared between consecutive observations
        self.action_slice: Optional[Tuple[int, int]] = None
        self._previous_flat: Optional[np.ndarray] = None
        if reward_formula is not None:
            from .two_device.reward_formula import action_slice_from_names
            term = getattr(reward_formula, "action_term", None)
            if term:
                self.action_slice = action_slice_from_names(
                    self.observation_names, int(np.prod(self.action_space.shape)), term)
            weight = float(getattr(getattr(reward_formula, "coefficients", None), "action_change", 0.0) or 0.0)
            if weight > 0 and self.action_slice is None:
                raise ValueError(f"reward_formula action_change > 0 needs the observation term {term!r} "
                                 "(the executed actions) in env.observation")

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

    def end_when_stalled(self) -> bool:
        """The next episode (only) ends once the devices stand still (two-device ``Stalled``); False when
        the env has no such part."""
        takers = [part for part in self._parts() if callable(getattr(part, "end_when_stalled", None))]
        for part in takers:
            part.end_when_stalled()
        return bool(takers)

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
        snapshot = None
        if options and options.get("snapshot") is not None:   # start from this saved state (snapshot.py)
            if not hasattr(self.intervention, "requested_snapshot"):
                raise ValueError("reset(options={'snapshot': ...}) needs env.start: FromSnapshot")
            snapshot = str(options["snapshot"])
            self.intervention.requested_snapshot = snapshot
            options = {k: v for k, v in options.items() if k != "snapshot"} or None
        observation, info = self._env.reset(seed=seed, options=options)
        for label in self.labels:        # every reset: the simulator may have rebuilt the scene
            label.attach(self)
        if self.reward_formula is not None:
            self.reward_formula.reset()
        flat_observation = self._flatten(observation)
        self._previous_flat = flat_observation
        out = dict(info)
        out["is_success"] = bool(out.get("is_success", False))
        out["simulation_error"] = bool(self.simulation.simulation_error)
        if snapshot is not None:
            last = dict(getattr(self._env.start, "last", None) or {})
            counted = 0
            if getattr(self._env.start, "count_saved_steps", False):   # env.start FromSnapshot's switch
                counted = int(last.get("step") or 0)
                advance_step_limit(self._env.truncation, counted)
            out["snapshot_start"] = {"path": snapshot, "grid": last.get("grid"), "saved_progress": last.get("progress"),
                                     "saved_step": last.get("step"), "tip_shift_mm": last.get("tip_shift_mm"),
                                     "progress": out.get("progress_ratio"), "counted_steps": counted}
        if self.shadow_expert is not None:     # after everything else: the controller reads the reset scene
            self.shadow_expert.reset(self)
        return flat_observation, out

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # the shadow expert's target is read off the state BEFORE this step's action is applied
        shadow = (None if self.shadow_expert is None
                  else self.shadow_expert.label(self, int(np.prod(self.action_space.shape))))
        raw = self._action_map.to_raw(action, self._raw_low, self._raw_high).reshape(self._raw_shape)
        for label in self.labels:
            label.begin_step()
        try:
            if self.randomizer is not None:      # only these steps get the drive's imperfections
                self.randomizer.live = True
            try:
                observation, reward, terminated, truncated, info = self._env.step(raw)
            finally:
                if self.randomizer is not None:
                    self.randomizer.live = False
            values, details = {}, {}
            for label in self.labels:
                values[label.name], details[label.name] = label.finish_step()
        except BaseException:
            for label in self.labels:
                label.abort_step()
            raise
        reward = float(reward)
        out = dict(info)
        flat_observation = self._flatten(observation)
        if self.reward_formula is not None:
            change = 0.0
            if self.action_slice is not None and self._previous_flat is not None:
                from .two_device.reward_formula import action_change_of
                lo, hi = self.action_slice
                change = action_change_of(self._previous_flat[lo:hi], flat_observation[lo:hi])
            formula_reward, materials = self.reward_formula.step(values, action_change=change)
            reward += float(formula_reward)
            out["reward_materials"] = np.asarray(materials, dtype=np.float32)
        if not np.isfinite(reward):
            raise FloatingPointError("stEVE returned a non-finite reward")
        out["is_success"] = bool(out.get("is_success", terminated))
        out["simulation_error"] = bool(self.simulation.simulation_error)
        out["raw_action"] = raw.reshape(-1).copy()
        out["labels"] = values
        out["label_details"] = details
        if shadow is not None:
            out.update(shadow)
        self._previous_flat = flat_observation
        return flat_observation, reward, bool(terminated), bool(truncated), out

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
