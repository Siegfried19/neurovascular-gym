"""Episode replay buffer.

Same sampling as the official TD-MPC2 ``Buffer`` (TorchRL ``SliceSampler`` with
``strict_length=True``): each batch element is a contiguous slice of
``horizon + 1`` observations and ``horizon`` actions/rewards/terminations from
one episode, start positions uniform over all valid starts. Labels ride along
with the transition they were measured on.

Every episode also keeps, per observation, the task phase the env reported
(``phase``, int8, 0 when the env has no phases) and whether it is a scripted
demonstration (``demo``). Two optional ways to bias the batch use them; both are
off by default, and then the sampler draws exactly what it always drew:

* **phase strata** (``configure(stratify_by="phase")``): a slice belongs to the
  stratum of the phase at its first observation. The strata are the stretches
  between the training cutoffs 2, 3, 4, 5 (``PHASE_STRATA``): phases {0, 1},
  {2}, {3}, {4}. The batch is cut into four equal quarters, one per stratum; a
  quarter is drawn from its own stratum when that stratum has more than
  ``stratify_min_samples`` (20 000) valid starts, otherwise uniformly from the
  whole replay. Once every stratum is that big, each gets exactly 25 %; a small
  stratum is never blown up to 25 %.
* **demo quota** (``configure(demo_fraction=f)``): at least ``ceil(f * batch)``
  slices of every batch come from demonstration episodes (uniform over their
  valid starts); the rest is drawn as above, from everything including the
  demos. Demonstration episodes are never dropped when the replay is full.

Can load the replay stored inside the reference G13 checkpoint
(``from_reference_state``), whose episodes keep the force label as ``forces``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

FORMAT = "guidewire_rl.replay/1"   # episodes may carry "phase" and "demo" (older files: zeros / False)
# round two (work log 2026-11-10): episodes may also carry "materials" (T x M, the reward's raw
# materials, two_device/reward_formula.py), "source" (T, int8: SOURCE_*), "protected" (never evicted:
# scripted demonstrations and the policy's own successful episodes kept as self-demonstrations, the
# latter up to ``self_demo_capacity`` episodes, oldest out) and "outcome" (how the episode ended).
# With a ``reward_fn`` configured, every sampled batch's rewards are recomputed from the materials.
# DAgger-style labels (2026-11-13, env/two_device/shadow.py): every episode also carries "expert"
# (T x action_dim float32): each step's imitation target, NaN where there is none. On the scripted
# steps (sources prefix and demo) it is always the executed action; elsewhere what the env's shadow
# expert reported (info["expert_action"]). Older episodes and files without it get exactly that
# (the action on scripted steps, NaN elsewhere).
SOURCE_POLICY, SOURCE_PREFIX, SOURCE_DEMO, SOURCE_SELF_DEMO = 0, 1, 2, 3
SOURCE_NAMES = ("policy", "prefix", "demo", "self_demo")
SCRIPTED_SOURCES = (SOURCE_PREFIX, SOURCE_DEMO)    # the script was in control: the target is the action

PHASE_STRATA = ((0, 1), (2,), (3,), (4,))     # the stretches ended by cutoffs 2, 3, 4, 5
STRATIFY_KEYS = (None, "phase")


def phase_stratum(phase: np.ndarray) -> np.ndarray:
    """Stratum index (0-3) of each phase: {0,1} -> 0, 2 -> 1, 3 -> 2, 4 -> 3 (clipped)."""
    return np.clip(np.asarray(phase, dtype=np.int64) - 1, 0, len(PHASE_STRATA) - 1)


@dataclass
class Batch:
    observations: torch.Tensor          # (H+1, B, obs_dim)
    actions: torch.Tensor               # (H, B, act_dim)
    rewards: torch.Tensor               # (H, B, 1)
    terminated: torch.Tensor            # (H, B, 1)
    labels: Dict[str, torch.Tensor]     # each (H, B, 1)
    source: Optional[torch.Tensor] = None   # (H, B) SOURCE_* of each transition (behaviour cloning)
    expert: Optional[torch.Tensor] = None   # (H, B, act_dim) imitation target, NaN where none


class EpisodeReplay:
    def __init__(self, *, capacity: int, observation_dim: int, action_dim: int,
                 horizon: int, batch_size: int, label_names: Sequence[str] = (),
                 seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.label_names = list(label_names)
        self.rng = np.random.default_rng(seed)
        self.episodes: List[Dict[str, Any]] = []
        self.size = 0
        self.stratify_by: Optional[str] = None
        self.stratify_min_samples = 20_000
        self.demo_fraction = 0.0
        self.reward_fn: Optional[Any] = None           # materials (..., M) -> rewards (...)
        self.n_materials: Optional[int] = None
        self.materials_upgrade: Optional[Any] = None   # one episode's older materials -> current ones
        self.self_demo_capacity: Optional[int] = None  # episodes; None = unbounded
        self._dirty = True
        self._flat: Dict[str, Any] = {}

    def configure(self, *, stratify_by: Optional[str] = None, stratify_min_samples: int = 20_000,
                  demo_fraction: float = 0.0, reward_fn: Optional[Any] = None,
                  n_materials: Optional[int] = None, self_demo_capacity: Optional[int] = None,
                  materials_upgrade: Optional[Any] = None) -> None:
        """Batch biasing (see the module docstring); not saved with the replay, the
        run's config sets it again on every start. ``reward_fn(materials) -> rewards`` recomputes
        every sampled reward from the stored materials (``n_materials`` fields each); every episode
        must then carry materials. ``self_demo_capacity``: how many of the policy's own successful
        episodes are kept protected (oldest out). ``materials_upgrade(materials, observations=...) -> materials``: brings
        one episode's materials recorded with fewer fields up to date (derived fields); applied to the
        episodes already here and to every one added from now on."""
        if stratify_by not in STRATIFY_KEYS:
            raise ValueError(f"replay.stratify_by must be one of {STRATIFY_KEYS}, got {stratify_by!r}")
        if not 0.0 <= float(demo_fraction) <= 1.0:
            raise ValueError(f"demo fraction must be in [0, 1], got {demo_fraction}")
        self.stratify_by = stratify_by
        self.stratify_min_samples = int(stratify_min_samples)
        self.demo_fraction = float(demo_fraction)
        self.reward_fn = reward_fn
        self.n_materials = None if n_materials is None else int(n_materials)
        self.self_demo_capacity = None if self_demo_capacity is None else int(self_demo_capacity)
        self.materials_upgrade = materials_upgrade
        if materials_upgrade is not None:
            for episode in self.episodes:
                if episode.get("materials") is not None:
                    episode["materials"] = np.asarray(
                        materials_upgrade(episode["materials"], observations=episode["observations"]),
                        dtype=np.float32)
        if self.n_materials is not None:
            wrong = [e["materials"].shape[1] for e in self.episodes
                     if e.get("materials") is not None and e["materials"].shape[1] != self.n_materials]
            if wrong:
                raise ValueError(f"{len(wrong)} episode(s) have materials with {sorted(set(wrong))} fields, "
                                 f"expected {self.n_materials}")
        if self.reward_fn is not None:
            lacking = [i for i, e in enumerate(self.episodes) if e.get("materials") is None]
            if lacking:
                raise ValueError(f"reward recomputation needs materials in every episode; {len(lacking)} "
                                 f"episode(s) have none (re-record the demonstrations with this config)")
        self._dirty = True

    # ------------------------------------------------------------------ add --
    def add(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            terminated: np.ndarray, labels: Optional[Mapping[str, np.ndarray]] = None, *,
            phase: Optional[np.ndarray] = None, demo: bool = False,
            materials: Optional[np.ndarray] = None, source: Optional[np.ndarray] = None,
            protected: Optional[bool] = None, outcome: str = "",
            expert: Optional[np.ndarray] = None) -> None:
        """``phase``: one int per observation (length + 1); None means all 0. ``materials``:
        (length, M) reward materials; ``source``: one ``SOURCE_*`` per step (None: policy, or demo
        when ``demo``); ``protected``: never evicted (default: ``demo``); ``outcome``: the end;
        ``expert``: (length, action_dim) imitation targets, NaN where none (None: none); on the
        scripted steps the action is the target whatever is given."""
        episode = {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1),
            "terminated": np.asarray(terminated, dtype=np.float32).reshape(-1),
            "labels": {k: np.asarray(v, dtype=np.float32).reshape(-1)
                       for k, v in (labels or {}).items()},
            "outcome": str(outcome),
        }
        length = len(episode["actions"])
        if materials is not None:
            episode["materials"] = np.asarray(materials, dtype=np.float32).reshape(length, -1)
            if self.materials_upgrade is not None:
                episode["materials"] = np.asarray(
                    self.materials_upgrade(episode["materials"], observations=episode["observations"]),
                    dtype=np.float32)
            if self.n_materials is not None and episode["materials"].shape[1] != self.n_materials:
                raise ValueError(f"materials must have {self.n_materials} fields, got "
                                 f"{episode['materials'].shape[1]}")
        elif self.reward_fn is not None:
            raise ValueError("this replay recomputes rewards from materials; the episode has none")
        else:
            episode["materials"] = None
        if source is None:
            source = np.full(length, SOURCE_DEMO if demo else SOURCE_POLICY, dtype=np.int8)
        episode["source"] = np.asarray(source).reshape(-1).astype(np.int8)
        if episode["source"].shape != (length,):
            raise ValueError(f"source must have one entry per step ({length})")
        episode["protected"] = bool(demo) if protected is None else bool(protected)
        if length < 1:
            raise ValueError("Episode has no transitions")
        if episode["observations"].shape != (length + 1, self.observation_dim):
            raise ValueError(f"observations must be ({length + 1}, {self.observation_dim}), "
                             f"got {episode['observations'].shape}")
        if episode["actions"].shape != (length, self.action_dim):
            raise ValueError(f"actions must be ({length}, {self.action_dim})")
        target = (np.full((length, self.action_dim), np.nan, np.float32) if expert is None
                  else np.asarray(expert, dtype=np.float32))
        if target.shape != (length, self.action_dim):
            raise ValueError(f"expert must be ({length}, {self.action_dim}), got {target.shape}")
        scripted = np.isin(episode["source"], SCRIPTED_SOURCES)[:, None]
        episode["expert"] = np.where(scripted, episode["actions"], target).astype(np.float32)
        for key in ("rewards", "terminated"):
            if episode[key].shape != (length,):
                raise ValueError(f"{key} must have length {length}")
        missing = [n for n in self.label_names if n not in episode["labels"]]
        if missing:
            raise ValueError(f"Episode lacks label(s) {missing}")
        for name in self.label_names:
            if episode["labels"][name].shape != (length,):
                raise ValueError(f"label {name} must have length {length}")
        episode["labels"] = {n: episode["labels"][n] for n in self.label_names}
        if phase is None:
            episode["phase"] = np.zeros(length + 1, dtype=np.int8)
        else:
            episode["phase"] = np.asarray(phase).reshape(-1).astype(np.int8)
            if episode["phase"].shape != (length + 1,):
                raise ValueError(f"phase must have one entry per observation ({length + 1})")
        episode["demo"] = bool(demo)
        self.episodes.append(episode)
        self.size += length
        if self.self_demo_capacity is not None:
            # the policy's own successful episodes: protected, but only the latest N of them
            self_demos = [i for i, e in enumerate(self.episodes)
                          if e["protected"] and not e["demo"]]
            for i in self_demos[: max(len(self_demos) - self.self_demo_capacity, 0)]:
                self.episodes[i]["protected"] = False
        while self.size > self.capacity:
            # the oldest episode that is not protected; never the one just added
            victim = next((i for i, e in enumerate(self.episodes) if not e["protected"]), None)
            if victim is None or victim == len(self.episodes) - 1:
                break
            self.size -= len(self.episodes.pop(victim)["actions"])
        self._dirty = True

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_demo_episodes(self) -> int:
        return sum(1 for e in self.episodes if e["demo"])

    @property
    def num_self_demo_episodes(self) -> int:
        return sum(1 for e in self.episodes if e["protected"] and not e["demo"])

    def _rebuild(self) -> None:
        H = self.horizon
        lengths = np.asarray([len(e["actions"]) for e in self.episodes], dtype=np.int64)
        obs_offsets = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]]).astype(np.int64)
        trans_offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        starts_per_episode = np.maximum(lengths - H + 1, 0)
        episode_of_start = np.repeat(np.arange(len(lengths)), starts_per_episode)
        first_start = np.concatenate([[0], np.cumsum(starts_per_episode)[:-1]])
        start_in_episode = (np.arange(int(starts_per_episode.sum()))
                            - np.repeat(first_start, starts_per_episode))
        self._flat = {
            "observations": np.concatenate([e["observations"] for e in self.episodes]),
            "actions": np.concatenate([e["actions"] for e in self.episodes]),
            "rewards": np.concatenate([e["rewards"] for e in self.episodes]),
            "terminated": np.concatenate([e["terminated"] for e in self.episodes]),
            # the policy's own successful episodes that are still kept as self-demonstrations read
            # SOURCE_SELF_DEMO in a batch (training.self_bc_coef clones them); the stored episode
            # keeps the source it was added with, so an episode that drops out of the latest
            # ``self_demo_capacity`` reads SOURCE_POLICY again
            "source": np.concatenate([
                np.full(len(e["actions"]), SOURCE_SELF_DEMO, dtype=np.int8)
                if (e["protected"] and not e["demo"]
                    and not np.isin(e["source"], SCRIPTED_SOURCES).any())
                else np.asarray(e["source"], dtype=np.int8) for e in self.episodes]),
            "expert": np.concatenate([e["expert"] for e in self.episodes]),
            "labels": {n: np.concatenate([e["labels"][n] for e in self.episodes])
                       for n in self.label_names},
            "obs_offsets": obs_offsets, "trans_offsets": trans_offsets,
            "episode_of_start": episode_of_start, "start_in_episode": start_in_episode,
        }
        if self.reward_fn is not None:
            materials = np.concatenate([e["materials"] for e in self.episodes])
            self._flat["rewards"] = np.asarray(self.reward_fn(materials), dtype=np.float32).reshape(-1)
        if self.stratify_by is not None or self.demo_fraction > 0:
            phase = np.concatenate([e["phase"] for e in self.episodes])
            start_phase = phase[obs_offsets[episode_of_start] + start_in_episode]
            stratum = phase_stratum(start_phase)
            self._flat["stratum_starts"] = [np.flatnonzero(stratum == g)
                                            for g in range(len(PHASE_STRATA))]
            # the demonstration quota draws from the scripted demonstrations and the policy's own
            # successful episodes kept as self-demonstrations alike
            is_demo = np.asarray([e["demo"] or (e["protected"] and not e["demo"])
                                  for e in self.episodes], dtype=bool)
            self._flat["demo_starts"] = np.flatnonzero(is_demo[episode_of_start])
        self._dirty = False

    def set_demo_fraction(self, fraction: float) -> None:
        """A new demonstration quota (the schedule's ramp); the next batch uses it."""
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"demo fraction must be in [0, 1], got {fraction}")
        self.demo_fraction = float(fraction)
        self._dirty = True

    def recompute_rewards(self) -> None:
        """After a change of coefficients: the next batch uses the new rewards."""
        self._dirty = True

    @property
    def num_valid_starts(self) -> int:
        if not self.episodes:
            return 0
        if self._dirty:
            self._rebuild()
        return len(self._flat["episode_of_start"])

    def can_sample(self) -> bool:
        return self.num_valid_starts > 0

    def stratum_sizes(self) -> List[int]:
        """Valid starts per phase stratum (needs stratification or a demo quota)."""
        if not self.can_sample() or "stratum_starts" not in self._flat:
            return [0] * len(PHASE_STRATA)
        return [len(s) for s in self._flat["stratum_starts"]]

    # --------------------------------------------------------------- sample --
    def _biased_picks(self, batch_size: int) -> np.ndarray:
        """Valid-start indices for one batch with the demo quota and/or phase strata."""
        flat, total = self._flat, self.num_valid_starts
        parts: List[np.ndarray] = []
        rest = batch_size
        demo_starts = flat["demo_starts"]
        if self.demo_fraction > 0 and len(demo_starts):
            count = min(batch_size, math.ceil(self.demo_fraction * batch_size - 1e-9))
            parts.append(demo_starts[self.rng.integers(0, len(demo_starts), size=count)])
            rest -= count
        if self.stratify_by is None:
            parts.append(self.rng.integers(0, total, size=rest))
        else:
            strata = flat["stratum_starts"]
            n = len(strata)
            for g, starts in enumerate(strata):
                size = rest // n + (1 if g < rest % n else 0)
                if len(starts) > self.stratify_min_samples:
                    parts.append(starts[self.rng.integers(0, len(starts), size=size)])
                else:
                    parts.append(self.rng.integers(0, total, size=size))
        return np.concatenate(parts).astype(np.int64)

    def sample_indices(self, batch_size: Optional[int] = None):
        """(episode index, start step) for each batch element."""
        if not self.can_sample():
            raise RuntimeError(f"No episode with at least horizon={self.horizon} transitions yet")
        if self.stratify_by is None and self.demo_fraction <= 0:
            picks = self.rng.integers(0, self.num_valid_starts, size=batch_size or self.batch_size)
        else:
            picks = self._biased_picks(batch_size or self.batch_size)
        return self._flat["episode_of_start"][picks], self._flat["start_in_episode"][picks]

    def sample(self, device: torch.device | str = "cpu") -> Batch:
        episodes, starts = self.sample_indices()
        H = self.horizon
        flat = self._flat
        obs_idx = (flat["obs_offsets"][episodes] + starts)[None, :] + np.arange(H + 1)[:, None]
        tr_idx = (flat["trans_offsets"][episodes] + starts)[None, :] + np.arange(H)[:, None]

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        return Batch(
            observations=tensor(flat["observations"][obs_idx]),
            actions=tensor(flat["actions"][tr_idx]),
            rewards=tensor(flat["rewards"][tr_idx][..., None]),
            terminated=tensor(flat["terminated"][tr_idx][..., None]),
            labels={n: tensor(flat["labels"][n][tr_idx][..., None]) for n in self.label_names},
            source=torch.as_tensor(flat["source"][tr_idx].astype(np.int64), device=device),
            expert=tensor(flat["expert"][tr_idx]),
        )

    # ---------------------------------------------------------- save / load --
    def state_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT, "capacity": self.capacity,
            "observation_dim": self.observation_dim, "action_dim": self.action_dim,
            "horizon": self.horizon, "batch_size": self.batch_size,
            "label_names": list(self.label_names), "episodes": self.episodes,
            "size": self.size, "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "EpisodeReplay":
        if state.get("format") != FORMAT:
            raise ValueError(f"Not a {FORMAT} replay (use from_reference_state for G13)")
        replay = cls(capacity=state["capacity"], observation_dim=state["observation_dim"],
                     action_dim=state["action_dim"], horizon=state["horizon"],
                     batch_size=state["batch_size"], label_names=state["label_names"])
        for e in state["episodes"]:
            replay.add(e["observations"], e["actions"], e["rewards"], e["terminated"], e["labels"],
                       phase=e.get("phase"), demo=e.get("demo", False), materials=e.get("materials"),
                       source=e.get("source"), protected=e.get("protected"), outcome=e.get("outcome", ""),
                       expert=e.get("expert"))          # replays saved before 2026-11-13: derived
        replay.rng.bit_generator.state = state["rng_state"]
        return replay

    @classmethod
    def from_reference_state(cls, state: Mapping[str, Any],
                             label_map: Mapping[str, str]) -> "EpisodeReplay":
        """G13 checkpoint replay; ``label_map`` maps its keys (``forces``) to label names."""
        replay = cls(capacity=state["capacity"], observation_dim=state["observation_dim"],
                     action_dim=state["action_dim"], horizon=state["horizon"],
                     batch_size=state["batch_size"], label_names=list(label_map.values()))
        for e in state["episodes"]:
            replay.add(e["observations"], e["actions"], e["rewards"], e["terminated"],
                       {new: e[old] for old, new in label_map.items()})
        if replay.size != state["size"]:
            raise RuntimeError(f"Reference replay size mismatch: {replay.size} != {state['size']}")
        replay.rng.bit_generator.state = state["rng_state"]
        return replay
