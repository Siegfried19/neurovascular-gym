"""Where actions come from.

A ``Policy`` maps an observation to a normalized action in [-1, 1]. Anything
with these methods plugs in (a trained checkpoint, a scripted controller, a
human-teleop adapter, the real robot's controller ...):

    reset()                           new episode
    act(obs, first_step) -> action
    predict(obs, action) -> {name: value}   optional; e.g. the force head's prediction
    debug() -> {key: value}           optional; shown in the panel, stored in Frame.extras

A ``SafetyFilter`` changes an action after the policy picked it (G16.1's fuzzy
force filter will be one); ``Filtered(policy, filter)`` combines them and keeps
the nominal action in ``debug()["nominal_action"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


class Policy(Protocol):
    label: str

    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray: ...
    def predict(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]: ...
    def debug(self) -> Dict[str, Any]: ...


class SafetyFilter(Protocol):
    label: str

    def reset(self) -> None: ...
    def filter(self, obs: np.ndarray, action: np.ndarray,
               predictions: Mapping[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]: ...


class _Base:
    label = "base"

    def reset(self) -> None:
        pass

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        return {}

    def debug(self) -> Dict[str, Any]:
        return {}


class RandomPolicy(_Base):
    """Uniform random actions (what training uses for its first steps)."""

    def __init__(self, action_dim: int, seed: Optional[int] = None, label: str = "随机动作") -> None:
        self.action_dim = int(action_dim)
        self.label = label
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)


class ConstantPolicy(_Base):
    """The same action every step; useful to check the simulator on its own."""

    def __init__(self, action: Sequence[float], label: str) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.label = label

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        return self.action.copy()


class CheckpointPolicy(_Base):
    """A TD-MPC2 agent from one of our checkpoints or a reference G13 checkpoint.

    Acts like ``guidewire_rl.train.evaluate`` (planner on, ``eval_mode=True``) and
    reports every auxiliary head's prediction for the chosen action (G13: the
    distal force of the coming step).
    """

    def __init__(self, config: Mapping[str, Any], path: str | Path, *, observation_dim: int,
                 action_dim: int, episode_length: int, device: Optional[str] = None) -> None:
        from guidewire_rl.model import build_agent
        from guidewire_rl.train.checkpoint import (is_reference_g13, load_checkpoint,
                                                   reference_agent_state)
        self.path = Path(path)
        self.label = checkpoint_label(self.path)
        self.agent = build_agent(config, observation_dim=observation_dim, action_dim=action_dim,
                                 episode_length=episode_length, device=device)
        checkpoint = load_checkpoint(self.path)
        if is_reference_g13(checkpoint):
            heads = config.get("aux_heads") or []
            if len(heads) != 1:
                raise ValueError("A G13 checkpoint needs exactly one aux head (the force head)")
            state = reference_agent_state(checkpoint, heads[0]["name"])
        else:
            state = checkpoint["agent"]
        self.agent.load_state_dict(state, load_optimizers=False)
        self.heads = [spec.name for spec in self.agent.aux_specs]

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        return self.agent.act(obs, first_step=first_step, eval_mode=True)

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        return {name: self.agent.aux_predict(name, obs, action) for name in self.heads}


class Filtered(_Base):
    def __init__(self, policy: Policy, safety: SafetyFilter) -> None:
        self.policy, self.safety = policy, safety
        self.label = f"{policy.label} + {safety.label}"
        self._debug: Dict[str, Any] = {}

    def reset(self) -> None:
        self.policy.reset()
        self.safety.reset()

    def act(self, obs: np.ndarray, *, first_step: bool) -> np.ndarray:
        nominal = self.policy.act(obs, first_step=first_step)
        predictions = self.policy.predict(obs, nominal)
        safe, info = self.safety.filter(obs, nominal, predictions)
        self._debug = {"nominal_action": np.asarray(nominal, np.float32), "filter": info,
                       **self.policy.debug()}
        return np.asarray(safe, dtype=np.float32)

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        return self.policy.predict(obs, action)

    def debug(self) -> Dict[str, Any]:
        return dict(self._debug)


class PassThroughFilter:
    """Placeholder SafetyFilter: leaves the action alone (where G16.1 will go)."""

    label = "不过滤"

    def reset(self) -> None:
        pass

    def filter(self, obs, action, predictions):
        return action, {"changed": False}


# ---------------------------------------------------------------- the menu --
def checkpoint_label(path: Path) -> str:
    parts = path.resolve().parts
    run = parts[-3] if len(parts) >= 3 and parts[-2] == "checkpoints" else path.parent.name
    return f"{run} / {path.stem}"


def discover_checkpoints(runs_dir: str | Path) -> List[Path]:
    """Checkpoints under ``runs/<run>/checkpoints/``, newest run first."""
    runs_dir = Path(runs_dir)
    found = sorted(runs_dir.glob("*/checkpoints/*.pt"),
                   key=lambda p: (p.parent.parent.stat().st_mtime, _step_of(p)), reverse=True)
    return found


def _step_of(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def policy_menu(action_dim: int, checkpoints: Sequence[Path],
                make_checkpoint: Callable[[Path], Policy],
                extra: Optional[Mapping[str, Callable[[], Policy]]] = None) -> Dict[str, Callable[[], Policy]]:
    """Label -> factory, in the order the dropdown shows them (``extra``, e.g. the keyboard, after
    the built-in ones and before the checkpoints)."""
    menu: Dict[str, Callable[[], Policy]] = {
        "随机动作": lambda: RandomPolicy(action_dim),
        "不动": lambda: ConstantPolicy(np.zeros(action_dim), "不动"),
        "只往前推": lambda: ConstantPolicy(np.eye(action_dim)[0], "只往前推"),
    }
    menu.update(extra or {})
    for path in checkpoints:
        menu[checkpoint_label(Path(path))] = (lambda p=Path(path): make_checkpoint(p))
    return menu
