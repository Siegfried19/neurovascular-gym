"""Training checkpoints (full state: agent, optimizers, replay, counters, RNG).

The top-level layout follows the reference G13 checkpoints, so a G13
``step_*.pt`` can seed a run here (``import_reference``). Checkpoints are
pickles: load only files you trust (our own runs and the hash-checked G13 file).
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

FORMAT_VERSION = 2
FRAMEWORK = "guidewire_rl"
REFERENCE_ALGORITHM = "Geometry-Force-TD-MPC2"
REFERENCE_FORCE_HEAD = "force_head"
REFERENCE_FORCE_KEY = "forces"


def capture_rng(trainer_rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if trainer_rng is not None:
        state["trainer"] = trainer_rng.bit_generator.state
    return state


def restore_rng(state: Mapping[str, Any], trainer_rng: Optional[np.random.Generator] = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if trainer_rng is not None and "trainer" in state:
        trainer_rng.bit_generator.state = state["trainer"]


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"format_version": FORMAT_VERSION, "framework": FRAMEWORK,
            "saved_at_unix": time.time(), **payload}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)          # never leave a half-written checkpoint behind
    return path


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)


def is_reference_g13(checkpoint: Mapping[str, Any]) -> bool:
    return (checkpoint.get("algorithm") == REFERENCE_ALGORITHM
            and checkpoint.get("framework") != FRAMEWORK)


def reference_agent_state(checkpoint: Mapping[str, Any], head_name: str) -> Dict[str, Any]:
    """G13 agent state with ``force_head.*`` renamed to ``aux_heads.<head_name>.*``.

    Optimizer states need no change: G13's param groups are encoder, dynamics,
    reward, termination, Q ensemble, force head — the same order as ours.
    """
    state = dict(checkpoint["agent"])
    prefix = REFERENCE_FORCE_HEAD + "."
    state["model"] = {
        (f"aux_heads.{head_name}." + key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state["model"].items()
    }
    return state
