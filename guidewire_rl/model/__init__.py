"""Solver side: TD-MPC2 world model, planner, updates and auxiliary heads."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ..config import agent_config
from .agent import TDMPC2Agent
from .aux_heads import AuxHeadSpec, parse_specs


def build_agent(config: Mapping[str, Any], *, observation_dim: int, action_dim: int,
                episode_length: int, device: str | torch.device | None = None) -> TDMPC2Agent:
    device = torch.device(device or config["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable (set training.device=cpu)")
    return TDMPC2Agent(observation_dim, action_dim, agent_config(config),
                       episode_length=episode_length, device=device,
                       aux_heads=parse_specs(config.get("aux_heads")))


__all__ = ["AuxHeadSpec", "TDMPC2Agent", "build_agent", "parse_specs"]
