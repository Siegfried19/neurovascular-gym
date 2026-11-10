"""Auxiliary prediction heads on the world model's predicted next latent.

G13's force head is one of these: ``softplus(MLP(f(z_t, a_t)))`` trained to
predict the next control step's distal contact force. A head reads the
*imagined* next latent, so with ``gradient: shared`` its loss also shapes the
encoder and dynamics; with ``gradient: detached`` it learns on the side only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import mlp, weight_init

OUTPUTS = {"softplus": nn.Softplus, "sigmoid": nn.Sigmoid, "identity": nn.Identity}
LOSSES = {
    "mse": lambda pred, target: F.mse_loss(pred, target, reduction="none"),
    "huber": lambda pred, target: F.smooth_l1_loss(pred, target, reduction="none"),
    "l1": lambda pred, target: F.l1_loss(pred, target, reduction="none"),
}
GRADIENTS = ("shared", "detached")


@dataclass(frozen=True)
class AuxHeadSpec:
    name: str
    label: str
    loss: str = "mse"
    weight: float = 1.0
    gradient: str = "shared"
    output: str = "identity"
    hidden: Optional[Tuple[int, ...]] = None   # default: [mlp_dim, mlp_dim]

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"aux head {self.name}: loss must be one of {list(LOSSES)}")
        if self.gradient not in GRADIENTS:
            raise ValueError(f"aux head {self.name}: gradient must be one of {GRADIENTS}")
        if self.output not in OUTPUTS:
            raise ValueError(f"aux head {self.name}: output must be one of {list(OUTPUTS)}")
        if self.weight < 0:
            raise ValueError(f"aux head {self.name}: weight must be >= 0")

    @classmethod
    def from_config(cls, entry: Mapping[str, Any]) -> "AuxHeadSpec":
        entry = dict(entry)
        if "hidden" in entry and entry["hidden"] is not None:
            entry["hidden"] = tuple(int(h) for h in entry["hidden"])
        return cls(**entry)


def parse_specs(entries: Optional[Sequence[Mapping[str, Any]]]) -> Tuple[AuxHeadSpec, ...]:
    specs = tuple(AuxHeadSpec.from_config(e) for e in (entries or []))
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate aux head names: {names}")
    return specs


def build_head(latent_dim: int, mlp_dim: int, spec: AuxHeadSpec) -> nn.Sequential:
    """Same structure as G13's ``force_head`` (so checkpoint keys line up)."""
    hidden = list(spec.hidden) if spec.hidden is not None else [mlp_dim, mlp_dim]
    head = nn.Sequential(mlp(latent_dim, hidden, 1), OUTPUTS[spec.output]())
    head.apply(weight_init)
    return head


def build_heads(latent_dim: int, mlp_dim: int, specs: Sequence[AuxHeadSpec]) -> nn.ModuleDict:
    return nn.ModuleDict({s.name: build_head(latent_dim, mlp_dim, s) for s in specs})


def head_loss(spec: AuxHeadSpec, prediction: torch.Tensor,
              target: torch.Tensor) -> torch.Tensor:
    """Elementwise loss, shape like ``prediction`` (horizon, batch, 1)."""
    return LOSSES[spec.loss](prediction, target)
