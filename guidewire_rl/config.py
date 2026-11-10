"""YAML configs: ``extends:`` inheritance and ``key.path=value`` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

SECTIONS = ("run", "env", "model", "aux_heads", "planning", "training", "replay",
            "evaluation", "checkpoint", "logging")


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """Dicts merge key by key; anything else (lists included) is replaced; null
    deletes the inherited key (so a component spec can drop its parent's arguments)."""
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_override(config: Dict[str, Any], assignment: str) -> None:
    """``training.total_steps=1000`` (value parsed as YAML)."""
    if "=" not in assignment:
        raise ValueError(f"Override must look like key.path=value, got {assignment!r}")
    path, raw = assignment.split("=", 1)
    keys = path.strip().split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Override path {path!r}: {key!r} is not a section")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Override path {path!r}: unknown key {keys[-1]!r}")
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    config = deep_merge(load_config(path.parent / parent), raw) if parent else raw
    for assignment in overrides:
        apply_override(config, assignment)
    unknown = [key for key in config if key not in SECTIONS]
    if unknown:
        raise KeyError(f"Unknown config section(s) {unknown}; expected {SECTIONS}")
    return config


def agent_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Flat dict the agent reads, merged the same way as the reference loader."""
    return {**config["model"], **config["training"], **config["planning"]}
