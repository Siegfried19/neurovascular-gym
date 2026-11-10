"""Wire config + env into a ready ``Trainer`` (fresh, resumed, or from G13);
an ``EnvPool`` instead of one env gets a ``ParallelTrainer`` (or an ``AsyncTrainer``
with ``training.async_envs``).

The replay's batch biasing comes from the config on every start (it is not part
of the checkpoint): ``replay.stratify_by`` / ``replay.stratify_min_samples`` and
the demonstration quota ``training.demos.fraction``. Demonstrations
(``training.demos.dir``) are loaded once, when the replay holds none yet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import yaml

from ..model import build_agent
from .checkpoint import (REFERENCE_FORCE_KEY, is_reference_g13, load_checkpoint,
                         reference_agent_state)
from .demos import load_demos
from .env_pool import EnvPool
from .parallel_trainer import AsyncTrainer, ParallelTrainer
from .replay import EpisodeReplay
from .trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_aux_labels(config: Mapping[str, Any], label_names) -> None:
    missing = [h["label"] for h in config.get("aux_heads") or [] if h["label"] not in label_names]
    if missing:
        raise ValueError(f"aux_heads use label(s) {missing} that env.labels does not define "
                         f"(defined: {list(label_names)})")


def make_replay(config: Mapping[str, Any], *, observation_dim: int, action_dim: int,
                label_names) -> EpisodeReplay:
    t = config["training"]
    return EpisodeReplay(capacity=int(t["replay_capacity"]), observation_dim=observation_dim,
                         action_dim=action_dim, horizon=int(t["horizon"]),
                         batch_size=int(t["batch_size"]), label_names=label_names,
                         seed=int(t["seed"]))


def configure_replay(replay: EpisodeReplay, config: Mapping[str, Any]) -> None:
    """Phase strata and the demonstration quota, from the config."""
    spec = config.get("replay") or {}
    unknown = set(spec) - {"stratify_by", "stratify_min_samples"}
    if unknown:
        raise KeyError(f"replay: unknown key(s) {sorted(unknown)}")
    demos = config["training"].get("demos") or {}
    replay.configure(stratify_by=spec.get("stratify_by"),
                     stratify_min_samples=int(spec.get("stratify_min_samples", 20_000)),
                     demo_fraction=float(demos.get("fraction", 0.0)) if demos.get("dir") else 0.0)


def start_run(config: Mapping[str, Any], env, run_dir: str | Path, *,
              resume: Optional[str | Path] = None,
              init_from_reference: Optional[str | Path] = None,
              device: Optional[str] = None, quiet: bool = False) -> Trainer:
    if resume and init_from_reference:
        raise ValueError("Use either resume or init_from_reference, not both")
    label_names = list(env.label_names)
    check_aux_labels(config, label_names)
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    episode_length = env.max_episode_steps
    if episode_length is None:
        raise ValueError("env.truncation needs a MaxSteps entry (sets the discount)")
    agent = build_agent(config, observation_dim=observation_dim, action_dim=action_dim,
                        episode_length=episode_length, device=device)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_state = None
    if resume:
        checkpoint = load_checkpoint(resume)
        agent.load_state_dict(checkpoint["agent"])
        replay = EpisodeReplay.from_state(checkpoint["replay"])
        resume_state = checkpoint
    elif init_from_reference:
        checkpoint = load_checkpoint(init_from_reference)
        if not is_reference_g13(checkpoint):
            raise ValueError(f"{init_from_reference} is not a reference G13 checkpoint")
        heads = config.get("aux_heads") or []
        if len(heads) != 1:
            raise ValueError("Starting from G13 needs exactly one aux head (the force head)")
        agent.load_state_dict(reference_agent_state(checkpoint, heads[0]["name"]))
        replay = EpisodeReplay.from_reference_state(
            checkpoint["replay"], {REFERENCE_FORCE_KEY: heads[0]["label"]})
        resume_state = {k: checkpoint[k] for k in
                        ("total_env_steps", "episode_index", "success_count")}
    else:
        replay = make_replay(config, observation_dim=observation_dim, action_dim=action_dim,
                             label_names=label_names)
    if replay.observation_dim != observation_dim:
        raise ValueError(f"Replay observations are {replay.observation_dim}-D but the env "
                         f"gives {observation_dim}-D; the observation config changed")
    configure_replay(replay, config)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    if isinstance(env, EnvPool):
        trainer_class = AsyncTrainer if config["training"].get("async_envs") else ParallelTrainer
    else:
        trainer_class = Trainer
    trainer = trainer_class(config, env, agent, replay, run_dir, resume_state=resume_state,
                            quiet=quiet)
    demos = config["training"].get("demos")
    if demos:
        outcome = load_demos(replay, demos, say=trainer.logger.print)
        trainer._event("demos", **outcome)  # pylint: disable=protected-access
    return trainer
