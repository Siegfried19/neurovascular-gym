"""Train a checkpoint further on its own replay, without stepping any simulator ("learn first").

    python scripts/offline_updates.py --config configs/two_devices_ica.yaml \
        --checkpoint runs/<run>/checkpoints/step_N.pt --updates 20000 \
        --out runs/<new_run>/checkpoints/offline_step_N.pt

What it does (work log 2026-11-13: first learn what the hand-written procedure does, then optimise):

* Loads the checkpoint's agent and replay; the replay is configured with ``--config`` (the reward is
  recomputed from the stored materials with that config's coefficients; materials recorded before a
  field existed are derived, e.g. ``action_change`` from the episodes' observations).
* Runs ``--updates`` ordinary agent updates on that replay (world model, critics, the policy prior;
  with ``training.bc_coef`` > 0 the prior also clones the scripted transitions: prefix steps and
  demonstrations). Nothing new is collected; the env step counter does not move.
* Writes a checkpoint with everything of the original (curriculum, schedule, counters) and the
  updated agent and replay, so ``scripts/train.py --resume`` continues from it.
* Prints the losses every ``--log-every`` updates and writes them to ``<out>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from guidewire_rl.config import load_config
from guidewire_rl.model import build_agent
from guidewire_rl.train.checkpoint import load_checkpoint, save_checkpoint
from guidewire_rl.train.replay import EpisodeReplay
from guidewire_rl.train.setup import configure_replay

KEEP = ("format_version", "framework", "saved_at_unix")


def episode_length(config) -> int:
    for spec in config["env"]["truncation"]:
        if isinstance(spec, dict) and spec.get("type") == "MaxSteps":
            return int(spec["max_steps"])
    raise ValueError("env.truncation needs a MaxSteps entry")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    checkpoint = load_checkpoint(args.checkpoint)
    replay = EpisodeReplay.from_state(checkpoint["replay"])
    configure_replay(replay, config, observation_names=checkpoint.get("observation_names"))
    device = args.device or config["training"].get("device", "cuda:0")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    agent = build_agent(config, observation_dim=replay.observation_dim, action_dim=replay.action_dim,
                        episode_length=episode_length(config), device=device)
    agent.load_state_dict(checkpoint["agent"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".jsonl")
    started = time.time()
    with log_path.open("w") as log:
        window = []
        for k in range(1, args.updates + 1):
            info = agent.update(replay)
            window.append(info)
            if k % args.log_every == 0 or k == args.updates:
                keys = {key for w in window for key, value in w.items() if isinstance(value, (int, float))}
                means = {}
                for key in sorted(keys):          # some fields are None on some batches (no terminal step)
                    values = [float(w[key]) for w in window if isinstance(w.get(key), (int, float))]
                    means[key] = sum(values) / len(values)
                row = {"update": k, "seconds": round(time.time() - started, 1), **means}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(" ".join(f"{key}={row[key]:.4g}" if isinstance(row[key], float) else f"{key}={row[key]}"
                               for key in ("update", "seconds", "bc_loss", "consistency_loss", "reward_loss",
                                           "value_loss", "policy_entropy") if key in row), flush=True)
                window = []

    payload = {key: value for key, value in checkpoint.items() if key not in KEEP}
    payload["agent"] = agent.state_dict()
    payload["replay"] = replay.state_dict()
    payload["offline_updates"] = {"from": str(args.checkpoint), "updates": int(args.updates),
                                  "config": str(args.config), "overrides": list(args.set)}
    save_checkpoint(out, payload)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
