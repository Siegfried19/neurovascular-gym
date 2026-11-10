"""Build the env from a config and show what the controller would see.

Prints the observation layout (index, term, name, value at reset), then
steps a few random actions and prints reward, done flags and labels. Use it
after editing the observation/reward/labels sections of a config.

    python scripts/inspect_env.py --config configs/g13.yaml --steps 5
"""

from __future__ import annotations

import argparse

import numpy as np

from guidewire_rl.config import load_config
from guidewire_rl.env import build_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    env = build_env(config["env"])
    try:
        observation, _ = env.reset(seed=args.seed)
        print(f"observation: {observation.shape[0]} values; labels: {env.label_names}; "
              f"action: {env.action_space.shape[0]} values in [-1, 1]")
        index = 0
        for term, size in env.observation_layout:
            for k in range(size):
                print(f"  [{index:2d}] {term:<10} {env.observation_names[index]:<26} "
                      f"{observation[index]: .4f}")
                index += 1
        rng = np.random.default_rng(args.seed)
        for step in range(args.steps):
            action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            labels = ", ".join(f"{k}={v:.4f}" for k, v in info["labels"].items())
            print(f"step {step + 1}: action {np.round(action, 3)} reward {reward:.4f} "
                  f"terminated {terminated} truncated {truncated} labels: {labels or '-'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
