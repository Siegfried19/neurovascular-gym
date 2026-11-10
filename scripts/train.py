"""Train TD-MPC2 on a YAML-configured guidewire env.

    python scripts/train.py --config configs/g13.yaml
    python scripts/train.py --config configs/g13.yaml --run-name smoke \
        --set training.total_steps=300 --set training.seed_steps=100
    python scripts/train.py --config configs/g13.yaml \
        --resume runs/<run>/checkpoints/step_20000.pt
    python scripts/train.py --config configs/g13.yaml \
        --init-from-reference reference/checkpoints_g13/step_260000.pt
    python scripts/train.py --config configs/g13.yaml --set training.num_envs=12
    python scripts/train.py --config configs/g13.yaml --set training.num_envs=24 \
        --set training.async_envs=true

Outputs go to runs/<run-name>/ (config.yaml, *.jsonl logs, checkpoints/):
episodes.jsonl one line per episode, metrics.jsonl one line per
logging.print_interval steps (reach rates, success, errors, step rate),
events.jsonl (checkpoints, dropped steps, worker replacements, cutoff weights).
With num_envs > 1 a scene that does not answer within training.step_timeout_s
seconds (default 120) is replaced by a fresh process, and so is a scene whose
process holds more than training.worker_max_rss_mb MB when its episode ends
(default: never; SOFA keeps part of every scene's memory, see
guidewire_rl/train/env_pool.py).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from guidewire_rl import RUNS_ROOT
from guidewire_rl.config import load_config
from guidewire_rl.env import build_env
from guidewire_rl.train.env_pool import EnvPool, split_cpus, step_timeout, worker_max_rss_mb
from guidewire_rl.train.setup import seed_everything, start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. training.total_steps=1000")
    parser.add_argument("--run-name", help="default: <run.name>_<timestamp>; "
                        "with --resume: the checkpoint's run folder")
    parser.add_argument("--resume", help="our checkpoint to continue from")
    parser.add_argument("--init-from-reference", help="reference G13 checkpoint to start from")
    parser.add_argument("--device", help="override training.device")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    if args.run_name:
        run_dir = RUNS_ROOT / args.run_name
    elif args.resume:
        run_dir = Path(args.resume).resolve().parents[1]
    else:
        run_dir = RUNS_ROOT / f"{config['run']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"

    seed = int(config["training"]["seed"])
    seed_everything(seed)
    num_envs = int(config["training"].get("num_envs", 1))
    if num_envs > 1:     # one SOFA scene per process; see guidewire_rl/train/parallel_trainer.py
        split = split_cpus()     # planning + updates get the fastest core to themselves
        if split is not None:
            os.sched_setaffinity(0, split[0])
            print(f"main process on CPUs {sorted(split[0])}, envs on the other {len(split[1])}")
        env = EnvPool(build_env, (config["env"],), num_envs=num_envs, seed=seed,
                      timeout=step_timeout(config), cpus=None if split is None else split[1],
                      max_rss_mb=worker_max_rss_mb(config))
    else:
        env = build_env(config["env"])
    try:
        trainer = start_run(config, env, run_dir, resume=args.resume,
                            init_from_reference=args.init_from_reference, device=args.device)
        print(f"run folder: {run_dir}")
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
