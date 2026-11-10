"""Build a snapshot library: the scripted procedure's states at chosen points, for episodes that
start there (guidewire_rl/env/two_device/snapshot.py, ``FromSnapshot``).

    conda run --no-capture-output -n guidewire_rl python scripts/make_snapshots.py \\
        --config configs/two_devices_ica.yaml --out runs/snapshots/two_devices_ica \\
        --seeds 1000-1019 --stages place,turn,pull,seat --steps 60 --workers 10

For every seed the scripted controller (``script.py``) runs from the insertion point; a snapshot is saved
at the first step of each listed stage (the controller's ``stage``) and at each listed fixed step, named
``seed<seed>_step<step>_<stage>.npz``, until the last listed stage is reached, the episode ends or
``--max-steps``. ``index.json`` lists them with their metadata (seed, step, stage, referee phase,
inserted lengths). Snapshots with referee phase <= 2 (before the wire leaves the catheter) do not depend
on the episode's target; ``FromSnapshot`` uses those by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def snapshots_of(env, seed: int, stages: Sequence[str], steps: Sequence[int], max_steps: int,
                 out: Path) -> List[Dict[str, Any]]:
    from guidewire_rl.env.two_device import snapshot
    from guidewire_rl.env.two_device.script import StagedController

    ctl = StagedController()
    env.reset(seed=seed)
    ctl.reset(env)
    wanted = list(stages)
    rows = []
    seen_stage = ctl.stage
    for k in range(max_steps):
        action = ctl.act(env)
        stage = ctl.stage
        if stage != seen_stage:                       # the state at the first step of a new stage
            seen_stage = stage
            if stage in wanted:
                rows.append(_save(env, out, seed, k, stage))
                if stage == wanted[-1]:
                    break
        if k in steps:
            rows.append(_save(env, out, seed, k, f"step{k}_{stage}"))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated or info.get("simulation_error") or getattr(ctl, "failed", False):
            break
    return rows


def _save(env, out: Path, seed: int, step: int, label: str) -> Dict[str, Any]:
    from guidewire_rl.env.two_device import snapshot
    sim = env.simulation
    snap = snapshot.capture(env, seed=seed, step=step, stage=label,
                            inserted_mm=[float(x) for x in np.asarray(sim.inserted_lengths).reshape(-1)])
    path = out / f"seed{seed}_step{step}_{label.split('_')[0] if label.startswith('step') else label}.npz"
    snapshot.save(path, snap)
    row = dict(file=path.name, **snap["meta"], nodes=int(len(snap["position"])))
    print(f"seed {seed} step {step} {label}: phase {row.get('phase')}, inserted {np.round(row['inserted_mm'], 1)}", flush=True)
    return row


def child(a) -> None:
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env
    out = Path(a.out)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    env = build_env(load_config(a.config, a.set)["env"])
    try:
        for seed in parse_seeds(a.seeds):
            rows = snapshots_of(env, seed, [s for s in a.stages.split(",") if s], [int(s) for s in a.steps.split(",") if s],
                                a.max_steps, out)
            (out / "rows" / f"seed_{seed}.json").write_text(json.dumps(rows))
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "two_devices_ica.yaml"))
    ap.add_argument("--out", default=None, help="default runs/snapshots/<config name>")
    ap.add_argument("--seeds", default="1000-1019")
    ap.add_argument("--stages", default="place,turn,pull,seat", help="script stages whose first step is saved")
    ap.add_argument("--steps", default="60", help="fixed steps that are saved too")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "runs" / "snapshots" / Path(a.config).stem)
    if a.child:
        child(a)
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(a.seeds)
    n = max(1, min(a.workers, len(seeds)))
    t0 = time.time()
    procs = []
    for chunk in [seeds[i::n] for i in range(n)]:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--config", a.config, "--out", str(out),
               "--seeds", ",".join(map(str, chunk)), "--stages", a.stages, "--steps", a.steps,
               "--max-steps", str(a.max_steps)]
        for item in a.set:
            cmd += ["--set", item]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    rows: List[Dict[str, Any]] = []
    for s in seeds:
        path = out / "rows" / f"seed_{s}.json"
        if path.exists():
            rows.extend(json.loads(path.read_text()))
    index = dict(config=a.config, overrides=a.set, seeds=seeds, stages=a.stages, steps=a.steps,
                 written=time.strftime("%Y-%m-%d %H:%M"), wall_s=round(time.time() - t0, 1), snapshots=rows)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    by_phase: Dict[int, int] = {}
    for r in rows:
        by_phase[r.get("phase", -1)] = by_phase.get(r.get("phase", -1), 0) + 1
    print(f"{len(rows)} snapshots in {out} ({time.time() - t0:.0f} s); by referee phase: {dict(sorted(by_phase.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
