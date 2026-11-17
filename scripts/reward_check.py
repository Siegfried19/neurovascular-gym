"""Reward check (round two, plan item A8, work log 2026-11-06 00:30): does the reward rank the scripted
way of doing the task above the two shortcuts the first round's policies learned?

    conda run --no-capture-output -n guidewire_rl python scripts/reward_check.py \\
        --seeds 1000-1019 --workers 12 --out runs/reward_check

Three controllers, each from the insertion point in the env of a config (default
``configs/two_devices_ica.yaml``), one episode per seed:

* ``good``: ``StagedController`` (``guidewire_rl/env/two_device/script.py``) as it is.
* ``wire_alone`` (first shortcut: the wire alone at the target, the catheter does not follow): the
  same until the wire is anchored at the target (the script's ``follow`` stage); from there the
  catheter is never pushed again and the wire is trimmed to the target and held. A wire that slips
  out of place goes back to the script's own stages to get there again.
* ``cath_lead`` (second shortcut: the catheter in front, the wire kept inside it, both pushed up
  together): the same until the catheter is seated in the trunk (the script's ``wire`` stage); from
  there the wire is kept ``D_INSIDE`` (10 mm) inside the catheter and both go up at 8 mm/s; when the
  catheter tip gains less than 1 mm along the route in 6 steps, or is in a wrong vessel, it backs
  off for 6 steps turning the catheter at 90 deg/s (alternating sides), then pushes on; it holds
  once the catheter tip is at the target's depth.

Every episode runs to the env's own end (the second tier, a hard failure, or MaxSteps); a script
that gives up sends zeros from there on, as an idle policy would. A step on which the simulation
went wrong ends the episode and is not counted (the trainers drop it).

Return = the sum over the episode of ``reward_formula.compute`` on the step's raw materials with the
config's coefficients (the env's reward), split into its terms (``reward_formula.components``). The
check passes when both shortcut scripts' mean returns are below the good script's.

Output: ``rows/<script>_<seed>.json`` and ``episodes/<script>_<seed>.npz`` (materials, rewards,
actions, the script's stage per step) per episode, ``summary.json`` and ``README.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "two_devices_ica.yaml"
SCRIPTS = ("good", "wire_alone", "cath_lead")


# ------------------------------------------------------------------ controllers ----
def controllers():
    """The three controllers (imported late: the parent process never builds an env)."""
    from guidewire_rl.env.two_device.script import CATH, ScriptParams, StagedController

    def without(*stages: str) -> ScriptParams:
        base = ScriptParams()
        return replace(base, stage_budgets={k: v for k, v in base.stage_budgets.items() if k not in stages})

    class WireAlone(StagedController):
        """The script up to the wire anchored at the target; then the catheter stays put."""

        def __init__(self) -> None:
            super().__init__(without("follow"))

        def _follow(self, r):
            p = self.p
            L = r.L[CATH]
            if self._wire_slipped(r):                   # lost its place: the script's way back there
                self.log.append(("follow:reanchor", self.steps))
                self._anchor = np.nan
                if self._ica_task(r):
                    self._go("ica", "", L)
                    return self._ica(r)
                self._go("wire", "aim" if p.aim else "fork", L)
                return self._wire(r)
            if self.substage != "lazy":
                self._go("follow", "lazy", L)
            depth_w = r.wire_chain if r.wire_carotid in ("cca", "ica") else -np.inf
            v_w = 0.0
            if np.isfinite(depth_w):
                err = r.target_depth - depth_w
                v_w = float(np.clip(p.k_trim_wire * err, -p.v_trim_wire, p.v_trim_wire))
                if abs(err) < p.wire_near_mm:
                    v_w = float(np.clip(v_w, -p.v_trim_wire_near, p.v_trim_wire_near))
            return v_w, 0.0, 0.0, 0.0                    # the catheter is never pushed

    class CathLead(StagedController):
        """The script up to the catheter seated in the trunk; then both up together, wire inside."""

        D_INSIDE = -10.0          # d = L_wire - L_cath: the wire's tip 10 mm inside the catheter
        V_PUSH = 8.0              # mm/s, the script's follow speed
        STALL_STEPS = 6
        BACK_STEPS = 6
        W_TURN = float(np.pi / 2)

        def __init__(self) -> None:
            super().__init__(without("wire"))

        def reset_state(self) -> None:
            super().reset_state()
            self._hist: List[float] = []
            self._back = 0
            self._turn_sign = 1.0

        def _wire(self, r):
            L = r.L[CATH]
            if self.substage != "lazy":
                self._go("wire", "lazy", L)
            keep = lambda v_c: self._keep_d(r, v_c, self.D_INSIDE)
            if self._back > 0:
                self._back -= 1
                return keep(-self.V_PUSH), 0.0, -self.V_PUSH, self._turn_sign * self.W_TURN
            depth_c = r.cath_chain if r.cath_carotid in ("cca", "ica") else -np.inf
            if np.isfinite(depth_c) and depth_c >= r.target_depth:     # there: hold
                self._hist = []
                return keep(0.0), 0.0, 0.0, 0.0
            wrong = self._cath_in_wrong_vessel(r) or r.cath_carotid in ("eca", "down")
            self._hist = (self._hist + [self.anatomy.s_route(r.cath_tip)])[-(self.STALL_STEPS + 1):]
            stalled = len(self._hist) > self.STALL_STEPS and max(self._hist) - self._hist[0] < 1.0
            if wrong or stalled:
                self.log.append((f"wire:lazy_{'wrong' if wrong else 'stall'}", self.steps))
                self._back = self.BACK_STEPS - 1
                self._hist = []
                self._turn_sign = -self._turn_sign
                return keep(-self.V_PUSH), 0.0, -self.V_PUSH, self._turn_sign * self.W_TURN
            return keep(self.V_PUSH), 0.0, self.V_PUSH, 0.0

    return {"good": lambda: StagedController(), "wire_alone": WireAlone, "cath_lead": CathLead}


# ------------------------------------------------------------------ one episode ----
def run_episode(env, make, script: str, seed: int, coeffs, max_steps: int, out_dir: Path) -> Dict[str, Any]:
    from guidewire_rl.env.two_device import reward_formula as rf
    from guidewire_rl.env.two_device.script import STAGES

    ctl = make()
    obs, info = env.reset(seed=seed)
    ctl.reset(env)
    mats, rewards, actions, stages = [], [], [], []
    end, t0 = "max_steps (safety stop)", time.time()
    for step in range(max_steps):
        action = ctl.act(env)
        stage_now = ctl.stage
        obs, reward, te, tr, info = env.step(action)
        if info.get("simulation_error") or info.get("wall_breached"):
            end = "simulation error" if info.get("simulation_error") else "wall breach"
            break
        mats.append(np.asarray(info["reward_materials"], np.float64))
        rewards.append(float(reward))
        actions.append(np.asarray(action, np.float32))
        stages.append(STAGES.index(stage_now) if stage_now in STAGES else -1)
        if (step + 1) % 100 == 0:
            so_far = float(rf.compute(np.stack(mats), coeffs).sum())
            print(f"  {script} {seed}: step {step + 1}, stage {ctl.stage}:{ctl.substage}, return {so_far:.2f}",
                  flush=True)
        if te or tr:
            end = "terminated" if te else "truncated"
            break
    M = np.stack(mats) if mats else np.zeros((0, len(rf.FIELDS)))
    col = lambda name: M[:, rf.INDEX[name]]
    r = rf.compute(M, coeffs)
    parts = {k: round(float(v.sum()), 4) for k, v in rf.components(M, coeffs).items()}
    np.savez_compressed(out_dir / "episodes" / f"{script}_{seed}.npz", materials=M.astype(np.float32),
                        fields=np.asarray(rf.FIELDS), reward=np.asarray(rewards, np.float32),
                        action=np.asarray(actions, np.float32).reshape(-1, 4), stage=np.asarray(stages, np.int8),
                        stage_names=np.asarray(STAGES), stage_log=np.asarray(json.dumps(ctl.log)))
    return dict(
        script=script, seed=seed, steps=len(M), end=end,
        return_=round(float(r.sum()), 4), env_return=round(float(np.sum(rewards)), 4), parts=parts,
        success=bool(col("success").any()) if len(M) else False,
        tier1=bool(col("tier1").any()) if len(M) else False,
        hard_failure=bool(col("hard_failure").any()) if len(M) else False,
        lead_end_mm=round(float(col("lead_mm")[-1]), 1) if len(M) else None,
        cath_force_max_N=round(float(np.nanmax(col("cath_force_N"))), 2) if len(M) else None,
        wire_force_max_N=round(float(np.nanmax(col("wire_force_N"))), 2) if len(M) else None,
        script_gave_up=bool(ctl.failed), give_up_reason=ctl.failure_reason, last_stage=f"{ctl.stage}:{ctl.substage}",
        lazy_events=sum(1 for name, _ in ctl.log if "lazy_" in str(name)), wall_s=round(time.time() - t0, 1))


def child(config: str, overrides: Sequence[str], jobs: Sequence[Tuple[str, int]], max_steps: int,
          out_dir: Path) -> None:
    from guidewire_rl.config import load_config
    from guidewire_rl.env import build_env
    from guidewire_rl.env.two_device import reward_formula as rf

    cfg = load_config(config, list(overrides))
    coeffs = rf.Coefficients.from_config(cfg["env"].get("reward_formula"))
    makers = controllers()
    env = build_env(cfg["env"])
    try:
        for script, seed in jobs:
            try:
                row = run_episode(env, makers[script], script, seed, coeffs, max_steps, out_dir)
            except Exception as exc:              # noqa: BLE001 - one bad episode must not stop the rest
                row = dict(script=script, seed=seed, steps=0, end=f"error: {exc!r}")
            print(f"{script} {seed}: {row.get('end')}, {row.get('steps')} steps, return {row.get('return_')}",
                  flush=True)
            (out_dir / "rows" / f"{script}_{seed}.json").write_text(json.dumps(row))
    finally:
        env.close()


# ------------------------------------------------------------------ parent ----
def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for part in (p.strip() for p in text.split(",")):
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def run_children(a, jobs: List[Tuple[str, int]], out_dir: Path) -> None:
    """One child per share of the jobs; a child that dies (SOFA can segfault after a numerical
    blow-up) or overruns its time is killed and its missing jobs tried once more."""
    base = [sys.executable, str(Path(__file__).resolve()), "--child", "--config", a.config, "--out", str(out_dir),
            "--max-steps", str(a.max_steps)]
    for item in a.set:
        base += ["--set", item]
    missing = lambda js: [j for j in js if not (out_dir / "rows" / f"{j[0]}_{j[1]}.json").exists()]
    for attempt in range(2):
        todo = missing(jobs)
        if not todo:
            return
        n = max(1, min(a.workers, len(todo)))
        procs = []
        for k in range(n):
            share = todo[k::n]
            cmd = base + ["--jobs", ",".join(f"{s}:{seed}" for s, seed in share)]
            procs.append((subprocess.Popen(cmd), share, time.time() + a.episode_timeout * len(share) + 300.0))
        while procs:
            time.sleep(5.0)
            still = []
            for proc, share, deadline in procs:
                if proc.poll() is not None:
                    continue
                if time.time() > deadline:
                    proc.kill()
                    proc.wait()
                    print(f"worker for {share} overran its time: killed", flush=True)
                    continue
                still.append((proc, share, deadline))
            procs = still
        lost = missing(todo)
        if lost:
            print(f"no result for {lost}" + ("; trying them once more" if attempt == 0 else ""), flush=True)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for script in SCRIPTS:
        ok = [r for r in rows if r.get("script") == script and "return_" in r]
        if not ok:
            continue
        ret = np.array([r["return_"] for r in ok])
        out[script] = dict(
            episodes=len(ok), return_mean=round(float(ret.mean()), 3), return_sd=round(float(ret.std(ddof=1)), 3)
            if len(ok) > 1 else 0.0, return_min=round(float(ret.min()), 3), return_max=round(float(ret.max()), 3),
            success=sum(r["success"] for r in ok), tier1=sum(r["tier1"] for r in ok),
            hard_failure=sum(r["hard_failure"] for r in ok), steps_mean=round(float(np.mean([r["steps"] for r in ok])), 1),
            parts_mean={k: round(float(np.mean([r["parts"][k] for r in ok])), 3) for k in ok[0]["parts"]},
            ends={e: sum(r["end"] == e for r in ok) for e in sorted({r["end"] for r in ok})})
    if "good" in out:
        g = out["good"]["return_mean"]
        out["verdict"] = {s: dict(below_good=bool(out[s]["return_mean"] < g),
                                  margin=round(g - out[s]["return_mean"], 3))
                          for s in ("wire_alone", "cath_lead") if s in out}
    return out


def write_readme(out_dir: Path, a, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines = [f"# Reward check ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"`scripts/reward_check.py`, config `{a.config}`" + (f" with `{' '.join(a.set)}`" if a.set else "")
             + f", seeds `{a.seeds}`. What the three scripts do: the script's docstring.", "",
             "| script | episodes | return mean (sd) | min - max | second tier | first tier | hard failure | steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in SCRIPTS:
        if s in summary:
            x = summary[s]
            lines.append(f"| {s} | {x['episodes']} | {x['return_mean']} ({x['return_sd']}) | {x['return_min']} - "
                         f"{x['return_max']} | {x['success']} | {x['tier1']} | {x['hard_failure']} | {x['steps_mean']} |")
    lines += ["", "Mean of each reward term per episode:", "",
              "| script | " + " | ".join(next(iter(summary[s]["parts_mean"] for s in SCRIPTS if s in summary))) + " |",
              "| --- " * (1 + len(next(iter(summary[s]["parts_mean"] for s in SCRIPTS if s in summary)))) + "|"]
    for s in SCRIPTS:
        if s in summary:
            lines.append(f"| {s} | " + " | ".join(str(v) for v in summary[s]["parts_mean"].values()) + " |")
    if "verdict" in summary:
        lines += ["", "Verdict (mean return below the good script's, by): "
                  + ", ".join(f"{s} {'yes' if v['below_good'] else 'NO'} ({v['margin']})"
                              for s, v in summary["verdict"].items())]
    lines += ["", "## Episodes", "", "| script | seed | return | end | steps | 2nd | 1st | lead at end mm | "
              "max force cath / wire N | gave up | last stage |", "| --- " * 11 + "|"]
    for r in sorted(rows, key=lambda r: (SCRIPTS.index(r["script"]), r["seed"])):
        lines.append(f"| {r['script']} | {r['seed']} | {r.get('return_')} | {r.get('end')} | {r.get('steps')} | "
                     f"{r.get('success')} | {r.get('tier1')} | {r.get('lead_end_mm')} | {r.get('cath_force_max_N')} / "
                     f"{r.get('wire_force_max_N')} | {r.get('give_up_reason') or '-'} | {r.get('last_stage')} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "runs" / "reward_check"))
    ap.add_argument("--seeds", default="1000-1019")
    ap.add_argument("--scripts", default=",".join(SCRIPTS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=1000, help="safety stop; the env's MaxSteps ends it first")
    ap.add_argument("--episode-timeout", type=float, default=2400.0, help="wall seconds per episode")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--jobs", default="", help=argparse.SUPPRESS)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    for sub in ("rows", "episodes"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if a.child:
        jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in a.jobs.split(",") if j]
        child(a.config, a.set, jobs, a.max_steps, out_dir)
        return 0
    scripts = [s for s in a.scripts.split(",") if s]
    unknown = sorted(set(scripts) - set(SCRIPTS))
    if unknown:
        raise SystemExit(f"unknown script(s) {unknown}; known: {SCRIPTS}")
    jobs = [(s, seed) for seed in parse_seeds(a.seeds) for s in scripts]   # interleaved: partial results compare
    t0 = time.time()
    run_children(a, jobs, out_dir)
    rows = []
    for s, seed in jobs:
        path = out_dir / "rows" / f"{s}_{seed}.json"
        rows.append(json.loads(path.read_text()) if path.exists() else
                    dict(script=s, seed=seed, steps=0, end="no result (worker died or was killed)"))
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(dict(config=a.config, overrides=a.set, seeds=a.seeds,
                                                          wall_s=round(time.time() - t0, 1), summary=summary,
                                                          rows=rows), indent=1))
    write_readme(out_dir, a, rows, summary)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
