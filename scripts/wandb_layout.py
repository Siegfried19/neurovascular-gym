"""Save the training dashboard as two wandb workspace views (docs/wandb_metrics.md section 4).

The curves come from ``guidewire_rl/train/wandb_sink.py`` (``DASHBOARD``); this script only lays them
out: which curves share a chart, stacked shares, log scale, readable line names. wandb's default
layout gives every number its own chart; saved views fix that once per project. Re-run it after
changing ``DASHBOARD``. Two views (user 2026-11-11):

  - "Round 2 core": the 21 charts to read every day (task, training episodes, world model and value,
    simulator, and since 2026-11-13 curriculum and schedule), chosen the way a world-model / RL
    practitioner reads a run (``CORE``). By default only
    these curves go to wandb (``logging.wandb.select: core``, ``wandb_sink.CORE_CURVES``);
  - "Round 2 full": every dashboard curve (``charts``), only useful for a run with ``select: dashboard``.

    conda run -n guidewire_rl python scripts/wandb_layout.py                     # a new core view
    conda run -n guidewire_rl python scripts/wandb_layout.py --core-url URL      # update that view in place
    conda run -n guidewire_rl python scripts/wandb_layout.py --full [--crutches] # also the full view
    conda run -n guidewire_rl python scripts/wandb_layout.py --check             # names only

Needs ``pip install wandb-workspaces`` (installed in guidewire_rl 2026-11-11) and a wandb login.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

from guidewire_rl.train.wandb_sink import CORE_CURVES, DASHBOARD, END_REASONS, GRIDS, PHASES, PROGRESS_BINS, TIMES

X = "env_step"
GAINS = ("progress", "follow", "tier1", "tier2")
PENALTIES = ("force_cath", "force_wire", "step", "wrong_vessel", "failure", "smooth")
AUX = ("wire_force", "cath_force", "wire_clearance", "cath_clearance")


def _sub(name: str, keys) -> List[str]:
    return [f"{name}/{k}" for k in dict.fromkeys(keys)]


# (panel, [(chart title, [curve names], options)]); options: plot_type, log_y, marks (eval points)
def charts(crutches: bool) -> List[Tuple[str, List[Tuple[str, List[str], Dict]]]]:
    e, l, d, c, s = "eval", "learning", "train", "crutches", "sim"
    pts = {"marks": True}
    panels = [
        (e, [
            ("Tier 2 success with 95% interval", [f"{e}/tier2_success", f"{e}/tier2_ci95_low",
                                                  f"{e}/tier2_ci95_high"], pts),
            ("Tier 1 rate with 95% interval", [f"{e}/tier1_rate", f"{e}/tier1_ci95_low", f"{e}/tier1_ci95_high"], pts),
            ("How episodes ended (share)", _sub(f"{e}/end", END_REASONS.values()), {"plot_type": "pct-area"}),
            ("Progress at the end, by 10 % bin (share)", _sub(f"{e}/progress_end", PROGRESS_BINS.values()),
             {"plot_type": "pct-area"}),
            ("Progress at the end (mean)", [f"{e}/progress_end_mean"], pts),
            ("Steps of successful episodes", [f"{e}/success_steps_mean", f"{e}/success_steps_median"], pts),
            ("Catheter tip force peak, N (p90, max)", [f"{e}/cath_force_peak_p90", f"{e}/cath_force_peak_max"], pts),
            ("Wire tip force peak, N (p90, max)", [f"{e}/wire_force_peak_p90", f"{e}/wire_force_peak_max"], pts),
            ("Wire lead at the end, mm (mean)", [f"{e}/lead_end_mean"], pts),
            ("Episodes ending with lead < 30 mm (share)", [f"{e}/lead_end_below_30mm"], pts),
            ("Events per episode", [f"{e}/wrong_vessel_per_ep", f"{e}/dislodged_per_ep"], pts),
            ("Return per episode", [f"{e}/return_mean"], pts),
            ("Steps per episode", [f"{e}/length_mean"], pts),
            ("Reward gains per episode", _sub(f"{e}/reward", GAINS), pts),
            ("Reward penalties per episode", _sub(f"{e}/reward", PENALTIES), pts),
            ("Shield and catheter activity (share of steps)", [f"{e}/shield_active_frac", f"{e}/cath_moving_frac"], pts),
            ("Wire lead, mm (median over steps)", [f"{e}/lead_p50"], pts),
        ]),
        (l, [
            ("World model: consistency loss", [f"{l}/consistency_loss"], {}),
            ("Reward prediction loss", [f"{l}/reward_loss"], {}),
            ("Value loss", [f"{l}/value_loss"], {}),
            ("Termination loss", [f"{l}/termination_loss"], {}),
            ("Termination head: recall, false positives", [f"{l}/termination_recall",
                                                           f"{l}/termination_false_positive"], {}),
            ("Gradient norm (log)", [f"{l}/model_grad_norm", f"{l}/policy_grad_norm"], {"log_y": True}),
            ("Policy entropy", [f"{l}/policy_entropy"], {}),
            ("Policy loss scale", [f"{l}/policy_scale"], {}),
            ("Prediction heads' losses", _sub(f"{l}/aux", AUX), {}),
            ("Policy loss", [f"{l}/policy_loss"], {}),
            ("Total loss", [f"{l}/total_loss"], {}),
            ("Q vs realized return (test episodes that ended)", [f"{l}/q_mean", f"{l}/g_mean"], pts),
            ("Q minus realized return", [f"{l}/q_minus_g"], pts),
            ("First step: Q vs realized return", [f"{l}/q0", f"{l}/g0"], pts),
            ("Test episodes in the Q comparison", [f"{l}/q_vs_return_episodes"], pts),
        ]),
        (d, [
            ("Training episodes: tier rates (last 100)", [f"{d}/train_tier2_rate", f"{d}/train_tier1_rate"], {}),
            ("Progress at the end (mean)", [f"{d}/progress_end_mean"], {}),
            ("Progress at the end, by 10 % bin (share)", _sub(f"{d}/progress_end", PROGRESS_BINS.values()),
             {"plot_type": "pct-area"}),
            ("Return per episode", [f"{d}/return_mean"], {}),
            ("Steps per episode", [f"{d}/length_mean"], {}),
            ("Reward gains per episode", _sub(f"{d}/reward", GAINS), {}),
            ("Reward penalties per episode", _sub(f"{d}/reward", PENALTIES), {}),
            ("Reward by kind per episode", [f"{d}/reward_hint", f"{d}/reward_penalty", f"{d}/reward_finish"], {}),
            ("Events per episode", [f"{d}/wrong_vessel_per_ep", f"{d}/dislodged_per_ep"], {}),
            ("Shield: share of steps changed", [f"{d}/shield_active_frac"], {}),
            ("Shield: mean change", [f"{d}/shield_delta_mean"], {}),
            ("Wire lead, mm (p10, p50, p90; median at the end)", [f"{d}/lead_p10", f"{d}/lead_p50", f"{d}/lead_p90",
                                                                f"{d}/lead_end_p50"], {}),
            ("Catheter moving (share of steps)", [f"{d}/cath_moving_frac"], {}),
        ]),
    ]
    if crutches:
        panels.append((c, [
            ("Start frontier (grid)", [f"{c}/start_frontier"], {}),
            ("Start grids: planned weights (share)", _sub(f"{c}/start_weight", GRIDS.values()), {"plot_type": "pct-area"}),
            ("Start grids: actual starts (share)", _sub(f"{c}/start_share", GRIDS.values()), {"plot_type": "pct-area"}),
            ("Success by start grid", _sub(f"{c}/success_by_start", GRIDS.values()), {}),
            ("Prefix: share of steps, gave-up rate", [f"{c}/prefix_step_share", f"{c}/prefix_gave_up_rate"], {}),
            ("Demonstration quota", [f"{c}/demo_fraction"], {}),
            ("Demonstrations in the replay (episodes)", [f"{c}/demo_episodes", f"{c}/self_demo_episodes"], {}),
            ("Schedule stage (0 curriculum, 1 demo ramp, 2 wrap-up, 3 hint ramp, 4 done)",
             [f"{c}/schedule_stage"], {}),
            ("Hint reward factor", [f"{c}/hint_scale"], {}),
            ("Force penalty factor (1 = the full penalty)", [f"{c}/force_scale"], {}),
            ("Ladder level", [f"{c}/ladder_level"], {}),
            ("Guard trips", [f"{c}/guard_trips"], {}),
            ("Residual box width", [f"{c}/residual_delta"], {}),
        ]))
    panels.append((s, [
        ("Env steps per second", [f"{s}/steps_per_s", f"{s}/steps_per_s_recent"], {}),
        ("Simulation errors and wall breaches (total)", [f"{s}/sim_errors_total", f"{s}/wall_breaches_total"], {}),
        ("Episodes ended by an error or a breach (share of the last 100)", [f"{s}/sim_error_rate",
                                                                            f"{s}/breach_rate"], {}),
        ("Where simulation errors happened (last 100 episodes)", _sub(f"{s}/sim_error_at", PHASES.values()),
         {"plot_type": "stacked-area"}),
        ("Simulator processes replaced (total)", [f"{s}/env_restarts", f"{s}/env_recycles"], {}),
        ("Memory, GB", [f"{s}/mem_available_gb", f"{s}/mem_main_gb", f"{s}/mem_worker_max_gb",
                        f"{s}/mem_workers_total_gb"], {}),
        ("Main process time (share)", _sub(f"{s}/time", TIMES.values()), {"plot_type": "pct-area"}),
        ("Episodes (total)", [f"{s}/episodes_total"], {}),
        ("Network updates (total)", [f"{s}/updates_total"], {}),
        ("Replay size (steps)", [f"{s}/replay_size"], {}),
        ("Evaluation time, minutes", [f"{s}/eval_minutes"], {"marks": True}),
    ]))
    return panels


# The core view: what to read every day, in the order a world-model / RL practitioner asks it (user
# 2026-11-11): did it learn the task (test episodes), what the training episodes do, is the world model
# and the value sound, is the simulator holding up. (section, [(title, [curve], {curve: line name}, options)])
CORE = [
    ("Task (test episodes)", [
        ("Tier 2 success, 95% interval", ["eval/tier2_success", "eval/tier2_ci95_low", "eval/tier2_ci95_high"],
         {"eval/tier2_success": "tier 2 success", "eval/tier2_ci95_low": "interval low",
          "eval/tier2_ci95_high": "interval high"}, {"marks": True}),
        ("Tier 1 rate, 95% interval", ["eval/tier1_rate", "eval/tier1_ci95_low", "eval/tier1_ci95_high"],
         {"eval/tier1_rate": "tier 1 rate", "eval/tier1_ci95_low": "interval low",
          "eval/tier1_ci95_high": "interval high"}, {"marks": True}),
        ("How the test episodes ended (share)", [f"eval/end/{k}" for k in END_REASONS.values()],
         {f"eval/end/{k}": k.replace("_", " ") for k in END_REASONS.values()}, {"plot_type": "pct-area"}),
        ("Progress at the end of an episode (mean)", ["eval/progress_end_mean", "train/progress_end_mean"],
         {"eval/progress_end_mean": "test episodes", "train/progress_end_mean": "training episodes"},
         {"marks": True}),
        ("Tip force peak, p90 over the test episodes (N)", ["eval/cath_force_peak_p90", "eval/wire_force_peak_p90"],
         {"eval/cath_force_peak_p90": "catheter", "eval/wire_force_peak_p90": "wire"}, {"marks": True}),
    ]),
    ("Training episodes (last 100)", [
        ("Return per episode", ["train/return_mean"], {"train/return_mean": "return"}, {}),
        ("Reward by kind per episode", ["train/reward_hint", "train/reward_penalty", "train/reward_finish"],
         {"train/reward_hint": "hints (progress + follow)", "train/reward_penalty": "penalties (force, step, wrong vessel)",
          "train/reward_finish": "finish (tiers - hard failure)"}, {}),
        ("Wire lead over the catheter (mm)", ["train/lead_p10", "train/lead_p50", "train/lead_p90"],
         {"train/lead_p10": "p10", "train/lead_p50": "median", "train/lead_p90": "p90"}, {}),
        ("Shield changed the action (share of steps)", ["train/shield_active_frac"],
         {"train/shield_active_frac": "shield"}, {}),
    ]),
    ("World model and value", [
        ("World model prediction loss (log)", ["learning/consistency_loss"],
         {"learning/consistency_loss": "next-state prediction"}, {"log_y": True}),
        ("Reward prediction loss (log)", ["learning/reward_loss"], {"learning/reward_loss": "reward prediction"},
         {"log_y": True}),
        ("Q vs realized return (test episodes that ended)", ["learning/q_mean", "learning/g_mean"],
         {"learning/q_mean": "Q (estimated)", "learning/g_mean": "realized return"}, {"marks": True}),
        ("Gradient norm (log)", ["learning/model_grad_norm", "learning/policy_grad_norm"],
         {"learning/model_grad_norm": "world model + value", "learning/policy_grad_norm": "policy"}, {"log_y": True}),
        ("Policy entropy", ["learning/policy_entropy"], {"learning/policy_entropy": "entropy"}, {}),
    ]),
    ("Simulator", [
        ("Env steps per second", ["sim/steps_per_s_recent", "sim/steps_per_s"],
         {"sim/steps_per_s_recent": "recent", "sim/steps_per_s": "since the start"}, {}),
        ("Simulation errors and wall breaches (total)", ["sim/sim_errors_total", "sim/wall_breaches_total"],
         {"sim/sim_errors_total": "simulation errors", "sim/wall_breaches_total": "wall breaches"}, {}),
    ]),
    # user 2026-11-13: the start curriculum and the schedule on the core view (empty in a run without them)
    ("Curriculum and schedule", [
        ("Start frontier (grid; 10 = next to the goal, 0 = insertion point)", ["crutches/start_frontier"],
         {"crutches/start_frontier": "frontier"}, {}),
        ("Training success by start grid (last 100 episodes)",
         [f"crutches/success_by_start/{g}" for g in GRIDS.values()],
         {f"crutches/success_by_start/{g}": g.replace("_", " ") for g in GRIDS.values()}, {}),
        ("Schedule stage (0 curriculum, 1 demo ramp, 2 wrap-up, 3 hint ramp, 4 done)", ["crutches/schedule_stage"],
         {"crutches/schedule_stage": "stage"}, {}),
        ("Demonstration quota (share of every batch)", ["crutches/demo_fraction"],
         {"crutches/demo_fraction": "quota"}, {}),
        ("Force penalty factor (light first; 1 = the full penalty)", ["crutches/force_scale"],
         {"crutches/force_scale": "factor"}, {}),
        ("Guard trips and ladder level", ["crutches/guard_trips", "crutches/ladder_level"],
         {"crutches/guard_trips": "guard trips", "crutches/ladder_level": "ladder level"}, {}),
    ]),
]


def dashboard_names() -> List[str]:
    """Every curve name ``DASHBOARD`` can send (dict sources expanded with their known keys)."""
    known = {"eval/end": END_REASONS.values(), "eval/progress_end": PROGRESS_BINS.values(),
             "train/progress_end": PROGRESS_BINS.values(), "eval/reward": GAINS + PENALTIES,
             "train/reward": GAINS + PENALTIES, "learning/aux": AUX,
             "crutches/start_weight": GRIDS.values(), "crutches/start_share": GRIDS.values(),
             "crutches/success_by_start": GRIDS.values(), "sim/sim_error_at": PHASES.values(),
             "sim/time": TIMES.values()}
    names: List[str] = []
    for name, _, source in DASHBOARD:
        names += _sub(name, known[name]) if name in known else [name]
    return names


def check(crutches: bool = True) -> Tuple[List[str], List[str]]:
    """(curves in a chart of either view but not on the dashboard, dashboard curves in no full-view chart)."""
    laid = [y for _, rows in charts(crutches) for _, ys, _ in rows for y in ys]
    core = [y for _, rows in CORE for _, ys, _, _ in rows for y in ys]
    names = dashboard_names()
    if set(core) != set(CORE_CURVES):
        raise ValueError(f"the core view and wandb_sink.CORE_CURVES differ: "
                         f"{sorted(set(core) ^ set(CORE_CURVES))}")
    return sorted(set(laid + core) - set(names)), sorted(set(names) - set(laid))


def _plot(wr, title: str, ys: Sequence[str], opt: Dict, titles: Dict[str, str] | None = None):
    kwargs = dict(title=title, x=X, y=list(ys), max_runs_to_show=10)
    if opt.get("plot_type"):
        kwargs["plot_type"] = opt["plot_type"]
    if opt.get("log_y"):
        kwargs["log_y"] = True
    if opt.get("marks"):
        kwargs["line_marks"] = {y: "points" for y in ys}
    if titles:
        kwargs["line_titles"] = dict(titles)
    return wr.LinePlot(**kwargs)


def build_core(entity: str, project: str, name: str):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    sections = [ws.Section(name=section, panels=[_plot(wr, title, ys, opt, titles) for title, ys, titles, opt in rows],
                           is_open=True) for section, rows in CORE]
    return ws.Workspace(entity=entity, project=project, name=name, sections=sections)


def build(entity: str, project: str, name: str, crutches: bool):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    sections = [ws.Section(name=panel, panels=[_plot(wr, title, ys, opt) for title, ys, opt in rows], is_open=True)
                for panel, rows in charts(crutches)]
    return ws.Workspace(entity=entity, project=project, name=name, sections=sections)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", default="neurovascular-gym")
    parser.add_argument("--entity", default=None, help="default: the logged-in user's entity")
    parser.add_argument("--name", default=None, help="full view name (default: Round 2 full[, crutches])")
    parser.add_argument("--crutches", action="store_true", help="the full view with the crutches panel")
    parser.add_argument("--full", action="store_true", help="also save the full view")
    parser.add_argument("--core-url", default=None, help="update this saved core view instead of a new one")
    parser.add_argument("--check", action="store_true", help="only compare the layout with DASHBOARD")
    args = parser.parse_args(argv)
    extra, missing = check(crutches=True)
    if extra or missing:
        raise SystemExit(f"layout and DASHBOARD differ: in charts only {extra}; on the dashboard only {missing}")
    if args.check:
        print("layout covers every dashboard curve")
        return
    import wandb
    entity = args.entity or wandb.Api().default_entity
    core = build_core(entity, args.project, "Round 2 core")
    if args.core_url:                                  # same link, new content
        import wandb_workspaces.workspaces as ws
        saved = ws.Workspace.from_url(args.core_url)
        saved.sections = core.sections
        core = saved
    core = core.save()
    print(f"saved core view: {getattr(core, 'url', core)}")
    if args.full:
        name = args.name or ("Round 2 full, crutches" if args.crutches else "Round 2 full")
        full = build(entity, args.project, name, args.crutches).save()
        print(f"saved full view: {getattr(full, 'url', full)}")


if __name__ == "__main__":
    main()
