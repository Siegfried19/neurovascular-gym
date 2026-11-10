"""Sidebar content as HTML strings (pure functions, testable without viser).

The status table, the policy's output (each instrument's push and rotation) and the
contact forces; ``lang`` is ``"zh"`` or ``"en"``. A force label not named in
``FORCE_TEXT`` shows under its raw name.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Tuple

import numpy as np

from .frame import Frame, SceneInfo
from .i18n import TEXT, Pair, pick, tr

ACTION_TEXT: List[Tuple[Pair, str]] = [(("推送", "push"), "mm/s"), (("旋转", "rotate"), "rad/s")]
# the contact-force labels; an environment's other labels (e.g. wall clearances) are not shown
FORCE_TEXT: Dict[str, Tuple[Pair, str]] = {
    "distal_force": (("导丝尖端接触力", "guidewire tip contact force"), "N"),
    "catheter_tip_force": (("导管尖端接触力", "catheter tip contact force"), "N"),
}

RED, TEAL, BLUE = "#E24B4A", "#1D9E75", "#378ADD"


def _bar(value: float, lo: float = -1.0, hi: float = 1.0, color: str = BLUE) -> str:
    """Centered bar for a value in [lo, hi] (fills from 0)."""
    v = float(np.clip(value, lo, hi))
    zero = (0.0 - lo) / (hi - lo)
    pos = (v - lo) / (hi - lo)
    left, width = (min(zero, pos) * 100.0, abs(pos - zero) * 100.0)
    return (f'<div style="position:relative;height:6px;background:rgba(128,128,128,.18);border-radius:3px">'
            f'<div style="position:absolute;left:{zero * 100:.1f}%;top:-2px;width:1px;height:10px;background:rgba(128,128,128,.6)"></div>'
            f'<div style="position:absolute;left:{left:.1f}%;width:{width:.1f}%;height:6px;background:{color};border-radius:3px"></div></div>')


_TABLE = 'style="width:100%;border-collapse:collapse;font-size:12px"'
_CELL = 'style="padding:2px 4px;white-space:nowrap"'


def action_html(info: SceneInfo, frame: Frame, nominal: Optional[np.ndarray] = None, lang: str = "zh") -> str:
    if frame.action is None:
        return f'<div style="opacity:.6">{tr("no_action", lang)}</div>'
    rows = []
    per = len(ACTION_TEXT)
    several = len(info.instruments) > 1 and len(frame.action) == per * len(info.instruments)
    for i, value in enumerate(frame.action):
        if several:                         # instrument by instrument: push, rotate, push, rotate
            name, unit = pick(ACTION_TEXT[i % per][0], lang), ACTION_TEXT[i % per][1]
            name = f"{instrument_name(info, i // per, lang)} · {name}"
        elif i < per:
            name, unit = pick(ACTION_TEXT[i][0], lang), ACTION_TEXT[i][1]
        else:
            name, unit = tr("action_n", lang, i=i), ""
        lo, hi = float(info.action_low[i]), float(info.action_high[i])
        physical = lo + 0.5 * (float(value) + 1.0) * (hi - lo)
        extra = ""
        if nominal is not None and i < len(nominal) and abs(float(nominal[i]) - float(value)) > 1e-6:
            extra = f' <span style="opacity:.6">{tr("nominal", lang, value=f"{float(nominal[i]):+.2f}")}</span>'
        rows.append(f'<tr><td {_CELL}>{name}</td><td {_CELL} align="right"><code>{float(value):+.2f}</code>{extra}</td>'
                    f'<td {_CELL} align="right">{physical:+.1f} {unit}</td>'
                    f'<td style="width:36%;padding:2px 4px">{_bar(value, color=TEAL)}</td></tr>')
    return f"<table {_TABLE}>{''.join(rows)}</table>"


def force_names(frame: Frame):
    """The contact-force labels of a frame (measured or predicted), in order."""
    names = dict.fromkeys(list(frame.labels) + list(frame.predictions))
    return [n for n in names if n in FORCE_TEXT or n.endswith("force")]


def force_html(frame: Frame, lang: str = "zh") -> str:
    names = force_names(frame)
    if not names:
        return f'<div style="opacity:.6">{tr("no_forces", lang)}</div>'
    rows = [f'<tr style="opacity:.6"><td {_CELL}></td><td {_CELL} align="right">{tr("measured", lang)}</td>'
            f'<td {_CELL} align="right">{tr("predicted", lang)}</td></tr>']
    for name in names:
        label, unit = ((pick(FORCE_TEXT[name][0], lang), FORCE_TEXT[name][1]) if name in FORCE_TEXT
                       else (name, ""))
        measured = frame.labels.get(name)
        predicted = frame.predictions.get(name)
        fmt = lambda v: "—" if v is None else f"{v:.3f} {unit}"  # noqa: E731
        rows.append(f'<tr><td {_CELL}>{html.escape(label)}</td><td {_CELL} align="right"><code>{fmt(measured)}</code></td>'
                    f'<td {_CELL} align="right"><code>{fmt(predicted)}</code></td></tr>')
    return f"<table {_TABLE}>{''.join(rows)}</table>"


def instrument_name(info: SceneInfo, index: int, lang: str = "zh") -> str:
    name = info.instruments[index].name if index < len(info.instruments) else f"#{index + 1}"
    return tr(f"inst_name_{name}", lang) if f"inst_name_{name}" in TEXT else html.escape(name)


def instruments_inserted(info: SceneInfo, frame: Frame, lang: str = "zh") -> str:
    """"length · rotation" of each instrument; with two or more each is prefixed by its name."""
    inserted = [] if frame.inserted_mm is None else [float(v) for v in np.ravel(frame.inserted_mm)]
    rotation = [] if frame.rotation_rad is None else [float(v) for v in np.ravel(frame.rotation_rad)]
    count = max(len(inserted), len(rotation), 1)
    parts = []
    for i in range(count):
        length = f"{inserted[i]:.1f} mm" if i < len(inserted) else "—"
        angle = f"{np.degrees(rotation[i]) % 360:.0f}°" if i < len(rotation) else "—"
        if count == 1:
            return f"{length} · {angle}"
        parts.append(f"{instrument_name(info, i, lang)} {length} · {angle}")
    return "<br>".join(parts)


def status_html(info: SceneInfo, frame: Frame, policy_label: str, state: str, lang: str = "zh") -> str:
    """``policy_label`` and ``state`` are shown as given (already in ``lang``)."""
    max_steps = info.max_steps or 0
    if frame.success:
        outcome = f'<span style="color:{TEAL}">{tr("outcome_reached", lang)}</span>'
    elif frame.extras.get("error"):
        outcome = f'<span style="color:{RED}">{tr("outcome_error", lang)}</span>'
    elif frame.truncated:
        outcome = f'<span style="opacity:.7">{tr("outcome_truncated", lang)}</span>'
    else:
        outcome = tr("outcome_running", lang)
    tip, target = frame.tip_mm, frame.target_mm
    gap = "—" if tip is None or target is None else f"{float(np.linalg.norm(tip - target)):.1f} mm"
    inserted_rotation = instruments_inserted(info, frame, lang)
    if not info.observation_names:      # nothing runs here (a view-only shape)
        outcome = ""
    rows = [
        (tr("status", lang), f"{state} · {outcome}" if outcome else state),
        (tr("policy", lang), html.escape(policy_label)),
        (tr("episode_step", lang), tr("episode_step_value", lang, episode=frame.episode, step=frame.step)
         + (tr("step_limit", lang, limit=max_steps) if max_steps else "")),
        (tr("sim_time", lang), f"{frame.step * info.step_seconds:.1f} s"),
        (tr("step_reward", lang), f"{frame.reward:+.3f}"),
        (tr("return", lang), f"{frame.episode_return:+.3f}"),
        (tr("tip_to_target", lang), gap + tr("reach_within", lang, mm=f"{info.target_threshold_mm:.0f}")),
        (tr("inserted_rotation", lang), inserted_rotation),
        (tr("seed", lang), "—" if frame.seed is None else str(frame.seed)),
        (tr("scene", lang), html.escape(info.description)),
    ]
    if frame.extras.get("error"):
        rows.append((tr("error", lang),
                     f'<span style="color:{RED}">{html.escape(str(frame.extras["error"]))[:200]}</span>'))
    body = "".join(f'<tr><td {_CELL} style="opacity:.65;padding:2px 4px">{k}</td><td {_CELL}>{v}</td></tr>' for k, v in rows)
    return f"<table {_TABLE}>{body}</table>"
