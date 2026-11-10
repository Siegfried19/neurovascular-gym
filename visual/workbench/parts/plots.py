"""Live plots over the current episode (viser's uPlot widget).

``PLOTS`` lists what is drawn; each entry pulls its series out of an
``EpisodeLog``, so adding a plot is adding an entry here. Titles and legends
are ``(中文, English)`` pairs. The two instruments' output and contact force are
drawn apart; a plot in ``SHOWN_IF`` shows only when its scene has what it needs
(the catheter's plots: a second instrument).

What reaches the browser has no NaN in it (uPlot's fit-to-data range breaks on a NaN
and the plot stays empty): a series with no value yet (the model's predicted force when
no model predicts one) is left out of its plot, gaps in the others are filled with the
value before. Only the data changes from step to step: a change of a plot's options
(series, scales) makes uPlot build the chart anew, which flashes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .frame import EpisodeLog
from .i18n import Pair, pick

Series = Tuple[Pair, str, Callable]  # (legend, colour, frame -> value)


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def _force(label: str) -> List[Series]:
    return [(("实测", "measured"), "#E24B4A", lambda f: _nan(f.labels.get(label))),
            (("模型预测", "predicted"), "#378ADD", lambda f: _nan(f.predictions.get(label)))]


def _action(first: int) -> List[Series]:
    """Push and rotation of one instrument: action dimensions ``first`` and ``first + 1``."""
    def dim(i):
        return lambda f: _nan(None if f.action is None or len(f.action) <= i else f.action[i])
    return [(("推送", "push"), "#1D9E75", dim(first)), (("旋转", "rotate"), "#EF9F27", dim(first + 1))]


PLOTS: Dict[str, Tuple[Pair, List[Series]]] = {
    "action_guidewire": (("导丝动作（归一化）", "Guidewire action (normalised)"), _action(0)),
    "action_catheter": (("导管动作（归一化）", "Catheter action (normalised)"), _action(2)),
    "force_guidewire": (("导丝尖端接触力 N", "Guidewire tip contact force (N)"), _force("distal_force")),
    "force_catheter": (("导管尖端接触力 N", "Catheter tip contact force (N)"), _force("catheter_tip_force")),
    "reward": (("奖励", "Reward"), [
        (("这一步", "this step"), "#888780", lambda f: f.reward),
        (("累计", "so far"), "#7F77DD", lambda f: f.episode_return),
    ]),
    "distance": (("尖端到目标 mm（直线）", "Tip to target (mm, straight line)"), [
        (("距离", "distance"), "#EF9F27", lambda f: _nan(None if f.tip_mm is None or f.target_mm is None
                                                         else np.linalg.norm(f.tip_mm - f.target_mm))),
    ]),
}
STEP_AXIS: Pair = ("步", "step")
SHOWN_IF: Dict[str, Callable] = {"action_catheter": lambda info: len(info.instruments) > 1,
                                 "force_catheter": lambda info: len(info.instruments) > 1}


def plot_data(log: EpisodeLog, key: str, upto: int | None = None) -> Tuple[np.ndarray, ...]:
    """(steps, series...) up to frame ``upto``; steps where every series is missing are left out
    (step 0 has no action or force yet), since uPlot draws nothing when a series starts with NaN."""
    frames = log.frames[: (upto + 1) if upto is not None else None]
    getters = [get for _, _, get in PLOTS[key][1]]
    x = np.array([f.step for f in frames], dtype=np.float64)
    ys = np.array([[get(f) for get in getters] for f in frames], dtype=np.float64).reshape(len(frames), len(getters))
    keep = ~np.all(np.isnan(ys), axis=1)
    if not keep.any():
        return (np.zeros(1), *[np.zeros(1) for _ in getters])
    return (x[keep], *[ys[keep, i] for i in range(len(getters))])


def drawable(data: Tuple[np.ndarray, ...]) -> Tuple[Tuple[int, ...], Tuple[np.ndarray, ...]]:
    """(the series that have a value, their data with gaps filled) for one plot."""
    x, ys = data[0], data[1:]
    shown = tuple(i for i, y in enumerate(ys) if np.isfinite(y).any()) or (0,)
    out = []
    for i in shown:
        y = np.array(ys[i], dtype=np.float64)
        ok = np.isfinite(y)
        if not ok.any():
            y[:] = 0.0
        else:
            last = np.maximum.accumulate(np.where(ok, np.arange(len(y)), -1))
            y = np.where(last >= 0, y[np.maximum(last, 0)], y[ok][0])     # fill a gap with the value before
        out.append(y)
    return shown, (x, *out)


class PlotPanel:
    def __init__(self, server, keys: Sequence[str] = tuple(PLOTS), lang: str = "zh") -> None:
        from viser import uplot

        self._server = server
        self._handles, self._series, self._shown = {}, {}, {}
        self._step = uplot.Series(label=pick(STEP_AXIS, lang))
        for key in keys:
            title, series = PLOTS[key]
            self._series[key] = [uplot.Series(label=pick(name, lang), stroke=color, width=2)
                                 for name, color, _ in series]
            self._shown[key] = (0,)
            self._handles[key] = server.gui.add_uplot(
                data=(np.zeros(1), np.zeros(1)), series=(self._step, self._series[key][0]),
                title=pick(title, lang), aspect=2.0, legend=uplot.Legend(show=True),
                scales={"x": uplot.Scale(time=False, auto=True), "y": uplot.Scale(auto=True)})

    def update(self, log: EpisodeLog, upto: int | None = None) -> None:
        for key, handle in self._handles.items():
            shown = SHOWN_IF.get(key, lambda info: True)(log.info)
            if handle.visible != shown:
                handle.visible = shown
            if not shown:
                continue
            which, data = drawable(plot_data(log, key, upto))
            with self._server.atomic():          # series and data change together
                if which != self._shown[key]:    # rare: a series gets its first value
                    self._shown[key] = which
                    handle.series = (self._step, *[self._series[key][i] for i in which])
                handle.data = data
