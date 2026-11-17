"""Plain-file run logs: one JSON object per line, plus a console copy; optionally forwarded to
Weights & Biases (``sink``: ``wandb_sink.WandbSink``, ``logging.wandb``)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class RunLogger:
    """``<run_dir>/{updates,episodes,eval,events}.jsonl`` and ``console.log``."""

    def __init__(self, run_dir: str | Path, *, quiet: bool = False, sink=None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.sink = sink

    def write(self, stream: str, record: Mapping[str, Any]) -> None:
        line = json.dumps({k: _plain(v) for k, v in record.items()})
        with open(self.run_dir / f"{stream}.jsonl", "a") as handle:
            handle.write(line + "\n")
        if self.sink is not None:
            self.sink.log(stream, record)

    def close(self) -> None:
        if self.sink is not None:
            self.sink.finish()

    def print(self, message: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        if not self.quiet:
            print(stamped, flush=True)
        with open(self.run_dir / "console.log", "a") as handle:
            handle.write(stamped + "\n")


class MetricAverager:
    """Average update metrics over a window before logging them. A key missing from some
    updates (a rate only defined when its event occurred) is averaged over the updates that had it."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self.count = 0

    def add(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            self.sums[key] += float(value)
            self.counts[key] += 1
        self.count += 1

    def pop(self) -> Dict[str, float]:
        means = {k: v / self.counts[k] for k, v in self.sums.items() if self.counts[k]} if self.count else {}
        self.sums.clear()
        self.counts.clear()
        self.count = 0
        return means
