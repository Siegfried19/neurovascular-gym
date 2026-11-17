"""The policy's own start library (user 2026-11-12: "let the policy's own far-reaching episodes be the
starting points"; work log 2026-11-12 14:xx).

Training episodes save a snapshot each time they first reach a new progress grid (the env side,
``two_device.snapshot.EpisodeSnapshots``); the files land in ``<run>/starts/grid_KK/`` and the trainer
keeps them here, by grid. The start curriculum (``curriculum.StartCurriculum``, ``supply: self``) draws
a grid among those that hold at least ``min_entries`` states; the episode then starts from one of them,
picked at random (``sample``). No script, no recording, no demonstration: every start is a state the
policy reached on its own, so the same machinery works on any vessel tree.

Each grid keeps the newest ``cap`` states (older files are deleted): the library follows the policy as
it gets better and stays small (a snapshot is a few tens of kB). Nothing is saved in the checkpoint;
after a restart the library is read back from the folder (``scan``).

Fixed library (``supply: demos``, user 2026-11-12, work log 19:26): the same folder layout, made once
beforehand from the scripted demonstrations (``scripts/record_demos.py --starts``: one state per grid
of every successful demonstration, ``collect_episode_states``) and named by
``training.start_curriculum.library``. A fixed library is read, never changed: every file it finds is
kept (no ``cap``), nothing is deleted, and states the training episodes reach are not added, so two
runs can share it. The envs are not asked to save states at all (``saving_to`` is None).
"""

from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional, Set

import numpy as np


def collect_episode_states(staging: str | Path, library: str | Path, keep: bool) -> int:
    """After one recorded episode: move the states it saved (``staging/grid_KK/*.npz``, written by
    ``EpisodeSnapshots``) into the library when the episode is to be kept (a successful demonstration),
    and remove the staging folder either way. Returns how many states went into the library."""
    staging, library = Path(staging), Path(library)
    moved = 0
    if keep and staging.is_dir():
        for path in sorted(staging.glob("grid_*/*.npz")):
            target = library / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            moved += 1
    shutil.rmtree(staging, ignore_errors=True)
    return moved


class StartLibrary:
    def __init__(self, directory: str | Path, *, bins: int = 10, cap: int = 100, min_entries: int = 5,
                 seed: int = 0, fixed: bool = False) -> None:
        self.directory = Path(directory)
        self.bins, self.cap, self.min_entries = int(bins), int(cap), int(min_entries)
        self.fixed = bool(fixed)                          # made beforehand: read, never changed
        self.entries: Dict[int, Deque[Dict[str, Any]]] = {k: deque() for k in range(1, self.bins + 1)}
        self.rng = np.random.default_rng(seed)
        self.added = 0
        self.scan()

    def scan(self) -> None:
        """Read back the files already in the folder (after a restart), oldest first."""
        if not self.directory.exists():
            return
        for k in range(1, self.bins + 1):
            files = sorted((self.directory / f"grid_{k:02d}").glob("*.npz"), key=lambda p: p.stat().st_mtime)
            for path in files:
                self._keep(k, {"path": str(path), "grid": k})

    @property
    def saving_to(self) -> Optional[str]:
        """Where the training episodes save the states they reach; None for a fixed library."""
        return None if self.fixed else str(self.directory)

    def _keep(self, grid: int, entry: Mapping[str, Any]) -> None:
        queue = self.entries[grid]
        queue.append(dict(entry))
        while not self.fixed and len(queue) > self.cap:
            old = queue.popleft()
            try:
                Path(old["path"]).unlink()
            except OSError:
                pass

    def add(self, saved: Mapping[str, Any]) -> None:
        """One state an env saved (``EpisodeSnapshots.after_step``'s report). A fixed library takes none."""
        grid = int(saved["grid"])
        if not self.fixed and 1 <= grid <= self.bins:
            self._keep(grid, saved)
            self.added += 1

    def available(self) -> Set[int]:
        """Grids that hold enough states to start from."""
        return {k for k, q in self.entries.items() if len(q) >= self.min_entries}

    def sample(self, grid: int) -> Optional[str]:
        """A random state of ``grid`` (its file), None when the grid has none."""
        queue = self.entries.get(int(grid))
        if not queue:
            return None
        return str(queue[int(self.rng.integers(len(queue)))]["path"])

    def counts(self) -> Dict[str, int]:
        return {str(k): len(q) for k, q in self.entries.items()}

    @classmethod
    def from_config(cls, spec: Optional[Mapping[str, Any]], run_dir: str | Path, seed: int) -> Optional["StartLibrary"]:
        """From ``training.start_curriculum``: supply ``self`` = the run's own ``<run>/starts``; supply
        ``demos`` = the fixed library its ``library`` names (relative paths from the workspace root, as
        ``training.demos.dir``); None otherwise."""
        if not spec or spec.get("enabled") is False or spec.get("supply") not in ("self", "demos"):
            return None
        bins, min_entries = int(spec.get("bins", 10)), int(spec.get("library_min", 5))
        if spec["supply"] == "self":
            return cls(Path(run_dir) / "starts", bins=bins, cap=int(spec.get("library_cap", 100)),
                       min_entries=min_entries, seed=seed)
        from .demos import resolve_dir
        if not spec.get("library"):
            raise ValueError("start_curriculum.supply 'demos' needs library: the folder of saved states")
        folder = resolve_dir(spec["library"])
        made = cls(folder, bins=bins, min_entries=min_entries, seed=seed, fixed=True)
        if not made.available():
            raise FileNotFoundError(
                f"start_curriculum.library {folder}: no grid holds {min_entries} states "
                f"(found {made.counts()}); make it with scripts/record_demos.py --starts {folder}")
        return made
