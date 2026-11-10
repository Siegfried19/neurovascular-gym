"""What can be opened in the workbench: a phantom version times a simulation setting.

A **phantom** (``PHANTOMS``) is the vessel: collision wall + centerlines. Either a
``vessel_tree`` spec the simulator builds (``kind="sim"``) or a mesh + centerline file
that is not in the simulator (``kind="shape"``: drawn only, nothing runs). Add a version
by adding an entry. The versions themselves are described in
``guidewire_rl/env/vessels/biro_phantom/README.md``.

A **setting** is everything else of an environment config: physics, instrument,
observation, reward. Settings are read from ``configs/``: every config that runs on our
simulator with the lab phantom, configs that differ only in the vessel counted once (the
one closest to the root of the ``extends`` chain stands for them). ``SETTING_TEXT`` gives
them readable names; a new config shows under its file name.

The environment = the setting's config with ``env.scene.vessel_tree`` replaced by the
phantom's. Checkpoints fit when their run was trained on the same observation and action
(``compatible``); the physics may differ.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .i18n import Pair

# the checkout whose simulator code and configs the workbench runs: its own, or WORKBENCH_CODE (e.g. a
# worktree with work not committed yet); runs/ and recordings stay in the workbench's own checkout
ROOT = Path(os.environ.get("WORKBENCH_CODE") or Path(__file__).resolve().parents[3]).resolve()
VESSELS = ROOT / "guidewire_rl" / "env" / "vessels" / "biro_phantom"

PHANTOMS: Dict[str, Dict[str, Any]] = {
    # the three the work is done on (the physical phantom has only the chest and the head):
    # 模型内壁版 (model inner wall, files rebuilt_*), 内接球版 (inscribed spheres, files spheres_*), Wilson 版;
    # both phantoms from the model (STL) alone. The lab-input inscribed-sphere version (lab_cut,
    # phantom_arch_all) stays in the simulator (BiroPhantomArch version "lab_cut" / "lab") but is not listed.
    "rebuilt_cut": {
        "label": ("模型内壁版（头和胸腔）", "model inner wall (chest and head)"),
        "note": ("全部从 phantom 模型（STL）用我们的代码算出来：模型自己的内壁做碰撞面，中心线在这面墙里按"
                 "'尽量离墙远'的定义重算，路口按连接关系定；右臂截在和左锁骨下一样长的地方。",
                 "Everything from the phantom model (STL) with our code: the model's own inner wall as the "
                 "collision surface, centerlines recomputed through it (as far from the wall as possible), "
                 "junctions from how the lines connect; the right arm cut where the left subclavian ends."),
        "vessel_tree": {"type": "BiroPhantomArch", "version": "rebuilt_cut"},
    },
    "spheres_cut": {
        "label": ("内接球版（头和胸腔）", "inscribed spheres (chest and head)"),
        "note": ("全部从 phantom 模型（STL）用我们的代码算出来：沿模型内壁版的中心线放内接球（半径至少 1 mm），"
                 "拼起来做碰撞墙，拼法照 lab 之前的算法；中心线和模型内壁版相同；右臂截在和左锁骨下一样长的地方。",
                 "Everything from the phantom model (STL) with our code: the inscribed spheres along the model inner "
                 "wall version's centerlines (radius at least 1 mm), joined into the collision wall the way the lab's "
                 "earlier algorithm did; the same centerlines; the right arm cut where the left subclavian ends."),
        "vessel_tree": {"type": "BiroPhantomArch", "version": "spheres_cut"},
    },
    "wilson": {
        "label": ("Wilson 版：G13 的生成主动脉弓（不是 phantom）", "Wilson: G13's generated arch (not the phantom)"),
        "note": ("stEVE 按种子 30 生成的主动脉弓，Wilson 训 G13 用的血管，拿来对照。",
                 "The aortic arch stEVE generates from seed 30, the vessel Wilson trained G13 on; for comparison."),
        "vessel_tree": {"type": "AorticArch", "seed": 30, "scaling_xyzd": [1.0, 1.0, 1.0, 0.75]},
    },
    # extra: the whole model, right arm included
    "rebuilt": {
        "label": ("模型内壁版（完整模型，带右臂）", "model inner wall (whole model, right arm included)"),
        "note": ("同模型内壁版，没有截掉右臂。模型带右臂，实物没有。",
                 "As the model inner wall version, with the right arm. The model has it, the physical phantom does not."),
        "vessel_tree": {"type": "BiroPhantomArch", "version": "rebuilt"},
    },
    "spheres": {
        "label": ("内接球版（完整模型，带右臂）", "inscribed spheres (whole model, right arm included)"),
        "note": ("同内接球版，没有截掉右臂。",
                 "As the inscribed spheres version, with the right arm."),
        "vessel_tree": {"type": "BiroPhantomArch", "version": "spheres"},
    },
}
DEFAULT_PHANTOM, DEFAULT_SETTING, DEFAULT_FRICTION = "rebuilt_cut", "phantom_g13", 0.1

# instrument settings (器械设置): the friction is chosen apart (``FRICTIONS``), so configs that differ only
# in it (and in the vessel) are one entry
SETTING_TEXT: Dict[str, Pair] = {
    "phantom_g13": ("只有导丝：默认质量、6 ms 物理步（G13 的设置）", "guidewire only: default mass, 6 ms step (G13's)"),
    "phantom_real_mass": ("只有导丝：真实质量（镍钛密度）、2 ms 物理步、预判 2 mm",
                          "guidewire only: real mass (nitinol), 2 ms step, 2 mm alarm"),
    # its target is a window on a named branch: needs a phantom with branches and junctions (not Wilson's)
    "two_devices_round1": ("导丝 + 钩形导管：第一轮（默认质量、6 ms），目标右颈总",
                           "guidewire + hook catheter: round 1 (default mass, 6 ms), target in the right CCA"),
    "two_devices_real_mass": ("导丝 + 钩形导管：真实质量（2 ms，各按自己的半径碰墙），目标右颈总",
                              "guidewire + hook catheter: real mass (2 ms, each with its own radius), target in the right CCA"),
    "two_devices_ica": ("导丝 + 钩形导管：真实质量，目标右颈内（分叉上 20–40 mm，手写推法 11-06 调的那套）",
                        "guidewire + hook catheter: real mass, target in the right ICA (20-40 mm past the split; "
                        "the 11-06 scripted procedure's)"),
}
FRICTIONS: Dict[float, Pair] = {
    0.001: ("0.001（G13 原值）", "0.001 (G13's value)"),
    0.1: ("0.1（lab 的 phantom 灌了水）", "0.1 (the lab's phantom, filled with liquid)"),
}


@dataclass(eq=False)
class Phantom:
    key: str
    label: Pair
    note: Pair
    vessel_tree: Optional[Dict[str, Any]] = None     # kind "sim"
    shape: Optional[Dict[str, Path]] = None          # kind "shape"

    @property
    def kind(self) -> str:
        return "shape" if self.shape is not None else "sim"


@dataclass(eq=False)
class Setting:
    key: str                  # the representative config's file stem
    label: Pair
    note: Pair
    config_path: Path
    also: Tuple[str, ...] = ()   # other configs with the same setting (they differ only in the vessel or friction)


def phantoms(specs: Mapping[str, Dict[str, Any]] = PHANTOMS) -> List[Phantom]:
    out = []
    for key, spec in specs.items():
        shape = spec.get("shape")
        if shape is not None and not all(Path(p).exists() for p in shape.values()):
            continue                                    # data not on this machine
        out.append(Phantom(key=key, label=tuple(spec["label"]), note=tuple(spec["note"]),
                           vessel_tree=copy.deepcopy(spec.get("vessel_tree")), shape=shape))
    return out


def _first_comment(path: Path, limit: int = 220) -> str:
    """The first sentence of the config's leading comment (it says what the config changes)."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("# ").strip())
    text = " ".join(line for line in lines if line)
    end = text.find(". ")
    text = text if end < 0 else text[:end + 1]
    return text if len(text) <= limit else text[:limit - 1] + "…"


def runs_on_upstream(config: Mapping[str, Any]) -> bool:
    return not (config.get("env", {}).get("scene", {}) or {}).get("devices")


def _depth(path: Path) -> int:
    import yaml
    depth, current = 0, path
    while True:
        parent = (yaml.safe_load(current.read_text()) or {}).get("extends")
        if not parent:
            return depth
        depth, current = depth + 1, current.parent / parent


def _setting_body(config: Mapping[str, Any]) -> str:
    """The config without what is chosen apart: the vessel and the wall friction."""
    body = copy.deepcopy(dict(config))
    body.pop("run", None)
    scene = body.get("env", {}).get("scene", {})
    scene.pop("vessel_tree", None)
    (scene.get("simulation") or {}).pop("friction", None)
    return json.dumps(body, sort_keys=True, default=str)


def friction_of(config_path: Path) -> Optional[float]:
    """The wall friction a config file sets (None: not set)."""
    from guidewire_rl.config import load_config

    value = ((load_config(config_path)["env"]["scene"].get("simulation") or {}).get("friction"))
    return None if value is None else float(value)


def find_setting(key: str, all_settings: Sequence[Setting]) -> Optional[Setting]:
    """The setting a config name stands under (itself or one folded into it)."""
    return next((s for s in all_settings if s.key == key or key in s.also), None)


def settings(configs_dir: Path = ROOT / "configs") -> List[Setting]:
    """One entry per distinct setting among the configs that run the lab phantom on our simulator."""
    from guidewire_rl.config import load_config

    groups: Dict[str, List[Path]] = {}
    for path in sorted(Path(configs_dir).glob("*.yaml")):
        config = load_config(path)
        vessel = (config["env"]["scene"].get("vessel_tree") or {})
        if not runs_on_upstream(config) or vessel.get("type") != "BiroPhantomArch":
            continue
        groups.setdefault(_setting_body(config), []).append(path)
    out = []
    for paths in groups.values():
        paths = sorted(paths, key=lambda p: (_depth(p), p.stem))
        head = paths[0]
        note = _first_comment(head)
        out.append(Setting(key=head.stem, label=SETTING_TEXT.get(head.stem, (head.stem, head.stem)),
                           note=(note, note), config_path=head, also=tuple(p.stem for p in paths[1:])))
    return sorted(out, key=lambda s: (_depth(s.config_path), s.key))


def build_config(phantom: Phantom, setting: Setting, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    """The setting's config with the phantom's vessel; ``overrides`` as in ``--set``."""
    from guidewire_rl.config import load_config

    config = load_config(setting.config_path, overrides)
    if phantom.vessel_tree is not None:
        config["env"]["scene"]["vessel_tree"] = copy.deepcopy(phantom.vessel_tree)
    config.setdefault("run", {})["name"] = f"{phantom.key}+{setting.key}"
    return config


def _same_vessel(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    clean = lambda d: {k: v for k, v in d.items() if v is not None}
    a, b = clean(a), clean(b)
    if a.get("type") == "BiroPhantomArch":
        a.setdefault("centerline", "original")
    if b.get("type") == "BiroPhantomArch":
        b.setdefault("centerline", "original")
    return a == b


def locate(config_path: Path, all_phantoms: Sequence[Phantom],
           all_settings: Sequence[Setting]) -> Tuple[Optional[str], Optional[str]]:
    """(phantom key, setting key) that rebuild the given config; None for a part that is not listed
    (e.g. the lab's centerlines and wall, no longer in the workbench)."""
    from guidewire_rl.config import load_config

    config = load_config(config_path)
    vessel = config["env"]["scene"].get("vessel_tree") or {}
    phantom = next((p.key for p in all_phantoms if p.vessel_tree and _same_vessel(p.vessel_tree, vessel)), None)
    body = _setting_body(config)
    setting = next((s.key for s in all_settings if _setting_body(load_config(s.config_path)) == body), None)
    return phantom, setting


def compatible(env_config: Mapping[str, Any], run_config: Mapping[str, Any]) -> bool:
    """A checkpoint trained with ``run_config`` can act in ``env_config``: same inputs and outputs."""
    a, b = env_config.get("env", {}), run_config.get("env", {})
    return a.get("observation") == b.get("observation") and a.get("action") == b.get("action")


def run_config_of(checkpoint: Path) -> Optional[Dict[str, Any]]:
    """The config saved next to a checkpoint (``runs/<run>/config.yaml``), if any."""
    import yaml

    path = Path(checkpoint).resolve().parents[1] / "config.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())
