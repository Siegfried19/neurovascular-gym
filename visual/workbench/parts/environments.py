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
    # no vessel: the instruments alone in empty space, live (parts/bench.py)
    "bench": {
        "label": ("无血管：器械仿真（空中）", "no vessel: instrument bench (in empty space)"),
        "note": ("没有血管，只有导丝和导管在空中，实时推、转：看导管的弯、导丝穿过导管。器械按下面两个菜单选，"
                 "物理（物理步、预判距离、质量）用所选设置的；观测、奖励、目标都不用。背景格子 1 cm 一格。",
                 "No vessel: the guidewire and the catheter alone in empty space, pushed and turned live, to see "
                 "the catheter's curve and the wire passing through it. The devices come from the two menus below, "
                 "the physics (step, alarm distance, masses) from the chosen setting; observation, reward and target "
                 "are not used. The background grid has 1 cm cells."),
        "bench": True,
    },
}
NO_CATHETER = "none"                 # the catheter menu's "no catheter" (instrument bench only)
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
    "two_devices_ica": ("导丝 + 钩形导管：第二轮配置，目标右颈内（分叉上 35–45 mm；两档终点、公式奖励、无规则壳）",
                        "guidewire + hook catheter: round-two config, target in the right ICA (35-45 mm past the "
                        "split; two-tier finish, formula reward, rule-free shell)"),
    # the other three ways to train round two (docs/training_framework.md, "四种训练方法") and the tried variant
    "two_devices_ica_nodemo": ("第二轮 · 不带拐棍（每局从插入点起，无示范）",
                               "round two · no crutches (every episode from the insertion point, no demonstrations)"),
    "two_devices_ica_nodemo_best": ("第二轮 · 不带拐棍 + 只奖励新进展（试过、失败，留作对照）",
                                    "round two · no crutches + only new progress pays (tried, failed; kept for reference)"),
    "two_devices_ica_selfstart": ("第二轮 · 自己的局当起点（起点库来自策略自己走到过的状态）",
                                  "round two · the policy's own states as starting points"),
    "two_devices_ica_demostart": ("第二轮 · 示范当起点（起点库来自手写推法的局）",
                                  "round two · states saved along the scripted demonstrations as starting points"),
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
    bench: bool = False                              # kind "bench": no vessel, instruments in empty space

    @property
    def kind(self) -> str:
        if self.bench:
            return "bench"
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
                           vessel_tree=copy.deepcopy(spec.get("vessel_tree")), shape=shape,
                           bench=bool(spec.get("bench", False))))
    return out


def instrument_count(setting: Setting) -> int:
    """How many instruments the setting's config has (1: guidewire only, 2: guidewire + catheter)."""
    from guidewire_rl.config import load_config

    return len(load_config(setting.config_path)["env"]["scene"].get("instruments") or [])


def with_devices(config: Mapping[str, Any], phantom: Phantom, wire: Optional[str] = None,
                 catheter: Optional[str] = None) -> Dict[str, Any]:
    """The config with the devices picked in the menus (keys of ``guidewire_rl.env.instruments.catalog``;
    None keeps the setting's own). In a vessel the devices are only swapped, never added or removed
    (the task's observation and action are built for that many); on the instrument bench a catheter
    can be added to a guidewire-only setting or taken away (``NO_CATHETER``)."""
    from guidewire_rl.env.instruments import catalog

    out = copy.deepcopy(dict(config))
    instruments = out["env"]["scene"]["instruments"]
    if phantom.kind != "bench":
        if catheter == NO_CATHETER or len(instruments) < 2:
            catheter = None
    elif catheter == NO_CATHETER:
        del instruments[1:]
        catheter = None
    elif catheter is not None and len(instruments) < 2:
        instruments.append(catalog.product(catheter).instrument_spec())
        catheter = None
    return catalog.with_products(out, wire, catheter)


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
