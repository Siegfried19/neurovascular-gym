"""The real devices we model, by name: what a menu lists and what a config's ``instruments:`` gets.

Each ``Product`` is one device with its spec (``type`` + parameters, as under ``instruments:``) and a
note on where its numbers come from. A spec repeats the class defaults the workbench's parameter page
shows (the masses), so the page reads them from the config; a test keeps the two equal. ``PRODUCTS`` lists the guidewires first, then the catheters.
``with_products`` puts chosen ones into an environment config in place of its own (the workbench's
device menus use it); the configs themselves are not touched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

Pair = Tuple[str, str]          # (中文, English)

WIRE, CATHETER = "guidewire", "catheter"


@dataclass(frozen=True)
class Product:
    key: str
    kind: str                   # WIRE or CATHETER
    label: Pair
    note: Pair
    spec: Dict[str, Any] = field(default_factory=dict)

    def instrument_spec(self) -> Dict[str, Any]:
        return copy.deepcopy(self.spec)


PRODUCTS: List[Product] = [
    Product("glidewire_stiff", WIRE,
            ("Glidewire 硬杆 0.035″ 带角度（实测，现在训练用的）", "Glidewire stiff 0.035″ angled (measured; the training's)"),
            ("Terumo GS3508。杆身 767、软头 10 N·mm²（Qiu 2023 实测），弯头半径 12.1 mm、72°，镍钛密度；"
             "和训练配置里手写的数相同。",
             "Terumo GS3508. Shaft 767, soft tip 10 N mm² (measured, Qiu 2023), tip bend 12.1 mm / 72°, nitinol "
             "density; the same numbers the training configs write out."),
            {"type": "GlidewireStiff", "mass_density": 6.45e-9, "flex_mass_density": 6.45e-9}),
    Product("cordis_sidewinder2", CATHETER,
            ("Cordis Sidewinder II 5F（SRD6109，Simmons 2）", "Cordis Sidewinder II 5F (SRD6109, Simmons 2)"),
            ("Cordis Tempo Aqua Sidewinder II。外径 1.65 mm 按包装；头形是 Qiu 2024 那根 Tempo SIM 2 照片的描线，"
             "硬度 486 N·mm² 是那根的实测；同一个弯、同一个硬度是推断。",
             "Cordis Tempo Aqua Sidewinder II. Outer 1.65 mm from the label; the head traced from Qiu 2024's photo "
             "of a Tempo SIM 2 and its measured 486 N mm²; that this one has the same curve and stiffness is "
             "inferred."),
            {"type": "CordisSidewinder2"}),
    Product("accuvu_omni_flush", CATHETER,
            ("Accu-Vu Omni Flush 5F（弯形、硬度估算）", "Accu-Vu Omni Flush 5F (curve and stiffness estimated)"),
            ("AngioDynamics Accu-Vu Omni Flush（13709703）。外径 1.8 mm 按包装；弯形（半径 7.5 mm 转 180°）、"
             "内径 0.97 mm、材料（按 Tempo 的尼龙，约 840 N·mm²）都是估算，等拍实物再换。",
             "AngioDynamics Accu-Vu Omni Flush (13709703). Outer 1.8 mm from the label; the curve (radius 7.5 mm, "
             "180°), the 0.97 mm lumen and the material (the Tempo's nylon, ~840 N mm²) are estimates until the "
             "catheter is photographed."),
            {"type": "AccuVuOmniFlush"}),
    Product("lab_hook", CATHETER,
            ("lab 钩形导管（现在训练用的）", "the lab's hooked catheter (the training's)"),
            ("lab 自编的钩（直 20 + 半径 8 弯 130° + 直 8 mm，外径 2.0 / 内径 1.0），不是哪个产品；"
             "硬度按 Tempo 实测改成 486 N·mm²，和训练配置相同。",
             "The lab's own hook (20 mm straight + 130° arc of radius 8 + 8 mm, outer 2.0 / inner 1.0), not a "
             "product; stiffness set to the measured Tempo's 486 N mm², as in the training configs."),
            {"type": "HookCatheter", "young_modulus": 660.0, "mass_density": 1.2e-9}),
]


def product(key: str) -> Product:
    for p in PRODUCTS:
        if p.key == key:
            return p
    raise KeyError(f"No device {key!r}; known: {[p.key for p in PRODUCTS]}")


def products(kind: str) -> List[Product]:
    return [p for p in PRODUCTS if p.kind == kind]


def with_products(config: Mapping[str, Any], wire: Optional[str] = None,
                  catheter: Optional[str] = None) -> Dict[str, Any]:
    """``config`` with its guidewire (``instruments[0]``) and / or catheter (``instruments[1]``) replaced
    by the named products; None keeps the config's own. The replaced entry's ``velocity_limit`` and
    ``name`` stay (they belong to the task's action scaling and node names); a catheter can only
    replace one the config has."""
    out = copy.deepcopy(dict(config))
    scene = out["env"]["scene"]
    instruments = scene["instruments"]
    for index, key, kind in ((0, wire, WIRE), (1, catheter, CATHETER)):
        if key is None:
            continue
        chosen = product(key)
        if chosen.kind != kind:
            raise ValueError(f"{key!r} is a {chosen.kind}, not a {kind}")
        if index >= len(instruments):
            raise ValueError(f"the config has no {kind} to replace")
        old = instruments[index] if isinstance(instruments[index], Mapping) else {"type": instruments[index]}
        new = chosen.instrument_spec()
        for kept in ("velocity_limit", "name"):
            if kept in old:
                new[kept] = copy.deepcopy(old[kept])
        instruments[index] = new
    return out
