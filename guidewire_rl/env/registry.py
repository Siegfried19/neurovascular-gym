"""Name → class lookup for environment components.

A YAML ``type:`` is resolved first among our own registered components, then
among stEVE's (``eve``) classes of the same kind. Constructor arguments that
name a scene object (``intervention``, ``vessel_tree``, ``pathfinder`` …) are
injected automatically, so YAML only lists the component's own parameters.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, List, Mapping, Union

EVE_MODULES: Dict[str, Union[str, None]] = {
    "vessel_tree": "eve.intervention.vesseltree",
    # Wilson's eve calls the wire a "device", upstream stEVE an "instrument";
    # both kinds resolve to whichever of the two modules this eve has.
    "device": "eve.intervention.device",
    "instrument": "eve.intervention.instrument",
    "simulation": "eve.intervention.simulation",
    "fluoroscopy": "eve.intervention.fluoroscopy",
    "target": "eve.intervention.target",
    "intervention": "eve.intervention",
    "start": "eve.start",
    "pathfinder": "eve.pathfinder",
    "observation": "eve.observation",
    "observation_wrapper": "eve.observation.wrapper",
    "reward": "eve.reward",
    "terminal": "eve.terminal",
    "truncation": "eve.truncation",
    "info": "eve.info",
    "visualisation": "eve.visualisation",
    # Ours only.
    "label": None,
    "action": None,
    "monitor": None,     # shared per-step parts (builder.py, env.monitors)
}

_OWN: Dict[str, Dict[str, type]] = {kind: {} for kind in EVE_MODULES}

# Our built-in components, imported on first use (the force label needs SOFA,
# so importing it eagerly would make every test depend on a SOFA install).
# The simulator fixes (simulator_modifications) take stEVE's class names and only
# register themselves with upstream stEVE; with Wilson's eve the name falls
# through to stEVE's own class.
_BUILTIN_MODULES: Dict[str, Dict[str, str]] = {
    "observation": {"VesselGeometry": "guidewire_rl.env.observations.vessel_geometry",
                    "Tracking2D": "simulator_modifications.observations"},
    "observation_wrapper": {"NormalizeTracking2DEpisode": "simulator_modifications.observations"},
    "label": {"DistalForce": "guidewire_rl.env.labels.distal_force"},
    "action": {"LinearToDeviceLimits": "guidewire_rl.env.actions"},
    "simulation": {"SofaBeamAdapter": "simulator_modifications.device_contact"},
    "pathfinder": {"BruteForceBFS": "simulator_modifications.pathfinder"},
    "fluoroscopy": {"TrackingOnly": "simulator_modifications.fluoroscopy"},
    "target": {"CenterlineRandom": "simulator_modifications.seeds",
               "Manual": "simulator_modifications.seeds"},
    "vessel_tree": {"AorticArchRandom": "simulator_modifications.seeds",
                    "BiroPhantomArch": "guidewire_rl.env.vessels.biro_phantom"},
    "instrument": {"Simmons2": "guidewire_rl.env.instruments.simmons2",
                   "HookCatheter": "guidewire_rl.env.instruments.hook"},
}

# The two-device task (guidewire_rl/env/two_device/), one line per part.
_TWO_DEVICE = "guidewire_rl.env.two_device."
_BUILTIN_MODULES["monitor"] = {"PhaseDetector": _TWO_DEVICE + "phases",
                               "NominalAdvisor": _TWO_DEVICE + "actions",
                               "WallDistance": _TWO_DEVICE + "wall"}
_BUILTIN_MODULES["action"]["TwoDeviceGuarded"] = _TWO_DEVICE + "actions"
_BUILTIN_MODULES["target"]["RouteWindowTarget"] = _TWO_DEVICE + "targets"
_BUILTIN_MODULES["observation"].update({name: _TWO_DEVICE + "observations" for name in (
    "InstrumentTip2D", "InstrumentShape2D", "TaskTarget2D", "JunctionMap2D", "PhaseOneHot",
    "InsertionDiff", "InsertionLengthsNormalized", "ScriptSuggestion", "FrameHistory")})
_BUILTIN_MODULES["observation_wrapper"].update({name: _TWO_DEVICE + "observations" for name in (
    "TipNoise2D", "Normalize2DTask")})
_BUILTIN_MODULES["reward"] = {name: _TWO_DEVICE + "rewards" for name in (
    "LeadingTipProgress", "CatheterToF", "HookPrepPotential", "PhasePotential", "WindowPenalty",
    "EventPenalty", "ShieldPenalty", "StepPenalty", "OutcomeReward")}
_BUILTIN_MODULES["terminal"] = {"PlatformReached": _TWO_DEVICE + "terminal",
                                "HardFailure": _TWO_DEVICE + "terminal"}
_BUILTIN_MODULES["truncation"] = {"PhaseCutoff": _TWO_DEVICE + "truncation",
                                  "TipForceBlowup": _TWO_DEVICE + "truncation",
                                  "WallBreach": _TWO_DEVICE + "wall"}
_BUILTIN_MODULES["start"] = {"WireFirst": _TWO_DEVICE + "start", "FromSnapshot": _TWO_DEVICE + "snapshot"}
_BUILTIN_MODULES["info"] = {"PhaseInfo": _TWO_DEVICE + "phases", "ShieldInfo": _TWO_DEVICE + "actions",
                            "WallInfo": _TWO_DEVICE + "wall", "ForceBlowupInfo": _TWO_DEVICE + "truncation"}
_BUILTIN_MODULES["label"].update({name: _TWO_DEVICE + "labels" for name in (
    "InstrumentTipForce", "CatheterTipForce", "TipClearance", "SubstepMotion")})
_BUILTIN_MODULES["label"]["WallPenetration"] = _TWO_DEVICE + "wall"


def _module_name(kind: str) -> Union[str, None]:
    if kind in ("device", "instrument"):
        import eve
        return EVE_MODULES["instrument" if hasattr(eve.intervention, "instrument") else "device"]
    return EVE_MODULES[kind]

Spec = Union[str, Mapping[str, Any]]


def register(kind: str, name: str | None = None) -> Callable[[type], type]:
    """Class decorator: make ``cls`` available as ``type: <name>`` for ``kind``."""

    if kind not in _OWN:
        raise KeyError(f"Unknown component kind {kind!r}; expected one of {sorted(_OWN)}")

    def decorator(cls: type) -> type:
        _OWN[kind][name or cls.__name__] = cls
        return cls

    return decorator


def available(kind: str) -> List[str]:
    names = set(_OWN[kind]) | set(_BUILTIN_MODULES.get(kind, {}))
    module_name = _module_name(kind)
    if module_name is not None:
        module = importlib.import_module(module_name)
        names.update(n for n, obj in vars(module).items()
                     if inspect.isclass(obj) and not n.startswith("_"))
    return sorted(names)


def resolve(kind: str, type_name: str) -> type:
    if kind not in _OWN:
        raise KeyError(f"Unknown component kind {kind!r}; expected one of {sorted(_OWN)}")
    if type_name not in _OWN[kind] and type_name in _BUILTIN_MODULES.get(kind, {}):
        importlib.import_module(_BUILTIN_MODULES[kind][type_name])
    if type_name in _OWN[kind]:
        return _OWN[kind][type_name]
    module_name = _module_name(kind)
    if module_name is not None:
        candidate = getattr(importlib.import_module(module_name), type_name, None)
        if inspect.isclass(candidate):
            return candidate
    raise KeyError(f"Unknown {kind} type {type_name!r}. Available: {available(kind)}")


def as_spec(spec: Spec) -> Dict[str, Any]:
    """Accept ``"Name"`` as shorthand for ``{type: Name}``."""

    if isinstance(spec, str):
        return {"type": spec}
    if not isinstance(spec, Mapping) or "type" not in spec:
        raise ValueError(f"Component spec needs a 'type': {spec!r}")
    return dict(spec)


def instantiate(kind: str, spec: Spec, dependencies: Mapping[str, Any],
                *positional: Any) -> Any:
    """Build one component from its spec, injecting scene objects by name."""

    params = as_spec(spec)
    type_name = params.pop("type")
    cls = resolve(kind, type_name)
    signature = inspect.signature(cls.__init__)
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                      for p in signature.parameters.values())
    unknown = [key for key in params
               if key not in signature.parameters and not accepts_any]
    if unknown:
        allowed = [n for n in signature.parameters if n != "self"]
        raise TypeError(f"{kind} {type_name}: unknown parameter(s) {unknown}; "
                        f"constructor takes {allowed}")
    for name in signature.parameters:
        if name in dependencies and name not in params:
            params[name] = dependencies[name]
    return cls(*positional, **params)
