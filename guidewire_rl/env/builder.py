"""Assemble a ``GuidewireEnv`` from the ``env:`` section of a config.

Scene objects are built first (vessel tree → instruments → simulation →
fluoroscopy → target → intervention), then every MDP part. Each entry is a
``{type: Name, ...params}`` spec resolved by ``registry``; scene objects are
passed to constructors that ask for them by name.

The wires/catheters are listed under ``instruments:`` (upstream stEVE's name)
or ``devices:`` (Wilson's eve, G13); the list is offered to constructors under
both names.

Shared parts (added for the two-device task; a config without them builds as before):

* ``monitors:`` (list of ``{name: ..., type: ...}``, kind ``monitor``): per-step
  state several parts read (e.g. ``PhaseDetector``). Built once after the scene,
  offered to later constructors under their ``name``, stepped right after the
  simulator and reset right after the scene (they ride on stEVE's pathfinder
  slot, which ``eve.Env`` steps before observations, rewards and terminals).
* The labels (as ``labels``) and the action map (as ``action_map``) are built
  before the MDP parts, so those can read them (a safety shell reading last
  step's force label; a reward reading the shell).
* Observation terms are offered, as built so far, to later terms
  (``observation_terms``: name -> term), so a term can read earlier ones.
* ``scene.fluoroscopy.image_rot_zx: per_target``: the C-arm angle is looked up per episode from
  the phantom's angle table for the branch of the episode's target (``c_arm/per_target.py``). The
  fluoroscopy is built at the table's all-5 angle, the per-target part (``c_arm``) after the
  target, and ``eve.Env`` gets the intervention wrapped so that part runs right after every
  intervention reset (target drawn), before everything else resets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import eve
from eve.pathfinder import PathfinderDummy

from . import registry
from .c_arm import per_target
from .guidewire_env import GuidewireEnv

INSTRUMENT_KEYS = ("instruments", "devices")

SCENE_ORDER = (
    ("vessel_tree", "vessel_tree"),
    ("instruments", "instrument"),
    ("simulation", "simulation"),
    ("fluoroscopy", "fluoroscopy"),
    ("target", "target"),
    ("intervention", "intervention"),
)


def _build_scene(scene: Mapping[str, Any]) -> Dict[str, Any]:
    deps: Dict[str, Any] = {}
    c_arm_table = None
    for key, kind in SCENE_ORDER:
        if key == "instruments":
            given = [k for k in INSTRUMENT_KEYS if scene.get(k) is not None]
            if len(given) != 1:
                raise KeyError(f"env.scene needs exactly one of {list(INSTRUMENT_KEYS)}, got {given}")
            instruments = [registry.instantiate(kind, spec, deps) for spec in scene[given[0]]]
            for name in INSTRUMENT_KEYS:
                deps[name] = instruments
            continue
        if key not in scene:
            raise KeyError(f"env.scene.{key} is missing")
        spec = scene[key]
        if key == "fluoroscopy" and per_target.is_per_target(spec):
            c_arm_table = per_target.table_for(deps["vessel_tree"])
            spec = {**registry.as_spec(spec), "image_rot_zx": per_target.initial_angle(c_arm_table)}
        deps[key] = registry.instantiate(kind, spec, deps)
    if c_arm_table is not None:
        deps["c_arm"] = per_target.PerTargetCArm(c_arm_table, deps["vessel_tree"],
                                                 deps["fluoroscopy"], deps["target"])
    return deps


def _build_observation_term(term: Mapping[str, Any], deps: Mapping[str, Any]):
    spec = dict(term)
    spec.pop("name")
    wrappers = spec.pop("wrappers", []) or []
    observation = registry.instantiate("observation", spec, deps)
    for wrapper in wrappers:
        observation = registry.instantiate("observation_wrapper", wrapper, deps, observation)
    return observation


def _combine(kind: str, specs: Sequence[Any], deps: Mapping[str, Any]):
    """One component as-is, several via stEVE's ``Combination`` of that kind."""
    if isinstance(specs, (str, Mapping)):
        specs = [specs]
    parts = [registry.instantiate(kind, spec, deps) for spec in specs]
    if not parts:
        raise ValueError(f"env.{kind} needs at least one entry")
    if len(parts) == 1:
        return parts[0]
    return registry.resolve(kind, "Combination")(parts)


def max_steps_of(truncation_specs: Any) -> Optional[int]:
    if isinstance(truncation_specs, (str, Mapping)):
        truncation_specs = [truncation_specs]
    for spec in truncation_specs or []:
        spec = registry.as_spec(spec)
        if spec["type"] == "MaxSteps":
            return int(spec["max_steps"])
    return None


def observation_term_names(config: Mapping[str, Any]) -> List[str]:
    names = [term["name"] for term in config["observation"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate observation term names: {names}")
    return names


class _PathfinderWithMonitors:
    """stEVE's pathfinder slot with the monitors after it: ``eve.Env`` steps (resets) this right
    after the simulator (the scene) and before observations, rewards and terminals."""

    def __init__(self, pathfinder, monitors: Mapping[str, Any]) -> None:
        self.pathfinder = pathfinder if pathfinder is not None else PathfinderDummy()
        self.monitors = list(monitors.values())

    def step(self) -> None:
        self.pathfinder.step()
        for monitor in self.monitors:
            monitor.step()

    def reset(self, episode_nr: int = 0) -> None:
        self.pathfinder.reset(episode_nr)
        for monitor in self.monitors:
            monitor.reset(episode_nr)

    def __getattr__(self, name: str):
        return getattr(self.__dict__["pathfinder"], name)


class _InterventionWithCArm:
    """The intervention with the per-target C-arm right after its reset (``c_arm/per_target.py``):
    ``eve.Env`` resets the intervention (scene, target) first, then start, pathfinder, observations,
    rewards, so all of them see the episode's angle. Everything else is the intervention's."""

    def __init__(self, intervention, c_arm) -> None:
        self.intervention = intervention
        self.c_arm = c_arm

    def reset(self, episode_number: int = 0, seed=None, options=None) -> None:
        self.intervention.reset(episode_number, seed, options)
        self.c_arm.reset()

    def __getattr__(self, name: str):
        try:
            intervention = self.__dict__["intervention"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(intervention, name)


def _build_monitors(specs: Optional[Sequence[Mapping[str, Any]]], deps: Dict[str, Any]) -> Dict[str, Any]:
    monitors: Dict[str, Any] = {}
    for spec in specs or []:
        spec = dict(spec)
        name = spec.pop("name", None)
        if not name:
            raise KeyError(f"env.monitors entry needs a name: {spec!r}")
        if name in deps or name in monitors:
            raise KeyError(f"env.monitors: name {name!r} is already taken")
        monitors[name] = deps[name] = registry.instantiate("monitor", spec, deps)
    return monitors


def build_env(config: Mapping[str, Any], *, render: bool = False) -> GuidewireEnv:
    deps = _build_scene(config["scene"])
    if config.get("pathfinder"):
        deps["pathfinder"] = registry.instantiate("pathfinder", config["pathfinder"], deps)
    monitors = _build_monitors(config.get("monitors"), deps)
    labels = [registry.instantiate("label", spec, deps) for spec in config.get("labels", [])]
    deps["labels"] = labels
    action_map = registry.instantiate("action", config.get("action", "LinearToDeviceLimits"), deps)
    deps["action_map"] = action_map

    names = observation_term_names(config)
    terms: Dict[str, Any] = {}
    deps["observation_terms"] = terms          # filled as it goes: a term sees the earlier ones
    for term in config["observation"]:
        terms[term["name"]] = _build_observation_term(term, deps)
    observation = eve.observation.ObsDict(terms)
    reward = _combine("reward", config["reward"], deps)
    terminal = _combine("terminal", config["terminal"], deps)
    truncation = _combine("truncation", config["truncation"], deps)
    info = _combine("info", config["info"], deps) if config.get("info") else None
    start = registry.instantiate("start", config.get("start", "InsertionPoint"), deps)

    visualisation = None
    if render:
        # SofaPygame enables SOFA visual nodes in its constructor, so it must
        # exist before the first reset.
        visualisation = registry.instantiate(
            "visualisation", config.get("visualisation", "SofaPygame"), deps)

    pathfinder = deps.get("pathfinder")
    if monitors:
        pathfinder = _PathfinderWithMonitors(pathfinder, monitors)
    intervention = deps["intervention"]
    if "c_arm" in deps:
        intervention = _InterventionWithCArm(intervention, deps["c_arm"])
    eve_env = eve.Env(
        intervention=intervention, observation=observation, reward=reward,
        terminal=terminal, truncation=truncation, info=info, start=start,
        pathfinder=pathfinder, visualisation=visualisation)
    return GuidewireEnv(
        eve_env=eve_env, simulation=deps["simulation"], observation_terms=names,
        labels=labels, action_map=action_map,
        observation_clip=config.get("observation_clip"),
        max_episode_steps=max_steps_of(config["truncation"]),
        render_mode="human" if render else None, monitors=monitors)
