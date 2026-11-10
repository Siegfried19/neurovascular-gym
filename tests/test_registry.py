import eve
import pytest

from guidewire_rl.env import registry


def test_resolves_eve_and_own_components():
    assert registry.resolve("observation", "Target2D") is eve.observation.Target2D
    assert registry.resolve("reward", "PathLengthDelta") is eve.reward.PathLengthDelta
    own = registry.resolve("observation", "VesselGeometry")
    assert own.__module__ == "guidewire_rl.env.observations.vessel_geometry"
    assert "VesselGeometry" in registry.available("observation")
    assert "DistalForce" in registry.available("label")


def test_unknown_type_lists_what_exists():
    with pytest.raises(KeyError, match="Tracking2D"):
        registry.resolve("observation", "Tracking2d")


def test_scene_objects_are_injected_and_typos_rejected():
    pathfinder = object()
    reward = registry.instantiate("reward", {"type": "PathLengthDelta", "factor": 0.5},
                                  {"pathfinder": pathfinder, "intervention": object()})
    assert reward.pathfinder is pathfinder and reward.factor == 0.5
    with pytest.raises(TypeError, match="factr"):
        registry.instantiate("reward", {"type": "PathLengthDelta", "factr": 0.5},
                             {"pathfinder": pathfinder})


def test_string_shorthand():
    assert registry.as_spec("MaxSteps") == {"type": "MaxSteps"}
    with pytest.raises(ValueError):
        registry.as_spec({"max_steps": 3})
