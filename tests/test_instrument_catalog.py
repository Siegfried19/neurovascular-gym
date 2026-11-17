"""The modelled real devices (guidewire_rl/env/instruments/): numbers, the catalog, free space."""

import copy

import numpy as np
import pytest

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config

pytestmark = pytest.mark.upstream


def _second_moment(section) -> float:
    return float(np.pi / 64.0 * (section.diameter_outer ** 4 - section.diameter_inner ** 4))


def _stiffness(section) -> float:
    """Bending stiffness E I, N mm^2."""
    return float(section.young_modulus) * _second_moment(section)


def _turn_deg(section) -> float:
    return float(sum(getattr(e, "angle_in_plane_deg", 0.0) for e in section.mesh_elements))


def _elements(section):
    return [(type(e).__name__, tuple(np.round(list(vars(e).values()), 9))) for e in section.mesh_elements]


def test_every_product_builds_from_its_spec():
    from guidewire_rl.env import registry
    from guidewire_rl.env.instruments import catalog

    keys = [p.key for p in catalog.PRODUCTS]
    assert len(keys) == len(set(keys))
    for p in catalog.PRODUCTS:
        instrument = registry.instantiate("instrument", p.instrument_spec(), {})
        assert instrument.name == p.kind                     # "guidewire" / "catheter"
        assert all(p.label) and all(p.note)
    assert [p.kind for p in catalog.PRODUCTS] == sorted(p.kind for p in catalog.PRODUCTS)[::-1]  # wires first


def test_spec_masses_are_the_class_defaults():
    """The specs repeat the masses (the workbench's parameter page reads them from the config)."""
    from guidewire_rl.env import registry
    from guidewire_rl.env.instruments import catalog

    for p in catalog.PRODUCTS:
        cls = registry.resolve("instrument", p.spec["type"])
        default = cls()
        for key in ("mass_density", "flex_mass_density"):
            if key in p.spec and p.spec["type"] != "HookCatheter":     # the hook's spec sets the training's
                assert p.spec[key] == getattr(default, key)


def test_glidewire_class_is_the_wire_the_training_writes_out():
    from guidewire_rl.env.instruments import GlidewireStiff

    wire = load_config(WORKSPACE_ROOT / "configs" / "two_devices_real_mass.yaml")["env"]["scene"]["instruments"][0]
    built = GlidewireStiff()
    for key, value in wire.items():
        if key == "type":
            continue
        got = getattr(built, key)
        assert (list(got) if isinstance(got, tuple) else got) == pytest.approx(value), key
    assert _stiffness(built.base_section) == pytest.approx(767.0, abs=0.5)      # Qiu 2023: 7.67 N cm^2
    assert _stiffness(built.tip_section) == pytest.approx(10.0, abs=0.05)       # 0.10 N cm^2


def test_lab_hook_product_is_the_training_catheter():
    from guidewire_rl.env.instruments import catalog

    catheter = load_config(WORKSPACE_ROOT / "configs" / "two_devices_real_mass.yaml")["env"]["scene"]["instruments"][1]
    spec = catalog.product("lab_hook").spec
    assert {k: catheter[k] for k in spec} == spec


def test_cordis_sidewinder2_is_the_traced_simmons_with_the_label_size():
    from guidewire_rl.env.instruments import CordisSidewinder2, Simmons2

    cordis, simmons = CordisSidewinder2(), Simmons2()
    for section in (cordis.base_section, cordis.tip_section):
        assert (section.diameter_outer, section.diameter_inner) == (1.65, 1.22)
        assert _stiffness(section) == pytest.approx(486.0, abs=1e-6)            # Qiu 2024's Tempo
    assert _stiffness(simmons.tip_section) == pytest.approx(486.0, abs=0.1)
    assert _elements(cordis.tip_section) == _elements(simmons.tip_section)     # the same traced head
    assert cordis.tip_section.length == pytest.approx(simmons.tip_section.length)


def test_accuvu_omni_flush_numbers():
    from guidewire_rl.env.instruments import AccuVuOmniFlush

    catheter = AccuVuOmniFlush()
    for section in (catheter.base_section, catheter.tip_section):
        assert (section.diameter_outer, section.diameter_inner) == (1.8, 0.97)
        assert section.young_modulus == pytest.approx(1.78e3)                   # the Tempo's nylon
    assert _stiffness(catheter.tip_section) == pytest.approx(840.0, abs=1.0)
    assert _turn_deg(catheter.tip_section) == pytest.approx(180.0)
    assert catheter.tip_section.length == pytest.approx(10 + 7.5 * np.pi + 5, abs=0.01)   # arc meshed in chords
    assert catheter.length == pytest.approx(450.0)


def test_with_products_swaps_and_keeps_the_task_limits():
    from guidewire_rl.env.instruments import catalog

    config = load_config(WORKSPACE_ROOT / "configs" / "two_devices_ica.yaml")
    original = copy.deepcopy(config)
    out = catalog.with_products(config, "glidewire_stiff", "cordis_sidewinder2")
    assert config == original                                                   # not changed in place
    wire, catheter = out["env"]["scene"]["instruments"]
    old_wire, old_catheter = original["env"]["scene"]["instruments"]
    assert wire["type"] == "GlidewireStiff" and catheter["type"] == "CordisSidewinder2"
    assert wire["velocity_limit"] == old_wire["velocity_limit"] and wire["name"] == old_wire["name"]
    assert catheter["velocity_limit"] == old_catheter["velocity_limit"] and "name" not in catheter
    assert catalog.with_products(config) == original
    with pytest.raises(ValueError):
        catalog.with_products(config, catheter="glidewire_stiff")                     # a wire is not a catheter
    single = load_config(WORKSPACE_ROOT / "configs" / "phantom_real_mass.yaml")
    with pytest.raises(ValueError):
        catalog.with_products(single, catheter="accuvu_omni_flush")             # nothing to replace


@pytest.mark.sofa
@pytest.mark.parametrize("catheter", ["CordisSidewinder2", "AccuVuOmniFlush"])
def test_free_space_wire_passes_through_the_catheter(catheter):
    """The catheter out with the 0.035″ Glidewire behind its head, then the wire through and 60 mm out of the
    tip: no simulation error, and the wire's tip ends up 60 mm along the beam past the catheter's."""
    from guidewire_rl.env import registry
    from guidewire_rl.env.instruments import GlidewireStiff
    from guidewire_rl.env.instruments.free_space import FreeSpace

    cath = registry.resolve("instrument", catheter)()
    bench = FreeSpace([GlidewireStiff(), cath])
    head = cath.tip_section.length
    out_at, home = head + 80.0, 60.0
    try:
        bench.reset(seed=0)

        def go(wire_mm, catheter_mm):
            for _ in range(400):
                lengths = bench.inserted_lengths
                top = np.array([15.0, 25.0])
                v = np.clip((np.array([wire_mm, catheter_mm]) - lengths) / bench.step_seconds, -top, top)
                if np.all(np.abs(v) < 1e-3):
                    return
                bench.step(np.c_[v, np.zeros(2)])
                assert not bench.simulation_error and np.all(np.isfinite(bench.dof_positions))

        go(home, out_at)
        go(out_at + 60.0, out_at)
        assert bench.inserted_lengths == pytest.approx([out_at + 60.0, out_at], abs=0.5)
        nodes = bench.dof_positions                                 # tip first
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
        assert s[-1] >= out_at + 59.0                               # the beam reaches the wire's tip
    finally:
        bench.close()
