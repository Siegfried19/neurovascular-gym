"""The contact-solve blow-up gate (two_device/truncation.py, ``TipForceBlowup`` + ``ForceBlowupInfo``;
work log 2026-11-09) and the phantom's wall-mesh override (vessels/biro_phantom.py ``mesh=``).

Without SOFA: with fake force labels, the gate stays quiet below the limit, trips in the step a substep
sample exceeds it, marks the simulation as errored (so ``SimError`` and the trainers drop the step like
NaN), reports the peak in the info term, and clears at the next step / reset; the phantom built with
another wall file loads that file and keeps the version's centerlines. With SOFA (``sofa``): the real-mass
env builds with the gate in its config and a dozen gentle steps never trip it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from two_device_scenes import random_steps, real_mass_env

pytestmark = pytest.mark.upstream


def fake_label(name, instrument, samples, peak=None):
    """A tip-force label whose monitor saw ``samples`` in the 3 mm window and ``peak`` as the largest
    single contact force (None: a monitor without that attribute, the samples' maximum is used)."""
    monitor = SimpleNamespace(samples={instrument: list(samples)})
    if peak is not None:
        monitor.peak_contact_N = peak
    return SimpleNamespace(name=name, instrument=instrument, monitor=monitor)


def test_force_gate_trips_marks_a_simulation_error_and_reports_the_peak():
    from guidewire_rl.env.two_device.truncation import ForceBlowupInfo, TipForceBlowup
    sim = SimpleNamespace(simulation_error=False)
    intervention = SimpleNamespace(simulation=sim)
    wire = fake_label("distal_force", 0, [0.1, 0.4, 0.3], peak=2.0)
    cath = fake_label("catheter_tip_force", 1, [1.0, 2.9, 1.5], peak=12.0)
    gate = TipForceBlowup(intervention, labels=[wire, cath, fake_label("other", 2, [900.0], peak=900.0)])
    told = ForceBlowupInfo(intervention)
    gate.reset()
    assert not gate.truncated and not sim.simulation_error and told.info == {"force_blowup_N": 0.0}
    cath.monitor.peak_contact_N = 38.0                    # one contact of this step blew up
    assert gate.truncated and sim.simulation_error
    assert told.info == {"force_blowup_N": 38.0}
    assert gate.truncated                                 # stays tripped within the step
    gate.step()                                           # the next step starts clean
    cath.monitor.peak_contact_N = 0.2
    assert gate.last_blowup_N == 0.0 and not gate.truncated and told.info == {"force_blowup_N": 0.0}
    with pytest.raises(KeyError):
        TipForceBlowup(intervention, labels=[fake_label("other", 2, [])])


def test_force_gate_limit_is_a_parameter_and_falls_back_to_the_window_samples():
    from guidewire_rl.env.two_device.truncation import TipForceBlowup
    sim = SimpleNamespace(simulation_error=False)
    gate = TipForceBlowup(SimpleNamespace(simulation=sim), labels=[fake_label("distal_force", 0, [12.0])], limit_N=10.0)
    assert gate.truncated and gate.last_blowup_N == 12.0 and sim.force_blowup_N == 12.0


def test_phantom_takes_another_wall_mesh_and_keeps_its_centerlines(tmp_path):
    from guidewire_rl.env.vessels.biro_phantom import VERSIONS, BiroPhantomArch
    source, spec = VERSIONS["rebuilt_cut"]
    lines = source.read_text().splitlines()
    copy = tmp_path / "wall_copy.obj"
    copy.write_text("\n".join(lines[: len(lines) // 2 + 1] if lines[len(lines) // 2].startswith("v ") else lines) + "\n")
    # a different file: fewer faces than the version's own wall
    vertices = [l for l in copy.read_text().splitlines() if l.startswith("v ")]
    assert vertices
    tree = BiroPhantomArch(version="rebuilt_cut", mesh=str(copy))
    assert tree.mesh_override == str(copy) and tree.mesh == str(copy)
    own = BiroPhantomArch(version="rebuilt_cut")
    assert own.mesh_override is None
    np.testing.assert_array_equal(tree.insertion.position, own.insertion.position)
    assert [b.name for b in tree.branches] == [b.name for b in own.branches]
    with pytest.raises(FileNotFoundError):
        BiroPhantomArch(version="rebuilt_cut", mesh=str(tmp_path / "missing.obj"))


@pytest.mark.sofa
def test_real_mass_env_has_the_gate_and_gentle_steps_do_not_trip_it():
    from guidewire_rl.env.two_device.truncation import TipForceBlowup
    with real_mass_env() as env:
        parts = getattr(env._env.truncation, "truncations", None) or getattr(env._env.truncation, "parts", None)
        assert parts is not None and any(isinstance(t, TipForceBlowup) for t in parts)
        env.reset(seed=9)
        for info in random_steps(env, seed=9):
            assert info["force_blowup_N"] == 0.0 and not info["simulation_error"]
