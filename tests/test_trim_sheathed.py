"""``trim_sheathed`` (simulator_modifications/device_contact.py; BeamAdapter patch
collision_start_abscissa.patch; docs/simulator_changes.md problem 26): the wire's collision chain
starts at the catheter's tip instead of the insertion point.

With SOFA (``sofa``): on a far-away mesh, wire 100 mm in / catheter 60 mm, the wire chain's first point
is at or just before the catheter tip's abscissa (its own: length - 100 + 60 = 410 mm; the shaft's
points are 10 mm apart, so at 400-410) and its last at the tip (450); untrimmed it starts at 350. With
only the soft tip out, the chain is just that; fully inside the catheter it is empty and the simulation
stays finite; without the trim nothing changes. On the flat wall the devices end a control step where
they end without the trim to a few micrometres (the dropped rows never carried force, but the contact
solve iterates to a 1e-4 tolerance and its sweep order changes), and each device's wall force within 0.01 N. The switch refuses a scene without per-device collision or the substep loop.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

import two_device_scenes as T

pytestmark = [pytest.mark.upstream, pytest.mark.sofa]


def wire_chain(sim):
    from simulator_modifications.device_contact import collision_node_name
    node = sim._instruments_combined.getChild(collision_node_name("guidewire"))
    return (np.asarray(node.collisMap.segmentsCurvAbs.value, dtype=float),
            np.asarray(node.CollisionDOFs.position.value, dtype=float)[:, :3],
            len(node.collisEdgeSet.edges.value))


def test_wire_chain_starts_at_the_catheter_tip():
    with T.scene_in_tmp(T.far_away_mesh, stop_on_nan=True, trim_sheathed=True) as sim:
        length = sim.instruments[0].length                      # 450
        T.move_to(sim, [100.0, 60.0], settle=20)
        sim.step(np.zeros((2, 2)), 1 / 7.5)
        abscissae, points, n_edges = wire_chain(sim)
        start = length - 100.0 + 60.0
        assert len(points) == len(abscissae) == n_edges + 1
        assert start - 10.0 - 1e-6 <= abscissae[0] <= start + 1e-6     # at or one shaft spacing before
        assert abscissae[-1] == pytest.approx(length)
        assert np.all(np.diff(abscissae) > 0)
        T.move_to(sim, [100.0, 95.0], settle=20)                 # only the soft tip's last 5 mm out
        sim.step(np.zeros((2, 2)), 1 / 7.5)
        abscissae, points, _ = wire_chain(sim)
        assert length - 5.0 - 0.5 - 1e-6 <= abscissae[0] <= length - 5.0 + 1e-6 and len(points) <= 12
        T.move_to(sim, [50.0, 60.0], settle=20)                  # fully inside
        sim.step(np.zeros((2, 2)), 1 / 7.5)
        abscissae, points, n_edges = wire_chain(sim)
        assert len(points) == 0 and n_edges == 0
        assert np.isfinite(sim._instruments_combined.DOFs.position.value).all()
        T.move_to(sim, [100.0, 0.0], settle=20)                  # no catheter: the whole inserted wire
        sim.step(np.zeros((2, 2)), 1 / 7.5)
        abscissae, _, _ = wire_chain(sim)
        assert abscissae[0] == pytest.approx(length - 100.0) and abscissae[-1] == pytest.approx(length)


def test_without_the_trim_the_chain_covers_the_inserted_wire():
    with T.scene_in_tmp(T.far_away_mesh, stop_on_nan=True) as sim:
        T.move_to(sim, [100.0, 60.0], settle=20)
        sim.step(np.zeros((2, 2)), 1 / 7.5)
        abscissae, _, _ = wire_chain(sim)
        assert abscissae[0] == pytest.approx(350.0) and abscissae[-1] == pytest.approx(450.0)


def test_trim_changes_nothing_on_the_wall():
    """Both devices pushed along the flat wall for a control step, the catheter leading (wire 80 mm
    in, catheter 100: the wire fully sheathed, its trimmed chain empty) and the wire leading out of a
    hook turned towards the wall (100 / 30): tips and each device's wall force as without the trim
    (the dropped wire points sat inside the catheter, never touching)."""
    from guidewire_rl.env.two_device.labels import CatheterTipForce, InstrumentTipForce
    from types import SimpleNamespace

    def run(trim, targets, catheter_rotation=0.0):
        with T.scene_in_tmp(T.flat_wall_mesh, stop_on_nan=True, trim_sheathed=trim) as sim:
            sim._instruments_combined.IRcontroller.rotationInstrument.value = [0.0, catheter_rotation]
            # whole-device windows: the hook's tip itself may point away from the wall
            wire_label = InstrumentTipForce(instrument=0, distal_length_mm=1000.0)
            cath_label = CatheterTipForce(distal_length_mm=1000.0)
            env = SimpleNamespace(simulation=sim)
            wire_label.attach(env)
            cath_label.attach(env)
            T.move_to(sim, targets, settle=100)                  # both past the wall at 40 mm
            for label in (wire_label, cath_label):
                label.begin_step()
            sim.step(np.array([[10.0, 0.0], [10.0, 0.0]]), 1 / 7.5)
            forces = (wire_label.finish_step()[0], cath_label.finish_step()[0])
            tips = np.asarray(sim.dof_positions, dtype=float)[:2].copy()   # leading tip and next node
            return forces, tips, len(wire_chain(sim)[1])

    forces_off, tips_off, n_off = run(False, [80.0, 100.0])
    forces_on, tips_on, n_on = run(True, [80.0, 100.0])
    assert forces_off[1] > 0.05 and n_on == 0 < n_off               # the catheter presses; the wire chain is gone
    np.testing.assert_allclose(tips_on, tips_off, atol=0.01)         # mm; the LCP's 1e-4 tolerance, not physics
    np.testing.assert_allclose(forces_on, forces_off, atol=0.01)

    # the wire leading out of a hook turned towards the wall (catheter 30 of 40 mm, rotated 180 deg;
    # with the hook as laid the wire leaves it pointing away from the wall and touches nothing): the
    # exposed wire presses with ~1.2 N; the chain lost only the points inside the catheter. The contact
    # solve iterates to 1e-4 with a different sweep, so tips and peak force agree to tens of um / 5 %,
    # not bit for bit. (The hook's own corner driven into the rigid plane, e.g. catheter 60, is a
    # runaway contact whose peak differs chaotically between any two runs, so it is not compared.)
    forces_off, tips_off, n_off = run(False, [100.0, 30.0], np.pi)
    forces_on, tips_on, n_on = run(True, [100.0, 30.0], np.pi)
    assert forces_off[0] > 0.5 and forces_off[1] == 0.0 and 0 < n_on < n_off
    np.testing.assert_allclose(tips_on, tips_off, atol=0.05)
    assert abs(forces_on[0] - forces_off[0]) < 0.05 * forces_off[0] and forces_on[1] == 0.0


def test_trim_needs_per_device_collision_and_the_substep_loop():
    with pytest.raises(ValueError):
        T.scene(T.far_away_mesh(tempfile.gettempdir()), stop_on_nan=False, trim_sheathed=True)
    with pytest.raises(ValueError):
        T.scene(T.far_away_mesh(tempfile.gettempdir()), instrument_collision=False, trim_sheathed=True)
