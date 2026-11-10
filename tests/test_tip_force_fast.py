"""The vectorized tip-force evaluation (two_device/labels.py, ``window_force_fast``) gives the same
number as the reference walk (``located_contacts`` + ``window_force``) bit for bit, and raises the
same errors. Without SOFA: random sparse Jacobians with point and line contacts, both row modes,
compared exactly over many cases; the error cases. With SOFA (``sofa``): the real-mass two-device env
for a dozen random steps with the monitor's evaluation swapped for one that runs both and compares
every substep; and the fast path is what the monitor uses.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from two_device_scenes import random_steps, real_mass_env

pytestmark = pytest.mark.upstream


def chain(n, rng):
    """A wiggly chain of n collision nodes along +x with ~1 mm spacing, and its edges."""
    coords = np.cumsum(np.column_stack([rng.uniform(0.6, 1.4, n), rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)]), axis=0)
    return coords, np.column_stack([np.arange(n - 1), np.arange(1, n)])


def jacobian(n_rows, n_nodes, contacts, rng):
    """Rows for the given contacts: (first row, node or (node, node+1), own) -> a normal row and two
    friction rows with random directions; a line contact's vectors are the normal split (1-a, a)."""
    rows, cols, vals = [], [], []
    for first, nodes, own in contacts:
        if not own:
            continue
        for k in range(3):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            if isinstance(nodes, tuple):
                a = rng.uniform(0.05, 0.95)
                parts = [(nodes[0], (1 - a) * direction), (nodes[1], a * direction)]
            else:
                parts = [(nodes, direction)]
            for node, vector in parts:
                for c in range(3):
                    rows.append(first + k)
                    cols.append(3 * node + c)
                    vals.append(vector[c])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, 3 * n_nodes))


def random_case(rng, own_rows_only):
    n_nodes = int(rng.integers(5, 60))
    coords, edges = chain(n_nodes, rng)
    n_contacts = int(rng.integers(1, 25))
    contacts = []
    for i in range(n_contacts):
        own = True if not own_rows_only else bool(rng.random() < 0.6)
        if rng.random() < 0.5:
            nodes = int(rng.integers(0, n_nodes))
        else:
            lo = int(rng.integers(0, n_nodes - 1))
            nodes = (lo, lo + 1)
        contacts.append((3 * i, nodes, own))
    n_rows = 3 * n_contacts + (3 if rng.random() < 0.3 else 0)       # sometimes rows beyond the matrix
    matrix = jacobian(3 * n_contacts, n_nodes, contacts, rng)
    lam = np.where(rng.random(n_rows) < 0.3, 0.0, rng.uniform(0, 2e-3, n_rows))
    if own_rows_only:
        lam[3 * n_contacts:] = 0.0
    else:
        lam = lam[:3 * n_contacts]
    for first, _, own in contacts:
        if not own and not own_rows_only:
            lam[first] = 0.0
    return lam, matrix, coords, edges


def reference(lam, matrix, coords, edges, dt, start, length, own):
    from guidewire_rl.env.two_device.labels import jacobian_rows, located_contacts, window_force
    return window_force(located_contacts(lam, jacobian_rows(matrix), coords, edges, dt, own_rows_only=own),
                        start, length)


@pytest.mark.parametrize("own_rows_only", [True, False])
def test_fast_window_force_equals_the_reference_bit_for_bit(own_rows_only):
    from guidewire_rl.env.two_device.labels import window_force_fast
    rng = np.random.default_rng(7 if own_rows_only else 8)
    nonzero = 0
    for _ in range(300):
        lam, matrix, coords, edges = random_case(rng, own_rows_only)
        dt = 0.002
        for start, length in ((0.0, 3.0), (0.0, 1000.0), (float(rng.uniform(0, 20)), 4.0)):
            fast = window_force_fast(lam, matrix, coords, edges, dt, start, length, own_rows_only)
            slow = reference(lam, matrix, coords, edges, dt, start, length, own_rows_only)
            assert fast == slow, (fast, slow)
            nonzero += slow > 0
    assert nonzero > 200


def test_fast_window_force_raises_the_reference_errors():
    from guidewire_rl.env.two_device.labels import window_force_fast
    rng = np.random.default_rng(3)
    coords, edges = chain(10, rng)
    good = jacobian(3, 10, [(0, 4, True)], rng)
    assert window_force_fast(np.array([1e-3, 0, 0]), good, coords, edges, 0.002, 0.0, 1000.0) > 0
    assert window_force_fast(np.zeros(3), good, coords, edges, 0.002, 0.0, 1000.0) == 0.0
    assert window_force_fast(np.zeros(0), good, coords, edges, 0.002, 0.0, 1000.0) == 0.0
    with pytest.raises(ValueError, match="Negative unilateral"):
        window_force_fast(np.array([-1e-3, 0, 0]), good, coords, edges, 0.002, 0.0, 1000.0)
    with pytest.raises(ValueError, match="finite frictional row triplets"):
        window_force_fast(np.array([1e-3, 0]), good, coords, edges, 0.002, 0.0, 1000.0)
    with pytest.raises(ValueError, match="no collision Jacobian row"):      # active row 3 not in the matrix
        window_force_fast(np.array([1e-3, 0, 0, 1e-3, 0, 0]), good, coords, edges, 0.002, 0.0, 1000.0)
    three = sp.csr_matrix((np.ones(9), (np.zeros(9, int), np.arange(9))), shape=(3, 30))
    with pytest.raises(ValueError, match="one or two collision nodes"):
        window_force_fast(np.array([1e-3, 0, 0]), three, coords, edges, 0.002, 0.0, 1000.0)
    apart = sp.csr_matrix((np.ones(6) * 0.5, (np.zeros(6, int), np.r_[np.arange(3), 12 + np.arange(3)])), shape=(3, 30))
    with pytest.raises(ValueError, match="not a current collision edge"):
        window_force_fast(np.array([1e-3, 0, 0]), apart, coords, edges, 0.002, 0.0, 1000.0)
    # not the fast path's case: a text Jacobian (SOFA 23.06) or a non-chain topology -> None
    assert window_force_fast(np.array([1e-3, 0, 0]), "text", coords, edges, 0.002, 0.0, 1000.0) is None
    assert window_force_fast(np.array([1e-3, 0, 0]), good, coords, edges[::-1][:-1], 0.002, 0.0, 1000.0) is None


@pytest.mark.sofa
def test_monitor_uses_the_fast_path_and_it_matches_the_reference_on_real_substeps(monkeypatch):
    from guidewire_rl.env.two_device import labels as L
    assert L.TipForceMonitor._window_force is L.window_force_fast
    seen = {"calls": 0, "nonzero": 0}

    def checked(multipliers, matrix, positions, edges, dt, start, length, own):
        fast = L.window_force_fast(multipliers, matrix, positions, edges, dt, start, length, own)
        slow = reference(np.asarray(multipliers), matrix, positions, edges, dt, start, length, own)
        assert fast == slow, (fast, slow)
        seen["calls"] += 1
        seen["nonzero"] += slow > 0
        return fast

    monkeypatch.setattr(L.TipForceMonitor, "_window_force", staticmethod(checked))   # put back afterwards
    with real_mass_env() as env:
        env.reset(seed=5)
        for info in random_steps(env, seed=5, high=0.8):
            assert np.isfinite(info["labels"]["distal_force"]) and np.isfinite(info["labels"]["catheter_tip_force"])
    assert seen["calls"] >= 2 * 33 and seen["nonzero"] > 0        # two labels x the substeps of one step
