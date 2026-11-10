"""Shared pieces of the two-device real-mass tests (test_device_contact, test_trim_sheathed,
test_wall_monitor, test_force_blowup, test_tip_force_fast, test_snapshot): the config, small wall
meshes, a SOFA scene built from the config's devices, driving the devices, reading the collision
chains, the random-push recipes, the config's env, and a runner for code that must start in a fresh
interpreter. Not a test module (pytest collects ``test_*.py`` only); the test modules import it by
name, which works because pytest puts ``tests/`` on ``sys.path`` for them."""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

from guidewire_rl import WORKSPACE_ROOT
from guidewire_rl.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = WORKSPACE_ROOT / "configs" / "two_devices_real_mass.yaml"
WALL_AT = 40.0             # mm along the insertion axis where the flat wall crosses it
WALL_ANGLE = math.radians(15.0)
WALL_NORMAL = np.array([-math.sin(WALL_ANGLE), math.cos(WALL_ANGLE), 0.0])


# ------------------------------------------------------------------------- meshes ----

def far_away_mesh(directory) -> str:
    """A vessel 'tree' of one triangle far from the devices: no contact at all."""
    path = os.path.join(directory, "far.obj")
    with open(path, "w") as fh:
        fh.write("v -5000 -5000 -5000\nv -5001 -5000 -5000\nv -5000 -5001 -5000\nf 1 2 3\n")
    return path


def flat_wall_mesh(directory) -> str:
    """A 600 x 600 mm plane through (WALL_AT, 0, 0), 15 degrees off the insertion axis (+x), facing
    the devices: they meet it at a glancing angle and are pressed along it."""
    along = np.array([math.cos(WALL_ANGLE), math.sin(WALL_ANGLE), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    p0 = np.array([WALL_AT, 0.0, 0.0])
    a, b, c, d = (p0 - 300 * along - 300 * up, p0 + 300 * along - 300 * up,
                  p0 + 300 * along + 300 * up, p0 - 300 * along + 300 * up)
    path = os.path.join(directory, "wall.obj")
    with open(path, "w") as fh:
        for v in (a, b, c, d):
            fh.write("v %.6f %.6f %.6f\n" % tuple(v))
        fh.write("f 1 3 2\nf 1 4 3\n")            # normals towards +WALL_NORMAL, the devices' side
    return path


def grid_wall_mesh(directory, cells=20) -> str:
    """The same plane as ``flat_wall_mesh``, cut into cells x cells squares of two triangles: many
    mesh edges across the devices' way (a device's segments touch the vessel's edges, not its faces)."""
    along = np.array([math.cos(WALL_ANGLE), math.sin(WALL_ANGLE), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    p0 = np.array([WALL_AT, 0.0, 0.0])
    ticks = np.linspace(-300.0, 300.0, cells + 1)
    index = lambda i, j: 1 + j * (cells + 1) + i
    path = os.path.join(directory, "grid_wall.obj")
    with open(path, "w") as fh:
        for v in ticks:
            for u in ticks:
                fh.write("v %.6f %.6f %.6f\n" % tuple(p0 + u * along + v * up))
        for j in range(cells):
            for i in range(cells):
                a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
                fh.write(f"f {a} {c} {b}\nf {a} {d} {c}\n")    # normals towards +WALL_NORMAL
    return path


def wall_distance(points: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float)[:, :3] - np.array([WALL_AT, 0.0, 0.0])) @ WALL_NORMAL


# ------------------------------------------------------------------------- scenes ----

def scene(mesh_path: str, **simulation):
    """The config's two devices and simulation settings (plus ``simulation`` overrides), inserted
    along +x from the origin; device rotations 0."""
    import simulator_modifications.device_contact  # noqa: F401  (registers the class)
    from guidewire_rl.env import registry
    spec = load_config(CONFIG)["env"]["scene"]
    instruments = [registry.resolve("instrument", s["type"])(**{k: v for k, v in s.items() if k != "type"})
                   for s in spec["instruments"]]
    settings = {k: v for k, v in spec["simulation"].items() if k != "type"}
    settings.update(friction=0.0, **simulation)
    # the sheathed-wire trim is tested on its own (test_trim_sheathed.py); the scenes here are about
    # wall contact and some switch off its prerequisites, so the config's switch must not leak in
    settings["trim_sheathed"] = simulation.get("trim_sheathed", False)
    sim = registry.resolve("simulation", "SofaBeamAdapter")(**settings)
    sim.reset(insertion_point=np.zeros(3), insertion_direction=np.array([1.0, 0.0, 0.0]),
              mesh_path=mesh_path, instruments=instruments, coords_high=np.full(3, 500.0),
              coords_low=np.full(3, -500.0), seed=0)
    sim._instruments_combined.IRcontroller.rotationInstrument.value = [0.0, 0.0]
    return sim


@contextlib.contextmanager
def scene_in_tmp(mesh, **simulation):
    """``scene`` on the mesh ``mesh(directory)`` writes into a temporary directory; closed afterwards."""
    with tempfile.TemporaryDirectory() as tmp:
        sim = scene(mesh(tmp), **simulation)
        try:
            yield sim
        finally:
            sim.close()


def move_to(sim, targets, step=0.1, settle=0):
    """Insert/retract both devices to ``targets`` (mm) by at most ``step`` per substep, then
    ``settle`` substeps; every substep must stay finite."""
    import Sofa
    ctrl = sim._instruments_combined.IRcontroller
    dofs = sim._instruments_combined.DOFs
    dt = sim.root.dt.value
    x = np.asarray(ctrl.xtip.value, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n = 0
    while np.any(x != targets) or n < settle:
        if np.any(x != targets):
            x = x + np.clip(targets - x, -step, step)
            ctrl.xtip.value = x.tolist()
        else:
            n += 1
        Sofa.Simulation.animate(sim.root, dt)
        assert np.isfinite(dofs.position.value).all()


def chain(sim, name: str) -> np.ndarray:
    from simulator_modifications.device_contact import collision_node_name
    node = sim._instruments_combined.getChild(collision_node_name(name))
    edges = np.asarray(node.collisEdgeSet.findData("edges").value, dtype=int)
    points = np.asarray(node.CollisionDOFs.position.value, dtype=float)
    np.testing.assert_array_equal(edges, np.column_stack([np.arange(len(points) - 1),
                                                          np.arange(1, len(points))]))
    return points


def beam_lengths(sim) -> np.ndarray:
    combined = sim._instruments_combined
    return np.concatenate([np.asarray(combined.getObject("Interpol_" + i.name).lengthList.value,
                                      dtype=float).reshape(-1) for i in sim._instruments])


# ---------------------------------------------------------------- the config's env ----

@contextlib.contextmanager
def real_mass_env(**env_overrides):
    """The config's env (``build_env``), its top-level ``env`` keys replaced by ``env_overrides``;
    closed afterwards."""
    from guidewire_rl.env import build_env
    spec = load_config(CONFIG)["env"]
    spec.update(env_overrides)
    env = build_env(spec)
    try:
        yield env
    finally:
        env.close()


def random_steps(env, seed, n=12, low=-0.3, high=0.6):
    """Up to ``n`` actions drawn uniformly in [low, high] on an env already reset (``seed`` seeds the
    draw); yields each step's info and stops when the episode ends. The gentle random steps of the
    env-level checks."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        _, _, terminated, truncated, info = env.step(rng.uniform(low, high, size=env.action_space.shape).astype(np.float32))
        yield info
        if terminated or truncated:
            break


def random_pushes(env, n_steps, seed):
    """The stress actions of the 2026-11-03 measurements: a new uniform action every 5 steps,
    pushes biased towards 100-250 mm. Returns ("nan", step) when the simulation failed numerically,
    ("ended", step) when the episode ended otherwise (a hard failure), or (None, n_steps)."""
    rng = np.random.default_rng(seed * 1000)
    env.reset(seed=seed * 1000)
    sim = env.intervention.simulation
    held = None
    for step in range(1, n_steps + 1):
        if (step - 1) % 5 == 0:
            xt = np.asarray(sim.inserted_lengths, float).reshape(-1)
            bias = [0.4 if xt[i] < 100 else (-0.4 if xt[i] > 250 else 0.0) for i in range(2)]
            held = np.array([np.clip(rng.uniform(-1, 1) + bias[0], -1, 1), rng.uniform(-1, 1),
                             np.clip(rng.uniform(-1, 1) + bias[1], -1, 1), rng.uniform(-1, 1)])
        _, _, terminated, truncated, info = env.step(held.astype(np.float32))
        if info["simulation_error"]:
            return "nan", step
        if terminated or truncated:
            return "ended", step
    return None, n_steps


# ------------------------------------------------------------- a fresh interpreter ----

def run_in_new_process(code: str, **fields):
    """Run ``code`` (``str.format``-ed with ``fields``; it prints ``RESULT <json>``) in a new interpreter
    with this directory on its path and return the JSON. Two scenes simulated in one process differ in
    the last bits (work log 2026-11-02 14:42) and contacts amplify that, so a comparison that must hold
    bit for bit, or a trajectory whose failure step is asserted, gets a fresh process each."""
    code = f"import sys\nsys.path.insert(0, {HERE!r})\n" + code.format(**fields)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    lines = [l for l in run.stdout.splitlines() if l.startswith("RESULT ")]
    assert run.returncode == 0 and lines, f"exit {run.returncode}: {run.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT "):])
