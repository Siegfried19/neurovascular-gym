"""Real SOFA scenes (a few seconds each)."""

import copy

import numpy as np
import pytest

from guidewire_rl.env import build_env

pytestmark = pytest.mark.sofa

ACTIONS = np.asarray([[0.9, 0.1], [0.8, -0.3], [1.0, 0.5], [0.6, -0.9],
                      [0.9, 0.2], [1.0, -0.1], [0.7, 0.8], [0.9, -0.5]], dtype=np.float32)


def rollout(env, seed=1000):
    try:
        observation, _ = env.reset(seed=seed)
        steps = [(observation, None, None)]
        for action in ACTIONS:
            observation, reward, _, _, info = env.step(action)
            force = (info["labels"]["distal_force"] if "labels" in info
                     else info["distal_contact_force_max_N"])
            steps.append((observation, reward, force))
        return steps
    finally:
        env.close()


@pytest.mark.reference
def test_g13_config_reproduces_the_reference_env(g13_config):
    """Same seed, same actions: observations, rewards and force labels are
    identical to the reference GeometryForceEnv."""
    from contact_aware_tdmpc2.env.g13_env import make_g13_env
    theirs = rollout(make_g13_env())
    ours = rollout(build_env(g13_config["env"]))
    for (o1, r1, f1), (o2, r2, f2) in zip(theirs, ours):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2 and f1 == f2
    assert any(f > 0 for _, _, f in ours[1:])       # the actions do touch the wall


def test_geometry_sampling_matches_tracking2d(env_config):
    """The check the reference ran every step: our 3 sampled wire points are
    Tracking2D's first 3 points."""
    env = build_env(env_config["env"])
    try:
        env.reset(seed=1000)
        for action in ACTIONS[:4]:
            env.step(action)
            geometry = env.term("geometry")
            sampled = geometry.last_result.sampled_tracking_2d_mm
            tracked = np.asarray(env.term("position").wrapped_obs.obs, dtype=np.float32)[:3]
            assert np.max(np.abs(sampled - tracked)) <= 1e-4
    finally:
        env.close()


def with_geometry(config, **geometry):
    config = copy.deepcopy(config)
    for term in config["env"]["observation"]:
        if term["name"] == "geometry":
            term.update(geometry)
    return config


def test_vessel_frame_only_changes_the_geometry_term(env_config):
    reference = rollout(build_env(env_config["env"]))
    fixed = rollout(build_env(with_geometry(env_config, frame="vessel")["env"]))
    for (o1, r1, f1), (o2, r2, f2) in zip(reference, fixed):
        np.testing.assert_array_equal(o1[:14], o2[:14])   # wire, target, rotation untouched
        assert r1 == r2 and f1 == f2
    # at the insertion point the wire is inside the lumen
    assert (fixed[0][0][14:17] > 0).all()


def test_observation_edits_change_the_dimension(env_config):
    subset = build_env(with_geometry(env_config, features=["clearance"])["env"])
    try:
        assert subset.observation_space.shape == (17,)
        assert subset.observation_names[-3:] == ["d0_normalized", "d1_normalized", "d2_normalized"]
        assert subset.reset(seed=1)[0].shape == (17,)
    finally:
        subset.close()
    config = copy.deepcopy(env_config)
    config["env"]["observation"] = [t for t in config["env"]["observation"]
                                    if t["name"] != "geometry"]
    plain = build_env(config["env"])
    try:
        assert plain.observation_space.shape == (14,)
    finally:
        plain.close()
