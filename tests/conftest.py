from __future__ import annotations

import copy

import pytest

from guidewire_rl import REFERENCE_G13_CHECKPOINT, WORKSPACE_ROOT
from guidewire_rl.config import load_config

G13_CONFIG = WORKSPACE_ROOT / "configs" / "g13.yaml"
UPSTREAM_CONFIG = WORKSPACE_ROOT / "configs" / "g13_upstream.yaml"


def _sofa_importable() -> bool:
    try:
        import Sofa  # noqa: F401
        return True
    except Exception:
        return False


def _upstream_steve() -> bool:
    from simulator_modifications import UPSTREAM
    return UPSTREAM


def _reference_importable() -> bool:
    try:
        import contact_aware_tdmpc2  # noqa: F401
        return True
    except Exception:
        return False


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="also run the tests marked slow (minutes each, e.g. the C-arm angle search)")


def pytest_collection_modifyitems(config, items):
    no_sofa = pytest.mark.skip(reason="Sofa not importable; run via "
                               "`conda run -n guidewire_rl pytest` (or contact_aware_tdmpc2)")
    no_slow = pytest.mark.skip(reason="slow: run with --run-slow")
    run_slow = config.getoption("--run-slow")
    no_checkpoint = pytest.mark.skip(reason=f"missing {REFERENCE_G13_CHECKPOINT}")
    no_upstream = pytest.mark.skip(reason="needs upstream stEVE: run in the conda env guidewire_rl")
    no_reference = pytest.mark.skip(reason="needs the reference package (Wilson's eve): "
                                    "run in the conda env contact_aware_tdmpc2")
    sofa_ok = _sofa_importable()
    upstream = _upstream_steve()
    reference = _reference_importable()
    for item in items:
        if "sofa" in item.keywords and not sofa_ok:
            item.add_marker(no_sofa)
        if "checkpoint" in item.keywords and not REFERENCE_G13_CHECKPOINT.is_file():
            item.add_marker(no_checkpoint)
        if "upstream" in item.keywords and not upstream:
            item.add_marker(no_upstream)
        if "reference" in item.keywords and not reference:
            item.add_marker(no_reference)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(no_slow)


@pytest.fixture
def g13_config():
    return load_config(G13_CONFIG)


@pytest.fixture
def upstream_config():
    return load_config(UPSTREAM_CONFIG)


@pytest.fixture
def env_config():
    """The G13 task on whichever stEVE this env has: g13.yaml with Wilson's eve,
    g13_upstream.yaml with upstream stEVE."""
    return load_config(UPSTREAM_CONFIG if _upstream_steve() else G13_CONFIG)


@pytest.fixture(scope="session")
def g13_checkpoint():
    from guidewire_rl.train.checkpoint import load_checkpoint
    return load_checkpoint(REFERENCE_G13_CHECKPOINT)


def tiny_config(**training):
    """G13 config shrunk so an agent trains in milliseconds on CPU."""
    config = copy.deepcopy(load_config(G13_CONFIG))
    config["model"].update(latent_dim=16, enc_dim=32, mlp_dim=32, num_q=2, num_bins=11)
    config["planning"].update(iterations=2, num_samples=16, num_elites=4, num_pi_trajs=2)
    config["training"].update(device="cpu", batch_size=8, seed_steps=20, initial_updates=10,
                              total_steps=60, **training)
    config["evaluation"].update(interval=30, episodes=2)
    config["checkpoint"].update(interval=30)
    config["logging"].update(update_interval=5, print_interval=1000)
    return config
