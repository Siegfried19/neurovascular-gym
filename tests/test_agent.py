import numpy as np
import pytest
import torch

from guidewire_rl.model import build_agent
from guidewire_rl.train.replay import EpisodeReplay

from conftest import tiny_config


def make_agent(aux_heads, seed=0):
    config = tiny_config()
    config["aux_heads"] = aux_heads
    torch.manual_seed(seed)
    return build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")


def fixed_batch(seed=0):
    rng = np.random.default_rng(seed)
    replay = EpisodeReplay(capacity=1000, observation_dim=6, action_dim=2, horizon=3,
                           batch_size=8, label_names=["force"], seed=seed)
    for _ in range(3):
        length = 12
        replay.add(rng.uniform(-1, 1, (length + 1, 6)), rng.uniform(-1, 1, (length, 2)),
                   rng.uniform(-1, 1, length), np.zeros(length),
                   {"force": rng.uniform(0, 3, length)})
    return replay.sample()


def head(gradient="shared", weight=0.5):
    return [{"name": "force", "label": "force", "loss": "mse", "weight": weight,
             "gradient": gradient, "output": "softplus"}]


def shared_params(agent):
    return {k: v.clone() for k, v in agent.model.state_dict().items()
            if not k.startswith("aux_heads.")}


def test_aux_head_gets_its_own_optimizer_group():
    plain, with_head = make_agent([]), make_agent(head())
    assert len(with_head.model_optimizer.param_groups) == len(plain.model_optimizer.param_groups) + 1
    assert any(k.startswith("aux_heads.force.") for k in with_head.model.state_dict())
    value = with_head.aux_predict("force", np.zeros(6, np.float32), np.zeros(2, np.float32))
    assert np.isfinite(value) and value >= 0          # softplus output


def run_update(agent, batch):
    torch.manual_seed(123)
    metrics = agent.update_on_batch(batch)
    return metrics, shared_params(agent)


def test_detached_head_leaves_world_model_untouched_shared_head_does_not():
    batch = fixed_batch()
    _, plain = run_update(make_agent([]), batch)
    metrics, detached = run_update(make_agent(head("detached")), batch)
    _, shared = run_update(make_agent(head("shared")), batch)
    assert "aux_force_loss" in metrics
    for key in plain:
        assert torch.equal(plain[key], detached[key]), key
    assert not torch.equal(plain["encoder.0.weight"], shared["encoder.0.weight"])


def test_zero_weight_shared_head_equals_no_head():
    batch = fixed_batch()
    _, plain = run_update(make_agent([]), batch)
    _, zero = run_update(make_agent(head("shared", weight=0.0)), batch)
    for key in plain:
        assert torch.equal(plain[key], zero[key]), key


def test_force_head_learns_on_a_fixed_batch():
    agent = make_agent(head("shared", weight=1.0))
    batch = fixed_batch()
    first = agent.update_on_batch(batch)["aux_force_loss"]
    for _ in range(60):
        last = agent.update_on_batch(batch)["aux_force_loss"]
    assert last < first


def test_bad_head_settings_fail_loudly():
    with pytest.raises(ValueError):
        make_agent([{"name": "f", "label": "f", "loss": "mae"}])
    with pytest.raises(ValueError):
        make_agent([{"name": "f", "label": "f", "gradient": "partial"}])
