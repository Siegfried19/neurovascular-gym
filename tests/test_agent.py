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


def _scripted_replay(seed=0, source_kind=1):
    """Episodes whose actions are a fixed scripted action; ``source_kind`` per step (1 = prefix)."""
    rng = np.random.default_rng(seed)
    replay = EpisodeReplay(capacity=1000, observation_dim=6, action_dim=2, horizon=3,
                           batch_size=16, label_names=["force"], seed=seed)
    for _ in range(4):
        length = 12
        replay.add(rng.uniform(-1, 1, (length + 1, 6)), np.tile([0.3, -0.2], (length, 1)),
                   rng.uniform(-1, 1, length), np.zeros(length), {"force": rng.uniform(0, 3, length)},
                   source=np.full(length, source_kind, np.int8))
    return replay


def test_batch_carries_each_transitions_source():
    from guidewire_rl.train.replay import SOURCE_PREFIX
    batch = _scripted_replay(source_kind=SOURCE_PREFIX).sample()
    assert batch.source.shape == batch.actions.shape[:2]
    assert (batch.source == SOURCE_PREFIX).all()


def test_behaviour_cloning_pulls_the_prior_to_the_scripted_actions():
    """training.bc_coef (2026-11-13): the policy prior clones the scripted transitions (prefix and
    demonstrations), not the policy's own; off (0) by default; faded by bc_scale."""
    def prior_gap(agent, batch):
        with torch.no_grad():
            z = agent.model.encode(batch.observations[0])
            _, info = agent.model.pi(z)
        return float((info["mean"] - torch.tensor([0.3, -0.2])).pow(2).mean())

    replay = _scripted_replay()
    config = tiny_config(bc_coef=5.0)
    config["aux_heads"] = head()
    torch.manual_seed(0)
    agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    batch = replay.sample()
    before = prior_gap(agent, batch)
    for _ in range(60):
        info = agent.update(replay)
    assert info["bc_loss"] >= 0 and info["bc_share"] == pytest.approx(1.0)
    assert prior_gap(agent, batch) < 0.25 * before
    # the policy's own transitions are not cloned; no coefficient, no cloning
    own = _scripted_replay(source_kind=0)
    assert "bc_loss" not in agent.update(own)
    agent.config["bc_coef"] = 0.0
    assert "bc_loss" not in agent.update(replay)
    agent.config["bc_coef"] = 5.0
    agent.bc_scale = 0.0                                  # the schedule took the demonstrations away
    assert "bc_loss" not in agent.update(replay)


TARGET = [0.3, -0.2]


def _shadow_replay(seed=0, labelled_steps=8):
    """The policy's own episodes (source 0, random actions) whose first ``labelled_steps`` steps carry
    the shadow expert's target ``TARGET``; the rest have none (NaN)."""
    rng = np.random.default_rng(seed)
    replay = EpisodeReplay(capacity=1000, observation_dim=6, action_dim=2, horizon=3,
                           batch_size=16, label_names=["force"], seed=seed)
    for _ in range(4):
        length = 12
        expert = np.full((length, 2), np.nan, np.float32)
        expert[:labelled_steps] = TARGET
        replay.add(rng.uniform(-1, 1, (length + 1, 6)), rng.uniform(-1, 1, (length, 2)),
                   rng.uniform(-1, 1, length), np.zeros(length), {"force": rng.uniform(0, 3, length)},
                   source=np.zeros(length, np.int8), expert=expert)
    return replay


def test_behaviour_cloning_on_expert_targets_reaches_the_policys_own_transitions():
    """training.bc_target: expert (DAgger-style, 2026-11-13): the prior clones the replay's expert target
    on every transition that has one, the policy's own included; rows without one (NaN) are left out;
    bc_target action (the default) leaves the policy's own transitions alone as before."""
    def prior_gap(agent, batch):
        with torch.no_grad():
            z = agent.model.encode(batch.observations[0])
            _, info = agent.model.pi(z)
        mask = torch.isfinite(batch.expert[0]).all(dim=-1)
        return float((info["mean"][mask] - torch.tensor(TARGET)).pow(2).mean())

    config = tiny_config(bc_coef=5.0, bc_target="expert")
    config["aux_heads"] = head()
    torch.manual_seed(0)
    agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    replay = _shadow_replay()
    batch = replay.sample()
    before = prior_gap(agent, batch)
    for _ in range(60):
        info = agent.update(replay)
    assert info["bc_loss"] >= 0 and 0.0 < info["bc_share"] < 1.0      # only the labelled rows
    assert prior_gap(agent, batch) < 0.25 * before
    assert all(torch.isfinite(p).all() for p in agent.model.policy.parameters())
    assert "bc_loss" not in agent.update(_shadow_replay(labelled_steps=0))   # no target anywhere: no cloning
    # the default leaves the policy's own transitions alone, targets or not
    config = tiny_config(bc_coef=5.0)
    config["aux_heads"] = head()
    plain = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    assert plain.bc_target == "action" and "bc_loss" not in plain.update(replay)
    with pytest.raises(ValueError):
        build_agent(tiny_config(bc_target="teacher"), observation_dim=6, action_dim=2, episode_length=50,
                    device="cpu")


@pytest.mark.parametrize("bound", [None, 0.5])
def test_expert_cloning_loss_is_the_gap_on_the_labelled_rows_only(bound):
    """The loss is mean over labelled (H, B) rows of mean_a (prior mean - clamp(target))^2; a NaN row
    adds nothing, whatever the other rows hold, and no NaN reaches the gradient. The clamp is +-1, or
    +-tanh(b) with training.policy_mean_bound b (lead 2026-11-13)."""
    config = tiny_config(bc_coef=1.0, bc_target="expert", entropy_coef=0.0, policy_mean_bound=bound)
    torch.manual_seed(0)
    agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    H, B = 3, 5
    latent = torch.randn(H + 1, B, agent.model.latent_dim)
    expert = torch.tensor([1.7, -0.4]).expand(H, B, 2).clone()        # 1.7: clamped to 1
    expert[0, 1] = float("nan")
    expert[2, :3] = float("nan")
    mask = torch.isfinite(expert).all(dim=-1)
    with torch.no_grad():
        mean = agent.model.pi(latent)[1]["mean"][:H]
    limit = 1.0 if bound is None else float(np.tanh(bound))
    expected = (mean - expert.clamp(-limit, limit)).pow(2).mean(dim=-1)[mask].mean()
    info = agent._update_policy(latent, actions=torch.zeros(H, B, 2), source=torch.zeros(H, B, dtype=torch.long),
                                expert=expert)
    assert info["bc_loss"] == pytest.approx(float(expected), rel=1e-6)
    assert info["bc_share"] == pytest.approx(float(mask.float().mean()))
    assert all(torch.isfinite(p).all() for p in agent.model.policy.parameters())


def test_smooth_l1_cloning_pulls_as_hard_on_a_small_miss_as_on_a_large_one():
    """training.bc_loss: smooth_l1 (lead 2026-11-14): beyond bc_loss_beta the loss is |gap| - beta/2,
    inside it gap^2 / (2 beta); bc_abs_error reports the mean absolute gap with either kind; l2 (the
    default) is unchanged. The point: the gradient on a slow target does not shrink with the gap."""
    H, B = 2, 4
    expert = torch.tensor([0.12, -0.6]).expand(H, B, 2).clone()

    def run(**training):
        config = tiny_config(bc_coef=1.0, bc_target="expert", entropy_coef=0.0, **training)
        torch.manual_seed(0)
        agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
        latent = torch.randn(H + 1, B, agent.model.latent_dim)
        with torch.no_grad():
            miss = (agent.model.pi(latent)[1]["mean"][:H] - expert).abs()
        info = agent._update_policy(latent, actions=torch.zeros(H, B, 2),
                                    source=torch.zeros(H, B, dtype=torch.long), expert=expert)
        return agent, miss, info

    agent, miss, info = run(bc_loss="smooth_l1", bc_loss_beta=0.05)
    assert agent.bc_loss_kind == "smooth_l1" and agent.bc_loss_beta == 0.05
    expected = torch.where(miss < 0.05, 0.5 * miss.pow(2) / 0.05, miss - 0.025).mean()
    assert info["bc_loss"] == pytest.approx(float(expected), rel=1e-5)
    assert info["bc_abs_error"] == pytest.approx(float(miss.mean()), rel=1e-5)
    agent, miss, info = run()                                   # the default: squared, as before
    assert agent.bc_loss_kind == "l2"
    assert info["bc_loss"] == pytest.approx(float(miss.pow(2).mean()), rel=1e-5)
    assert info["bc_abs_error"] == pytest.approx(float(miss.mean()), rel=1e-5)
    # the gradient on the prior's mean: l2 shrinks with the gap, smooth_l1 does not (beyond beta)
    for gap, l2_grad, l1_grad in ((0.5, 1.0, 1.0), (0.1, 0.2, 1.0), (0.02, 0.04, 0.4)):
        m = torch.tensor(gap, requires_grad=True)
        (m.pow(2)).backward()
        assert float(m.grad) == pytest.approx(l2_grad)
        m = torch.tensor(gap, requires_grad=True)
        torch.where(m < 0.05, 0.5 * m.pow(2) / 0.05, m - 0.025).backward()
        assert float(m.grad) == pytest.approx(l1_grad)
    for bad in ({"bc_loss": "huber"}, {"bc_loss": "smooth_l1", "bc_loss_beta": -1.0}):
        with pytest.raises(ValueError):
            build_agent(tiny_config(**bad), observation_dim=6, action_dim=2, episode_length=50, device="cpu")


def test_the_smoothness_term_makes_the_prior_change_less_from_step_to_step():
    """training.policy_smooth_coef (2026-11-13): the policy prior's own loss carries the mean squared
    gap between its mean actions on consecutive latents of the rollout (every transition, not only the
    scripted ones); off (0) by default."""
    def step_gap(agent, batch):
        with torch.no_grad():
            z0 = agent.model.encode(batch.observations[0])
            z1 = agent.model.next(z0, batch.actions[0])
            return float((agent.model.pi(z1)[1]["mean"] - agent.model.pi(z0)[1]["mean"]).pow(2).mean())

    def trained(coef):
        replay = _scripted_replay(source_kind=0)             # the policy's own transitions
        config = tiny_config(policy_smooth_coef=coef)
        config["aux_heads"] = head()
        torch.manual_seed(0)
        agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
        batch = replay.sample()
        for _ in range(60):
            info = agent.update(replay)
        return info, step_gap(agent, batch)

    plain, gap_plain = trained(0.0)
    smooth, gap_smooth = trained(20.0)
    assert "policy_smooth_loss" not in plain
    assert smooth["policy_smooth_loss"] >= 0 and smooth["policy_smooth_coef"] == 20.0
    assert gap_smooth < 0.5 * gap_plain


def test_self_imitation_clones_only_the_transitions_of_kept_own_successes():
    """training.self_bc_coef (2026-11-16): the cloning gap to the replayed action on the transitions
    a batch marks SOURCE_SELF_DEMO (3), whatever bc_scale is; off (0) by default; nothing is added
    when the batch has no such transition."""
    H, B = 2, 4
    actions = torch.tensor([0.7, -0.3]).expand(H, B, 2).clone()
    source = torch.zeros(H, B, dtype=torch.long)
    source[:, :2] = 3                                         # half the batch: own successes

    def run(source, **training):
        config = tiny_config(entropy_coef=0.0, bc_loss="smooth_l1", **training)
        torch.manual_seed(0)
        agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
        agent.bc_scale = 0.0                                  # the teacher is gone
        latent = torch.randn(H + 1, B, agent.model.latent_dim)
        with torch.no_grad():
            miss = (agent.model.pi(latent)[1]["mean"][:H] - actions).abs()
        return miss, agent._update_policy(latent, actions=actions, source=source)

    miss, info = run(source, self_bc_coef=0.5)
    gap = torch.where(miss < 0.05, 0.5 * miss.pow(2) / 0.05, miss - 0.025).mean(dim=-1)
    assert info["self_bc_loss"] == pytest.approx(float(gap[:, :2].mean()), rel=1e-5)
    assert info["self_bc_share"] == pytest.approx(0.5)
    _, off = run(source)
    assert "self_bc_loss" not in off
    assert info["policy_loss"] == pytest.approx(off["policy_loss"] + 0.5 * info["self_bc_loss"], rel=1e-4)
    _, none = run(torch.zeros(H, B, dtype=torch.long), self_bc_coef=0.5)
    assert "self_bc_loss" not in none
    with pytest.raises(ValueError):
        build_agent(tiny_config(self_bc_coef=-1.0), observation_dim=6, action_dim=2, episode_length=50,
                    device="cpu")


def test_config_std_bounds_are_honoured_and_win_over_a_loaded_checkpoint():
    """model.log_std_max (lead 2026-11-13: -1.6 in the DAgger config, sigma <= 0.2 before tanh) bounds
    the policy's std; the bounds are buffers saved with the model, and the config's win on loading."""
    def largest_log_std(agent):
        with torch.no_grad():
            return float(agent.model.pi(torch.randn(256, agent.model.latent_dim) * 5)[1]["log_std"].max())

    narrow = tiny_config()
    narrow["model"]["log_std_max"] = -1.6
    torch.manual_seed(0)
    agent = build_agent(narrow, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    assert largest_log_std(agent) <= -1.6 + 1e-6
    wide = build_agent(tiny_config(), observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    with torch.no_grad():                                     # its std at its cap (e^2), as measured at 160k
        wide.model.policy[-1].bias[2:].fill_(10.0)
    assert largest_log_std(wide) > 1.9
    agent.load_state_dict(wide.state_dict())                 # a checkpoint trained with the cap at 2
    assert largest_log_std(agent) <= -1.6 + 1e-6
    assert float(agent.model.log_std_range) == pytest.approx(-1.6 + 10.0)


def test_policy_mean_bound_pulls_saturated_means_back_and_reports_them():
    """training.policy_mean_bound b with policy_mean_bound_coef c (lead 2026-11-13): the loss gains
    c * mean(relu(|mu_pre| - b)^2) over the rollout's latents (the mean before tanh, its gradient not
    through tanh), the metrics report the share above b and the largest |mu_pre|; off by default."""
    def trained(**training):
        # Adam moves a parameter by about lr per update: a larger lr than the runs' 3e-4 so 60 updates
        # show the pull (from |mu| ~ 20 the bound needs ~1e4 updates at 3e-4; it is meant to keep a
        # fresh run from saturating)
        config = tiny_config(bc_coef=1.0, lr=3e-3, **training)
        config["aux_heads"] = head()
        torch.manual_seed(0)
        agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
        with torch.no_grad():                                 # saturate the prior: mu_pre ~ 20
            agent.model.policy[-1].bias[:2].fill_(20.0)
        replay = _scripted_replay()
        first = agent.update(replay)
        for _ in range(60):
            info = agent.update(replay)
        with torch.no_grad():
            z = agent.model.encode(replay.sample().observations[0])
            size = float(agent.model.pi(z)[1]["mean_pre"].abs().mean())
        return first, info, size

    first, info, size = trained(policy_mean_bound=2.5, policy_mean_bound_coef=0.05)
    assert first["policy_mean_over_bound"] > 0.9 and first["policy_mean_abs_max"] > 15
    assert first["policy_mean_bound_loss"] > 100
    assert info["policy_mean_bound_loss"] < 0.7 * first["policy_mean_bound_loss"]
    _, plain_info, plain_size = trained()
    assert "policy_mean_over_bound" not in plain_info and plain_size > 18     # nothing pulls it back
    assert size < plain_size - 3
    _, watched, _ = trained(policy_mean_bound=2.5)                         # bound without a coefficient:
    assert "policy_mean_over_bound" in watched and "policy_mean_bound_loss" not in watched   # metrics only
    with pytest.raises(ValueError):
        build_agent(tiny_config(policy_mean_bound=0.0), observation_dim=6, action_dim=2, episode_length=50,
                    device="cpu")


def test_policy_mean_bound_loss_value():
    config = tiny_config(entropy_coef=0.0, policy_mean_bound=1.0, policy_mean_bound_coef=1.0)
    torch.manual_seed(0)
    agent = build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device="cpu")
    latent = torch.randn(4, 5, agent.model.latent_dim) * 3
    with torch.no_grad():
        agent.model.policy[-1].bias[:2].fill_(1.5)
        size = agent.model.pi(latent)[1]["mean_pre"].abs()
    info = agent._update_policy(latent)
    assert info["policy_mean_bound_loss"] == pytest.approx(float(torch.relu(size - 1.0).pow(2).mean()), rel=1e-6)
    assert info["policy_mean_over_bound"] == pytest.approx(float((size > 1.0).float().mean()))
    assert info["policy_mean_abs_max"] == pytest.approx(float(size.max()))
