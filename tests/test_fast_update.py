"""training.compile_update (guidewire_rl/model/fast_update.py): the update rearranged for a compiler.

"eager" (rearranged, not compiled) must be the update of ``update_on_batch``: same random draws in the
same order, same parameters afterwards, same reported numbers. Compiled, it is the same update
statistically (the compiler draws its random numbers its own way), tested on the GPU only.
"""

import numpy as np
import pytest
import torch

from guidewire_rl.model import build_agent
from guidewire_rl.train.replay import EpisodeReplay

from conftest import tiny_config

HEADS = [{"name": "force", "label": "force", "loss": "mse", "weight": 0.5, "gradient": "shared",
          "output": "softplus"},
         {"name": "force2", "label": "force", "loss": "huber", "weight": 0.3, "gradient": "detached",
          "output": "identity", "target_scale": 2.0}]

VARIANTS = {
    "plain": dict(),
    "clone the replayed actions": dict(bc_coef=5.0),
    "round 2 as trained": dict(bc_coef=1.0, bc_target="expert", bc_loss="smooth_l1", policy_smooth_coef=1.0,
                               policy_mean_bound=3.0, policy_mean_bound_coef=0.5),
    "bound reported, not pulled": dict(bc_coef=1.0, bc_target="expert", policy_mean_bound=3.0),
    "self imitation after the teacher": dict(bc_coef=1.0, bc_target="expert", bc_loss="smooth_l1",
                                             policy_smooth_coef=1.0, self_bc_coef=0.25),
}


def replay_with_everything(seed=0, batch_size=16, terminal=True):
    """Own and scripted episodes, expert targets on some steps (NaN elsewhere), one terminal step."""
    rng = np.random.default_rng(seed)
    replay = EpisodeReplay(capacity=2000, observation_dim=6, action_dim=2, horizon=3,
                           batch_size=batch_size, label_names=["force"], seed=seed)
    for episode in range(6):
        length = 14
        expert = np.full((length, 2), np.nan, np.float32)
        expert[:9] = [0.3, -0.2]
        ended = np.zeros(length)
        if terminal and episode % 2 == 0:
            ended[-1] = 1.0
        replay.add(rng.uniform(-1, 1, (length + 1, 6)), rng.uniform(-1, 1, (length, 2)),
                   rng.uniform(-1, 1, length), ended, {"force": rng.uniform(0, 3, length)},
                   source=np.full(length, episode % 3, np.int8), expert=expert,
                   protected=(episode == 3))            # one own success kept as a self-demonstration
    return replay


def make(mode, training, device="cpu", dropout=0.01):
    config = tiny_config(compile_update=mode, **training)
    config["model"]["dropout"] = dropout
    config["aux_heads"] = HEADS
    torch.manual_seed(0)
    return build_agent(config, observation_dim=6, action_dim=2, episode_length=50, device=device)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_rearranged_update_is_the_reference_update(name):
    reference, fast = make(False, VARIANTS[name]), make("eager", VARIANTS[name])
    replays = replay_with_everything(), replay_with_everything()
    for step in range(6):
        if step == 4:                                   # the schedule fades the cloning
            reference.bc_scale = fast.bc_scale = 0.5
        infos = []
        for agent, replay in zip((reference, fast), replays):
            torch.manual_seed(100 + step)
            infos.append(agent.update(replay))
        assert list(infos[0]) == list(infos[1]), "same numbers reported, in the same order"
        for key, value in infos[0].items():
            if value is None:
                assert infos[1][key] is None, key
            else:
                assert infos[1][key] == pytest.approx(value, rel=1e-5, abs=1e-7), key
    for (key, a), b in zip(reference.model.state_dict().items(), fast.model.state_dict().values()):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), key
    assert torch.allclose(reference.scale.value, fast.scale.value, rtol=1e-6)
    assert reference.update_count == fast.update_count == 6
    assert not any(module.training for module in fast.model.modules()), "left in eval mode, like the reference"


def test_rearranged_update_reports_cloning_and_termination_only_when_there_is_something_to_report():
    training = dict(bc_coef=5.0)
    replay = replay_with_everything()
    for mode in (False, "eager"):
        agent = make(mode, training)
        assert "bc_loss" in agent.update(replay)
        agent.bc_scale = 0.0                                # the schedule took the demonstrations away
        assert "bc_loss" not in agent.update(replay)
        agent.bc_scale = 1.0
        agent.config["bc_coef"] = 0.0
        assert "bc_loss" not in agent.update(replay)
        no_end = agent.update(replay_with_everything(terminal=False))
        assert no_end["termination_recall"] is None and no_end["termination_false_positive"] is not None


def test_bad_setting_fails_loudly():
    with pytest.raises(ValueError, match="compile_update"):
        make("fast", {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled update: GPU only")
def test_compiled_update_learns_like_the_uncompiled_one_and_draws_fresh_noise():
    training = VARIANTS["round 2 as trained"]

    def run(mode, updates=300):
        agent = make(mode, training, device="cuda:0")
        replay = replay_with_everything(batch_size=64)
        torch.manual_seed(0)
        infos = [agent.update(replay) for _ in range(updates)]
        return agent, replay, infos

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)     # the fallback to uncompiled is a failure here
        compiled, replay, c_infos = run(True)
    assert compiled._fast_update.mode == "compiled"
    _, _, e_infos = run("eager")
    assert list(c_infos[-1]) == list(e_infos[-1])

    def mean(infos, key):
        return float(np.mean([i[key] for i in infos]))

    # measured 2026-11-15 (RTX 3060): the two agree to about 1 % on every reported number after 400 updates
    for key in ("consistency_loss", "reward_loss", "value_loss", "aux_force_loss"):
        assert mean(c_infos[-30:], key) < 0.8 * mean(c_infos[:20], key), f"{key} did not go down"
    for key in ("consistency_loss", "reward_loss", "value_loss", "bc_loss", "aux_force_loss", "policy_loss",
                "policy_smooth_loss", "model_grad_norm", "policy_grad_norm"):
        assert mean(c_infos[-30:], key) == pytest.approx(mean(e_infos[-30:], key), rel=0.15, abs=5e-3), key
    assert mean(c_infos[-30:], "policy_entropy") == pytest.approx(mean(e_infos[-30:], "policy_entropy"), abs=0.3)

    # the same batch twice without a step in between: the policy's sample and the critics' dropout
    # must be new draws each time (a compiled graph that froze its noise would repeat the number)
    fast = compiled._fast_update
    batch = replay.sample("cuda:0")
    fast._set_dropout(True)
    for p in fast._critic_params:
        p.requires_grad_(False)
    losses = []
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        _, rollout, _ = fast._model_fn(batch.observations, batch.actions, batch.rewards, batch.terminated,
                                       tuple(batch.labels[s.label] for s in compiled.aux_specs))
        loss, _, _ = fast._policy_fn(rollout.clone(), batch.actions, batch.source, batch.expert,
                                     compiled.scale.value, fast._coef)
        losses.append(float(loss))
    for p in fast._critic_params:
        p.requires_grad_(True)
    fast._set_dropout(False)
    assert len(set(losses)) == 3, f"policy loss repeated: {losses}"
