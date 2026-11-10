"""Our copy of the TD-MPC2 model/planner/update must behave exactly like the
reference release when loaded with the G13 checkpoint (CPU, one thread)."""

import numpy as np
import pytest
import torch

from guidewire_rl import REFERENCE_G13_CHECKPOINT
from guidewire_rl.model import build_agent
from guidewire_rl.train.checkpoint import reference_agent_state
from guidewire_rl.train.replay import EpisodeReplay

pytestmark = [pytest.mark.checkpoint, pytest.mark.reference]


@pytest.fixture(scope="module")
def reference_agent():
    from contact_aware_tdmpc2.tdmpc2.checkpoint import load_g13
    torch.set_num_threads(1)
    agent, _ = load_g13(REFERENCE_G13_CHECKPOINT, device="cpu")
    return agent


def our_agent(config, checkpoint, weight=None):
    config["training"]["device"] = "cpu"
    if weight is not None:
        config["aux_heads"][0]["weight"] = weight
    agent = build_agent(config, observation_dim=22, action_dim=2, episode_length=200)
    agent.load_state_dict(reference_agent_state(checkpoint, "distal_force"), load_optimizers=False)
    return agent


def test_planner_action_matches_reference(g13_config, g13_checkpoint, reference_agent):
    agent = our_agent(g13_config, g13_checkpoint)
    observation = np.linspace(-0.5, 0.5, 22, dtype=np.float32)
    torch.set_num_threads(1)
    torch.manual_seed(73)
    ours = agent.act(observation, first_step=True, eval_mode=True)
    # value recorded by the reference repo's own test for this checkpoint
    np.testing.assert_allclose(ours, [-0.9963191, -0.9924057], atol=1e-6)
    torch.manual_seed(73)
    theirs = reference_agent.act(observation, first_step=True, eval_mode=True)
    np.testing.assert_array_equal(ours, theirs)


def test_force_prediction_matches_reference(g13_config, g13_checkpoint, reference_agent):
    agent = our_agent(g13_config, g13_checkpoint)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.uniform(-1, 1, 22).astype(np.float32)
        action = rng.uniform(-1, 1, 2).astype(np.float32)
        ours = agent.aux_predict("distal_force", obs, action)
        with torch.no_grad():
            o = torch.as_tensor(obs).unsqueeze(0)
            a = torch.as_tensor(action).unsqueeze(0)
            m = reference_agent.model
            theirs = float(m.force(m.next(m.encode(o), a))[0, 0])
        assert ours == theirs


def test_one_update_matches_reference_when_force_weight_is_zero(
        g13_config, g13_checkpoint, reference_agent):
    """The reference release has no force loss; with weight 0 our update must
    change every shared parameter exactly as the reference update does."""
    from contact_aware_tdmpc2.tdmpc2.checkpoint import load_g13
    torch.set_num_threads(1)
    theirs, _ = load_g13(REFERENCE_G13_CHECKPOINT, device="cpu")
    # load_g13 freezes the model for inference; unfreeze it like a fresh agent.
    theirs.model.requires_grad_(True)
    theirs.model.target_q_ensemble.requires_grad_(False)
    ours = our_agent(g13_config, g13_checkpoint, weight=0.0)
    replay = EpisodeReplay.from_reference_state(g13_checkpoint["replay"],
                                                {"forces": "distal_force"})
    batch = replay.sample()

    class OneBatch:
        def sample(self, device):
            return batch.observations, batch.actions, batch.rewards, batch.terminated

    torch.manual_seed(5)
    theirs.update(OneBatch())
    torch.manual_seed(5)
    ours.update_on_batch(batch)
    their_state = theirs.model.state_dict()
    for key, value in ours.model.state_dict().items():
        if key.startswith("aux_heads."):
            continue
        assert torch.equal(value, their_state[key]), key
