import numpy as np
import pytest

from guidewire_rl.train.replay import EpisodeReplay


def make_episode(episode_id: int, length: int, obs_dim: int = 3):
    """Observation k of episode e encodes (e, k) so slices can be checked."""
    observations = np.zeros((length + 1, obs_dim), dtype=np.float32)
    observations[:, 0] = episode_id
    observations[:, 1] = np.arange(length + 1)
    actions = np.stack([np.full(length, episode_id), np.arange(length)], axis=1)
    rewards = np.arange(length) + 100 * episode_id
    terminated = np.zeros(length)
    terminated[-1] = 1
    labels = {"force": np.arange(length) + 0.5 + 100 * episode_id}
    return observations, actions, rewards, terminated, labels


def make_replay(**kwargs):
    defaults = dict(capacity=1000, observation_dim=3, action_dim=2, horizon=3,
                    batch_size=64, label_names=["force"], seed=0)
    return EpisodeReplay(**{**defaults, **kwargs})


def test_slices_are_contiguous_within_one_episode_and_aligned():
    replay = make_replay()
    for e, length in enumerate([5, 2, 9]):        # the 2-step episode is too short
        replay.add(*make_episode(e, length))
    batch = replay.sample()
    H, B = 3, 64
    assert batch.observations.shape == (H + 1, B, 3)
    assert batch.actions.shape == (H, B, 2)
    assert batch.rewards.shape == batch.terminated.shape == batch.labels["force"].shape == (H, B, 1)
    episode = batch.observations[0, :, 0]
    start = batch.observations[0, :, 1]
    assert set(episode.tolist()) <= {0.0, 2.0}       # never the short episode
    for h in range(H + 1):
        assert (batch.observations[h, :, 0] == episode).all()
        assert (batch.observations[h, :, 1] == start + h).all()
    for h in range(H):
        step = start + h                               # transition index in the episode
        assert (batch.actions[h, :, 1] == step).all()
        assert (batch.rewards[h, :, 0] == step + 100 * episode).all()
        assert (batch.labels["force"][h, :, 0] == step + 0.5 + 100 * episode).all()


def test_starts_are_uniform_over_valid_positions():
    replay = make_replay(batch_size=1)
    replay.add(*make_episode(0, 4))     # 2 valid starts
    replay.add(*make_episode(1, 8))     # 6 valid starts
    episodes, _ = replay.sample_indices(40000)
    assert abs(np.mean(episodes == 1) - 6 / 8) < 0.01


def test_cannot_sample_before_a_long_enough_episode():
    replay = make_replay()
    replay.add(*make_episode(0, 2))
    assert not replay.can_sample()
    with pytest.raises(RuntimeError):
        replay.sample()


def test_shape_and_label_checks():
    replay = make_replay()
    obs, act, rew, term, labels = make_episode(0, 5)
    with pytest.raises(ValueError):
        replay.add(obs[:-1], act, rew, term, labels)
    with pytest.raises(ValueError):
        replay.add(obs, act, rew, term, {})


def test_capacity_drops_oldest_episodes():
    replay = make_replay(capacity=12)
    for e in range(4):
        replay.add(*make_episode(e, 5))
    assert replay.size <= 12 and replay.episodes[0]["observations"][0, 0] == 2


def test_state_round_trip_keeps_data_and_random_stream():
    replay = make_replay()
    for e, length in enumerate([6, 7]):
        replay.add(*make_episode(e, length))
    copy = EpisodeReplay.from_state(replay.state_dict())
    a, b = replay.sample(), copy.sample()
    assert (a.observations == b.observations).all() and (a.labels["force"] == b.labels["force"]).all()


@pytest.mark.checkpoint
def test_reference_g13_replay_imports(g13_checkpoint):
    state = g13_checkpoint["replay"]
    replay = EpisodeReplay.from_reference_state(state, {"forces": "distal_force"})
    assert replay.size == 259987 and replay.num_episodes == 3800
    np.testing.assert_array_equal(replay.episodes[7]["labels"]["distal_force"],
                                  state["episodes"][7]["forces"])
    batch = replay.sample()
    assert batch.observations.shape == (4, 256, 22)
