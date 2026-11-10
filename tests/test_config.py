import pytest
import yaml

from guidewire_rl.config import load_config
from guidewire_rl.env.builder import max_steps_of, observation_term_names

from conftest import G13_CONFIG


def test_g13_config_loads(g13_config):
    assert observation_term_names(g13_config["env"]) == ["position", "target", "rotation", "geometry"]
    assert max_steps_of(g13_config["env"]["truncation"]) == 200
    assert [h["label"] for h in g13_config["aux_heads"]] == ["distal_force"]


def test_extends_merges_and_lists_replace(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({
        "extends": str(G13_CONFIG),
        "training": {"total_steps": 1234},
        "env": {"reward": [{"type": "TargetReached", "factor": 2.0}]},
    }))
    config = load_config(child)
    assert config["training"]["total_steps"] == 1234
    assert config["training"]["batch_size"] == 256            # inherited
    assert config["env"]["reward"] == [{"type": "TargetReached", "factor": 2.0}]
    assert len(config["env"]["observation"]) == 4             # inherited


def test_overrides_parse_yaml_and_reject_typos():
    config = load_config(G13_CONFIG, ["training.total_steps=10", "training.device=cpu"])
    assert config["training"]["total_steps"] == 10 and config["training"]["device"] == "cpu"
    with pytest.raises(KeyError):
        load_config(G13_CONFIG, ["training.totl_steps=10"])


@pytest.mark.checkpoint
def test_g13_numbers_match_the_checkpoint(g13_config, g13_checkpoint):
    """configs/g13.yaml must carry the training config stored in step_260000.pt."""
    stored = g13_checkpoint["config"]
    assert g13_config["model"] == stored["model"]
    assert g13_config["planning"] == stored["planning"]
    ours = dict(g13_config["training"])
    assert ours.pop("num_envs") == 1      # ours only: G13 stepped one env
    assert ours.pop("async_envs") is False
    theirs = dict(stored["training"])
    assert g13_config["aux_heads"][0]["weight"] == theirs.pop("lambda_force_train")
    assert ours == theirs
    env, scene = g13_config["env"], g13_config["env"]["scene"]
    e = stored["environment"]
    assert max_steps_of(env["truncation"]) == e["max_episode_steps"]
    assert scene["vessel_tree"]["seed"] == e["vessel_seed"]
    assert scene["fluoroscopy"]["image_frequency"] == e["image_frequency"]
    assert scene["target"]["threshold"] == e["target_threshold"]
    assert scene["target"]["branches"] == e["target_branches"]
    assert scene["simulation"]["friction"] == e["friction"]
    assert g13_config["evaluation"]["episodes"] == stored["evaluation"]["episodes"]
    assert g13_config["evaluation"]["seed"] == stored["evaluation"]["seed"]
    assert g13_config["checkpoint"]["interval"] == stored["checkpoint"]["interval"]


def test_null_drops_an_inherited_key(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({
        "extends": str(G13_CONFIG),
        "env": {"scene": {"vessel_tree": {"type": "Other", "seed": None}}},
    }))
    assert load_config(child)["env"]["scene"]["vessel_tree"] == {
        "type": "Other", "scaling_xyzd": [1.0, 1.0, 1.0, 0.75]}
