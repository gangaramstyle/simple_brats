from dataclasses import replace

import pytest

from simple_brats.config import ExperimentConfig, ModelConfig, load_experiment_config


def test_registered_config_loads_and_has_stable_digest() -> None:
    first = load_experiment_config("configs/v0_cross_matching.toml")
    second = load_experiment_config("configs/v0_cross_matching.toml")
    assert first == second
    assert len(first.sha256) == 64
    assert first.sha256 == second.sha256
    assert first.patch.tensor_shape == (16, 16, 1)


def test_rotary_head_width_must_be_even() -> None:
    with pytest.raises(ValueError, match="head width must be even"):
        ModelConfig(width=30, heads=6)


def test_artifact_and_checkpoint_cadences_must_align() -> None:
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="coincide"):
        replace(config, artifact_every_steps=5_001)
