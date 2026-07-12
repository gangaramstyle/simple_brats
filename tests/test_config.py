from dataclasses import replace

import pytest

from simple_brats.config import ExperimentConfig, ModelConfig, PatchConfig, load_experiment_config


def test_registered_config_loads_and_has_stable_digest() -> None:
    first = load_experiment_config("configs/v0_cross_matching.toml")
    second = load_experiment_config("configs/v0_cross_matching.toml")
    assert first == second
    assert len(first.sha256) == 64
    assert first.sha256 == second.sha256
    assert first.patch.physical_extent_mm == (4.0, 4.0, 4.0)
    assert first.patch.source_shape == (4, 4, 4)
    assert first.patch.tensor_shape == (8, 8, 8)
    assert first.patch.is_cubic


def test_registered_8mm_ablation_changes_physical_scale_not_model_shape() -> None:
    primary = load_experiment_config("configs/v0_cross_matching_small.toml")
    ablation = load_experiment_config("configs/v0_cross_matching_small_8mm.toml")

    assert primary.model == ablation.model
    assert primary.task.prism_extent_mm == (32.0, 32.0, 32.0)
    assert ablation.task.prism_extent_mm == (64.0, 64.0, 64.0)
    assert primary.task.source_patches_per_bag == ablation.task.source_patches_per_bag == 96
    assert primary.registered_single_d_arm == "32mm-prism_4mm-cube"
    assert ablation.registered_single_d_arm == "64mm-prism_8mm-cube"
    assert ablation.patch.physical_extent_mm == (8.0, 8.0, 8.0)
    assert ablation.patch.source_shape == (8, 8, 8)
    assert ablation.patch.tensor_shape == primary.patch.tensor_shape == (8, 8, 8)


def test_only_registered_cube_or_legacy_slab_geometry_is_accepted() -> None:
    assert PatchConfig(footprint_mm=4.0, thin_mm=1.0, tensor_shape=(16, 16, 1))
    with pytest.raises(ValueError, match="isotropic cube"):
        PatchConfig(footprint_mm=8.0, thin_mm=1.0, tensor_shape=(16, 16, 1))


def test_rotary_head_width_must_be_even() -> None:
    with pytest.raises(ValueError, match="head width must be even"):
        ModelConfig(width=30, heads=6)


def test_artifact_and_checkpoint_cadences_must_align() -> None:
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="coincide"):
        replace(config, artifact_every_steps=5_001)
