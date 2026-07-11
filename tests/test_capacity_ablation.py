from simple_brats.config import load_experiment_config
from simple_brats.training import build_matching_system


def _trainable_parameters(config_path: str) -> int:
    system = build_matching_system(load_experiment_config(config_path))
    return sum(parameter.numel() for parameter in system.parameters() if parameter.requires_grad)


def test_small_capacity_arm_changes_only_registered_model_scale() -> None:
    base = load_experiment_config("configs/v0_cross_matching.toml")
    small = load_experiment_config("configs/v0_cross_matching_small.toml")

    assert small.patch == base.patch
    assert small.task == base.task
    assert small.seed == base.seed
    assert small.checkpoint_every_steps == base.checkpoint_every_steps
    assert small.artifact_every_steps == base.artifact_every_steps
    assert (base.model.width, base.model.depth, base.model.heads) == (384, 12, 6)
    assert (small.model.width, small.model.depth, small.model.heads) == (256, 8, 4)
    assert small.model.mlp_ratio == base.model.mlp_ratio
    assert small.model.predictor_depth == base.model.predictor_depth
    assert small.model.teacher_ema_momentum == base.model.teacher_ema_momentum


def test_registered_capacity_counts_are_stable() -> None:
    base_parameters = _trainable_parameters("configs/v0_cross_matching.toml")
    small_parameters = _trainable_parameters("configs/v0_cross_matching_small.toml")

    assert base_parameters == 24_203_904
    assert small_parameters == 7_963_392
    assert base_parameters / small_parameters > 3.0
