import copy
import inspect
from dataclasses import replace

import pytest
import torch

from simple_brats.config import ExperimentConfig, ModelConfig
from simple_brats.models import EncoderStemPatchTeacher
from simple_brats.training import (
    build_matching_system,
    make_synthetic_matching_batch,
    optimizer_parameter_groups,
    run_synthetic_smoke,
    validate_matching_batch,
)


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(model=ModelConfig(width=24, depth=2, heads=3, mlp_ratio=2.0))


def test_online_patch_teacher_shares_trained_stem_and_is_patch_only() -> None:
    system = build_matching_system(_tiny_config())
    teacher = system.online_patch_view
    assert teacher.geometry_encoder is system.encoder.patch_stem.projection
    assert teacher.output_norm is not system.encoder.output_norm
    assert not teacher.output_norm.elementwise_affine
    assert list(inspect.signature(EncoderStemPatchTeacher.forward).parameters) == [
        "self",
        "patches",
    ]


def test_synthetic_batch_has_no_target_or_slab_leakage() -> None:
    config = _tiny_config()
    batch, geometry = make_synthetic_matching_batch(config, batch_size=2, positions=8)
    validate_matching_batch(batch, geometry=geometry)
    assert batch.source_patches.shape[:2] == (2, 96)
    assert batch.target_patches.shape[:2] == (2, 32)
    assert batch.source_patches.shape[-3:] == (8, 8, 8)
    assert batch.target_patches.shape[-3:] == (8, 8, 8)

    leaked = batch.source_modality_ids.clone()
    leaked[:, 0] = batch.query_modality_ids[:, 0]
    leaked_positions = batch.source_position_ids.clone()
    leaked_positions[:, 0] = batch.query_position_ids[:, 0]
    leaked_coordinates = batch.source_coordinates_mm.clone()
    leaked_coordinates[:, 0] = batch.query_coordinates_mm[:, 0]
    bad = type(batch)(
        **{
            **batch.__dict__,
            "source_modality_ids": leaked,
            "source_position_ids": leaked_positions,
            "source_coordinates_mm": leaked_coordinates,
        }
    )
    with pytest.raises(ValueError, match="visible at its target position"):
        validate_matching_batch(bad, geometry=geometry)


def test_one_tiny_training_step_is_finite_and_updates_ema() -> None:
    metrics = run_synthetic_smoke(
        _tiny_config(),
        device=torch.device("cpu"),
        batch_size=2,
        positions=8,
    )
    assert metrics["status"] == "ok"
    assert metrics["gradient_norm"] > 0
    assert metrics["ema_update_norm"] > 0
    assert metrics["teacher_updates"] == 1
    assert metrics["target_effective_rank"] > 1


def test_full_system_is_invariant_to_independent_target_table_permutation() -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=2, positions=8)
    permutation = torch.randperm(32, generator=torch.Generator().manual_seed(11))
    permuted = replace(
        batch,
        target_patches=batch.target_patches[:, permutation],
        target_modality_ids=batch.target_modality_ids[:, permutation],
        target_position_ids=batch.target_position_ids[:, permutation],
        target_coordinates_mm=batch.target_coordinates_mm[:, permutation],
        target_bag_ids=batch.target_bag_ids[:, permutation],
        target_pair_ids=batch.target_pair_ids[:, permutation],
    )

    reference_system = build_matching_system(config).eval()
    permuted_system = copy.deepcopy(reference_system)
    reference = reference_system(batch)
    independently_permuted = permuted_system(permuted)
    torch.testing.assert_close(independently_permuted.loss, reference.loss)
    torch.testing.assert_close(independently_permuted.predictions, reference.predictions)

    reference.loss.backward()
    independently_permuted.loss.backward()
    reference_gradient = reference_system.encoder.patch_stem.projection.weight.grad
    permuted_gradient = permuted_system.encoder.patch_stem.projection.weight.grad
    torch.testing.assert_close(permuted_gradient, reference_gradient)


def test_unimplemented_objective_does_not_silently_run_matching() -> None:
    config = _tiny_config()
    config = type(config)(
        **{
            **config.__dict__,
            "task": type(config.task)(**{**config.task.__dict__, "objective": "mae"}),
        }
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_matching_system(config)


def test_optimizer_excludes_norm_bias_and_embeddings_from_weight_decay() -> None:
    system = build_matching_system(_tiny_config())
    groups = optimizer_parameter_groups(system, weight_decay=0.05)
    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    not_decayed = {id(parameter) for parameter in groups[1]["params"]}
    assert decayed.isdisjoint(not_decayed)
    for name, parameter in system.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or "embedding" in name or parameter.ndim < 2:
            assert id(parameter) in not_decayed
