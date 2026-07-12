from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from simple_brats.config import ExperimentConfig, ModelConfig, PatchConfig
from simple_brats.data.case_grids import (
    CaseGridManifest,
    CaseGridRecord,
    ExtractionPolicy,
    SpatialGrid,
    derive_prepared_grid,
)
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.splits import (
    SplitFraction,
    SplitManifest,
    SubjectAssignment,
)
from simple_brats.evaluation.checkpoint import (
    CheckpointEvaluationError,
    ColocatedFourModalityTokenEncoder,
    SingletonOnlineTokenEncoder,
    build_random_online_encoder,
    configure_deterministic_evaluation_runtime,
    load_online_encoder_checkpoint,
)
from simple_brats.evaluation.patches import (
    BinaryPatchLabelRule,
    EvaluationPatchManifest,
    EvaluationPatchRecord,
    SegmentationAuditRecord,
)
from simple_brats.long_run import SubjectBalancedSchedule
from simple_brats.training.matching import build_matching_system

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(index: int) -> CaseRecord:
    case_id = f"BraTS-MET-{index:05d}-000"
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=(FileRecord("t1n", f"{case_id}/t1n.nii.gz", _digest(case_id)),),
    )


def _inputs():
    train, train_two, validation, test = (_case(index) for index in (1, 2, 3, 4))
    manifest = DatasetManifest(cases=(train, train_two, validation, test))
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=0,
        fractions=(
            SplitFraction("train", "0.5"),
            SplitFraction("validation", "0.25"),
            SplitFraction("test", "0.25"),
        ),
        assignments=(
            SubjectAssignment(train.subject_id, "train"),
            SubjectAssignment(train_two.subject_id, "train"),
            SubjectAssignment(validation.subject_id, "validation"),
            SubjectAssignment(test.subject_id, "test"),
        ),
    )
    policy = ExtractionPolicy()
    native = SpatialGrid(shape=(20, 20, 20), affine=IDENTITY)
    prepared = derive_prepared_grid(native, policy)
    records = tuple(
        CaseGridRecord(
            data_manifest_sha256=manifest.sha256,
            case=case,
            declared_spatial_units=("mm", "mm", "mm", "mm"),
            extraction_policy_sha256=policy.sha256,
            native_grid=native,
            modality_native_grids=(native, native, native, native),
            prepared_grid=prepared,
            extraction_spec_sha256=policy.extraction_spec(prepared).sha256,
        )
        for case in manifest.cases
    )
    grids = CaseGridManifest(
        data_manifest_sha256=manifest.sha256,
        policy=policy,
        records=records,
    )

    def patch(case: CaseRecord, partition: str, label: int, offset: float):
        return EvaluationPatchRecord.create(
            source=case.source,
            release=case.release,
            case_id=case.case_id,
            subject_id=case.subject_id,
            partition=partition,
            center_mm=(5.5 + offset, 5.5, 5.5),
            seg_positive_voxels=16 if label else 0,
            crop_voxels=64,
            halo_clear=not bool(label),
            label=label,
        )

    evaluation = EvaluationPatchManifest(
        data_manifest_sha256=manifest.sha256,
        subject_split_sha256=split.sha256,
        case_grid_manifest_sha256=grids.sha256,
        segmentation_label_audit_sha256=_digest("label-audit"),
        patch_config=PatchConfig(),
        seed=0,
        label_rule=BinaryPatchLabelRule(),
        probe_train_subjects=(train.subject_id,),
        validation_subjects=(validation.subject_id,),
        ineligible_probe_train_subjects=(),
        locked_test_subject_count=1,
        segmentation_audit=(
            SegmentationAuditRecord(train.case_id, _digest("train-seg"), (1, 2, 3)),
            SegmentationAuditRecord(validation.case_id, _digest("val-seg"), (1, 2, 3)),
        ),
        records=(
            patch(train, "probe_train", 0, 0.0),
            patch(train, "probe_train", 1, 1.0),
            patch(validation, "validation", 0, 0.0),
            patch(validation, "validation", 1, 1.0),
        ),
    )
    config = ExperimentConfig(
        patch=PatchConfig(),
        model=ModelConfig(width=16, depth=1, heads=2, mlp_ratio=2.0),
    )
    return config, manifest, split, grids, evaluation, (train, train_two)


def test_loads_runner_v3_online_encoder_and_singleton_api(tmp_path: Path) -> None:
    config, manifest, split, grids, evaluation, train_cases = _inputs()
    system = build_matching_system(config)
    provenance = {
        "manifest_sha256": manifest.sha256,
        "split_sha256": split.sha256,
        "case_grid_manifest_sha256": grids.sha256,
        "config_sha256": config.sha256,
        "selected_train_subject_ids": [case.subject_id for case in train_cases],
    }
    checkpoint = tmp_path / "step.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 1000,
            "metadata": provenance,
            "state": {
                "runner_schema_version": 3,
                "step": 1000,
                "provenance": provenance,
                "model": system.state_dict(),
            },
        },
        checkpoint,
    )

    loaded = load_online_encoder_checkpoint(
        checkpoint,
        config=config,
        manifest=manifest,
        split=split,
        case_grids=grids,
        evaluation_patches=evaluation,
        device="cpu",
    )

    assert loaded.step == 1000
    assert loaded.deterministic_runtime["torch_deterministic_algorithms"] is True
    assert loaded.deterministic_runtime["cuda_matmul_allow_tf32"] is False
    assert list(inspect.signature(SingletonOnlineTokenEncoder.forward).parameters) == [
        "self",
        "patches",
        "modality_ids",
    ]
    assert not any(parameter.requires_grad for parameter in loaded.encoder.parameters())
    patches = torch.randn(2, 8, 8, 8)
    output = loaded.encoder(patches, torch.tensor([0, 1]))
    changed = patches.clone()
    changed[1] += 100
    changed_output = loaded.encoder(changed, torch.tensor([0, 1]))
    assert output.shape == (2, config.model.width)
    torch.testing.assert_close(output[0], changed_output[0])


def test_cuda_determinism_requires_pre_python_workspace_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(CheckpointEvaluationError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_deterministic_evaluation_runtime(torch.device("cuda"))


def test_random_control_is_architecture_matched_and_seeded() -> None:
    config, *_ = _inputs()
    first = build_random_online_encoder(config, seed=19, device="cpu")
    second = build_random_online_encoder(config, seed=19, device="cpu")
    patches = torch.randn(3, 8, 8, 8)
    modalities = torch.tensor([0, 1, 2])

    torch.testing.assert_close(first(patches, modalities), second(patches, modalities))


def test_evaluation_patch_geometry_must_match_checkpoint_config(tmp_path: Path) -> None:
    config, manifest, split, grids, evaluation, _ = _inputs()
    mismatched = replace(
        config,
        patch=PatchConfig(
            footprint_mm=4.0,
            thin_mm=4.0,
            tensor_shape=(16, 16, 16),
        ),
    )

    with pytest.raises(CheckpointEvaluationError, match="geometry does not exactly match"):
        load_online_encoder_checkpoint(
            tmp_path / "not-read.pt",
            config=mismatched,
            manifest=manifest,
            split=split,
            case_grids=grids,
            evaluation_patches=evaluation,
            device="cpu",
        )


def test_partial_train_checkpoint_requires_explicit_mechanics_override(tmp_path: Path) -> None:
    config, manifest, split, grids, evaluation, train_cases = _inputs()
    system = build_matching_system(config)
    provenance = {
        "manifest_sha256": manifest.sha256,
        "split_sha256": split.sha256,
        "case_grid_manifest_sha256": grids.sha256,
        "config_sha256": config.sha256,
        "selected_subject_ids": [train_cases[0].subject_id],
    }
    checkpoint = tmp_path / "partial.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 1000,
            "metadata": provenance,
            "state": {
                "runner_schema_version": 3,
                "step": 1000,
                "provenance": provenance,
                "model": system.state_dict(),
            },
        },
        checkpoint,
    )
    with pytest.raises(CheckpointEvaluationError, match="requires all SSL-train"):
        load_online_encoder_checkpoint(
            checkpoint,
            config=config,
            manifest=manifest,
            split=split,
            case_grids=grids,
            evaluation_patches=evaluation,
            device="cpu",
        )
    loaded = load_online_encoder_checkpoint(
        checkpoint,
        config=config,
        manifest=manifest,
        split=split,
        case_grids=grids,
        evaluation_patches=evaluation,
        device="cpu",
        require_all_ssl_train_subjects=False,
    )
    assert loaded.consumed_ssl_train_subject_count == 1
    assert loaded.total_ssl_train_subject_count == 2
    assert not loaded.complete_ssl_train_subject_coverage


def test_long_checkpoint_uses_actual_step_prefix_not_static_full_cohort(
    tmp_path: Path,
) -> None:
    config, manifest, split, grids, evaluation, train_cases = _inputs()
    system = build_matching_system(config)
    schedule = SubjectBalancedSchedule(train_cases, seed=config.seed, bags_per_subject=8)
    provenance = {
        "schema": "simple-brats.long-real-matching",
        "manifest_sha256": manifest.sha256,
        "split_sha256": split.sha256,
        "case_grid_manifest_sha256": grids.sha256,
        "config_sha256": config.sha256,
        "selected_train_subject_ids": list(schedule.subject_ids),
        "schedule": {
            "bags_per_subject": 8,
            "subject_schedule_sha256": schedule.sha256,
        },
    }

    def save(step: int) -> Path:
        path = tmp_path / f"step-{step:09d}.pt"
        torch.save(
            {
                "schema_version": 1,
                "step": step,
                "metadata": provenance,
                "state": {
                    "runner_schema_version": 3,
                    "step": step,
                    "provenance": provenance,
                    "model": system.state_dict(),
                },
            },
            path,
        )
        return path

    first_block = save(8)
    with pytest.raises(CheckpointEvaluationError, match="requires all SSL-train"):
        load_online_encoder_checkpoint(
            first_block,
            config=config,
            manifest=manifest,
            split=split,
            case_grids=grids,
            evaluation_patches=evaluation,
            device="cpu",
        )
    partial = load_online_encoder_checkpoint(
        first_block,
        config=config,
        manifest=manifest,
        split=split,
        case_grids=grids,
        evaluation_patches=evaluation,
        device="cpu",
        require_all_ssl_train_subjects=False,
    )
    assert partial.consumed_ssl_train_subject_count == 1
    assert not partial.complete_ssl_train_subject_coverage

    complete = load_online_encoder_checkpoint(
        save(16),
        config=config,
        manifest=manifest,
        split=split,
        case_grids=grids,
        evaluation_patches=evaluation,
        device="cpu",
    )
    assert complete.consumed_ssl_train_subject_count == 2
    assert complete.complete_ssl_train_subject_coverage


def test_colocated_joint_view_has_patch_only_api_and_fixed_concat() -> None:
    config, *_ = _inputs()
    singleton = build_random_online_encoder(config, seed=3, device="cpu")
    joint = ColocatedFourModalityTokenEncoder(singleton.encoder)
    assert list(inspect.signature(ColocatedFourModalityTokenEncoder.forward).parameters) == [
        "self",
        "patches",
    ]
    patches = torch.randn(2, 4, 8, 8, 8)
    output = joint(patches)
    changed = patches.clone()
    changed[1] += 10
    changed_output = joint(changed)

    assert output.shape == (2, 4 * config.model.width)
    torch.testing.assert_close(output[0], changed_output[0])
