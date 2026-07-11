from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import simple_brats.short_run as short_run_module
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.splits import (
    SplitFraction,
    SplitManifest,
    SubjectAssignment,
)
from simple_brats.short_run import (
    DeterministicRealBatchFactory,
    ShortRunError,
    _build_fixed_target_probe,
    _held_out_probe_cases,
    _MetricsLogger,
    _ordered_train_cases,
    _wandb_for_schedule,
    assignment_for_step,
    run_classification,
)
from simple_brats.training import (
    PREDICTION_DIAGNOSTIC_STREAM,
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
    TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
    RepresentationStats,
    StepMetrics,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(index: int) -> CaseRecord:
    case_id = f"BraTS-MET-{index:05d}-000"
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=tuple(
            FileRecord(
                modality=modality,
                path=f"{case_id}/{case_id}-{modality}.nii.gz",
                sha256=_digest(f"{case_id}-{modality}"),
            )
            for modality in ("t1n", "t1c", "t2w", "t2f")
        ),
    )


def test_assignment_uses_consecutive_case_blocks_and_absolute_epochs() -> None:
    observed = [
        assignment_for_step(step, case_count=4, bags_per_case=25)
        for step in (0, 24, 25, 49, 75, 99, 100, 124)
    ]
    assert [(item.case_index, item.epoch, item.bag_index) for item in observed] == [
        (0, 0, 0),
        (0, 0, 24),
        (1, 0, 0),
        (1, 0, 24),
        (3, 0, 0),
        (3, 0, 24),
        (0, 1, 0),
        (0, 1, 24),
    ]


def test_run_classification_tracks_checkpoint_availability() -> None:
    assert (
        run_classification(total_steps=100, checkpoint_every_steps=1_000)
        == "optimization_stability_diagnostic_not_representation_result"
    )
    assert (
        run_classification(total_steps=1_000, checkpoint_every_steps=1_000)
        == "checkpointed_representation_pretraining"
    )


def test_wandb_is_optional_only_below_artifact_cadence(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def without_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("simulated unavailable tracking extra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_wandb)
    assert _wandb_for_schedule(total_steps=100, artifact_every_steps=5_000) is None
    with pytest.raises(ShortRunError, match="artifact cadence"):
        _wandb_for_schedule(total_steps=5_000, artifact_every_steps=5_000)


def test_case_selection_uses_only_train_partition_and_is_seed_deterministic() -> None:
    cases = tuple(_case(index) for index in range(1, 7))
    manifest = DatasetManifest(cases=cases)
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=0,
        fractions=(SplitFraction("train", "0.5"), SplitFraction("test", "0.5")),
        assignments=tuple(
            SubjectAssignment(case.subject_id, "train" if index < 4 else "test")
            for index, case in enumerate(cases)
        ),
    )
    first = _ordered_train_cases(manifest, split, seed=11, max_cases=3)
    second = _ordered_train_cases(manifest, split, seed=11, max_cases=3)
    assert first == second
    assert len(first) == 3
    assert all(split.split_of(case.subject_id) == "train" for case in first)


def test_probe_cases_are_train_partition_and_subject_disjoint() -> None:
    cases = tuple(_case(index) for index in range(1, 11))
    manifest = DatasetManifest(cases=cases)
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=0,
        fractions=(SplitFraction("train", "0.8"), SplitFraction("test", "0.2")),
        assignments=tuple(
            SubjectAssignment(case.subject_id, "train" if index < 8 else "test")
            for index, case in enumerate(cases)
        ),
    )
    optimization = _ordered_train_cases(manifest, split, seed=11, max_cases=4)
    probe = _held_out_probe_cases(
        manifest,
        split,
        seed=11,
        optimization_cases=optimization,
        case_count=4,
    )

    assert len(probe) == 4
    assert {case.subject_id for case in probe}.isdisjoint(
        {case.subject_id for case in optimization}
    )
    assert len({case.subject_id for case in probe}) == 4
    assert all(split.split_of(case.subject_id) == "train" for case in probe)


def test_metrics_jsonl_records_both_streams_and_batch_plan(tmp_path: Path) -> None:
    factory = SimpleNamespace(
        last_record={
            "completed_step": 1,
            "case_id": "BraTS-MET-00001-000",
            "plan_sha256": "a" * 64,
        }
    )
    stats = RepresentationStats(
        count=8,
        variance=0.5,
        effective_rank=4.0,
        off_diagonal_cosine=0.1,
    )
    logger = _MetricsLogger(tmp_path / "metrics.jsonl", factory, None)  # type: ignore[arg-type]
    logger(
        StepMetrics(
            step=1,
            loss=1.2,
            accuracy=0.25,
            chance=0.125,
            ema_update_count=1,
            diagnostics_by_stream={
                TEACHER_TARGET_DIAGNOSTIC_STREAM: {0: stats},
                TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM: {0: stats},
                PREDICTION_DIAGNOSTIC_STREAM: {0: stats},
            },
        )
    )
    logger.close()

    rows = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert set(record["diagnostics_by_stream"]) == {
        TEACHER_TARGET_DIAGNOSTIC_STREAM,
        TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
        PREDICTION_DIAGNOSTIC_STREAM,
    }
    assert record["batch"]["plan_sha256"] == "a" * 64


def test_real_batch_factory_reuses_and_bounds_prepared_case_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = tuple(_case(index) for index in range(1, 6))
    extractor_specs: list[str] = []
    universe_cases: list[str] = []

    def fake_extractor(**kwargs: object) -> SimpleNamespace:
        spec = str(kwargs["extraction_spec"])
        extractor_specs.append(spec)
        return SimpleNamespace(extraction_spec_sha256=_digest(spec))

    def fake_universe(
        extractor: SimpleNamespace,
        case: CaseRecord,
        *,
        geometry: object,
    ) -> SimpleNamespace:
        del extractor, geometry
        universe_cases.append(case.case_id)
        return SimpleNamespace(case=case)

    monkeypatch.setattr(short_run_module, "CachedNiftiPatchExtractor", fake_extractor)
    monkeypatch.setattr(short_run_module, "prepare_case_candidate_universe", fake_universe)
    config = SimpleNamespace(
        patch=SimpleNamespace(
            footprint_mm=4.0,
            thin_mm=4.0,
            tensor_shape=(16, 16, 16),
        )
    )
    case_grids = SimpleNamespace(
        extraction_spec_for_case=lambda case, patch_config: (  # noqa: ARG005
            f"spec-{case.case_id}"
        )
    )
    factory = DeterministicRealBatchFactory(
        data_root=tmp_path,
        manifest=SimpleNamespace(sha256=_digest("manifest")),  # type: ignore[arg-type]
        case_grids=case_grids,  # type: ignore[arg-type]
        cases=cases,
        config=config,  # type: ignore[arg-type]
        plans_dir=tmp_path,
        bags_per_case=25,
        candidate_pool_size=512,
        max_plan_attempts=8,
    )

    first = factory._activate(0)
    assert factory._activate(0) is first
    for case_index in range(1, 5):
        factory._activate(case_index)
    reloaded = factory._activate(0)

    assert reloaded is not first
    assert universe_cases == [
        cases[0].case_id,
        cases[1].case_id,
        cases[2].case_id,
        cases[3].case_id,
        cases[4].case_id,
        cases[0].case_id,
    ]
    assert extractor_specs == [f"spec-{case_id}" for case_id in universe_cases]
    assert len(factory._case_cache) == 4


def test_fixed_probe_uses_four_cases_two_bags_and_64_samples_per_modality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = tuple(_case(index) for index in range(1, 5))

    class FakeFactory:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["cases"] == cases
            assert kwargs["bags_per_case"] == 2
            self.last_record: dict[str, object] | None = None

        def __call__(self, index: int) -> SimpleNamespace:
            modality_ids = torch.arange(4).repeat_interleave(8).reshape(1, 32)
            patches = torch.full((1, 32, 2, 2, 2), float(index), dtype=torch.float32)
            self.last_record = {
                "case_id": cases[index // 2].case_id,
                "plan_sha256": _digest(str(index)),
            }
            return SimpleNamespace(
                target_patches=patches,
                target_modality_ids=modality_ids,
            )

    monkeypatch.setattr(short_run_module, "DeterministicRealBatchFactory", FakeFactory)
    config = SimpleNamespace(
        task=SimpleNamespace(
            positions_per_bag=32,
            modalities=("t1n", "t1c", "t2w", "t2f"),
        )
    )
    built = _build_fixed_target_probe(
        data_root=tmp_path,
        manifest=SimpleNamespace(),  # type: ignore[arg-type]
        case_grids=SimpleNamespace(),  # type: ignore[arg-type]
        cases=cases,
        config=config,  # type: ignore[arg-type]
        plans_dir=tmp_path,
        candidate_pool_size=512,
        max_plan_attempts=8,
    )

    assert built.bags_per_case == 2
    assert len(built.records) == 8
    assert built.probe.sample_count_by_modality == {0: 64, 1: 64, 2: 64, 3: 64}
    assert len(built.probe.sha256) == 64
