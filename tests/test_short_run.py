from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.splits import (
    SplitFraction,
    SplitManifest,
    SubjectAssignment,
)
from simple_brats.short_run import (
    _MetricsLogger,
    _ordered_train_cases,
    assignment_for_step,
    run_classification,
)
from simple_brats.training import (
    PREDICTION_DIAGNOSTIC_STREAM,
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
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
        PREDICTION_DIAGNOSTIC_STREAM,
    }
    assert record["batch"]["plan_sha256"] == "a" * 64
