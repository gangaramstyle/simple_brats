from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from simple_brats.a40_throughput_smoke import (
    REPLAY_ABSOLUTE_STEP_INDEX,
    A40ThroughputSmokeError,
    batch_semantic_sha256,
    compare_absolute_batch_replay,
    compare_reference_and_optimized_batches,
    first_eight_subject_blocks,
    steady_throughput_report,
    validate_a40_identity,
    validate_compile_counters,
    validate_runtime_stats,
)
from simple_brats.config import ExperimentConfig
from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.long_run import SubjectBalancedSchedule
from simple_brats.training import make_synthetic_matching_batch


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(subject: int) -> CaseRecord:
    case_id = f"BraTS-MET-{subject:05d}-000"
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=tuple(
            FileRecord(
                modality,
                f"{case_id}/{case_id}-{modality}.nii.gz",
                _digest(f"{case_id}:{modality}"),
            )
            for modality in ("t1n", "t1c", "t2w", "t2f")
        ),
    )


def _plan_record() -> dict[str, object]:
    return {
        "absolute_step_index": 0,
        "completed_step": 1,
        "case_id": "BraTS-MET-00001-000",
        "subject_id": "BraTS-MET-00001",
        "visit_id": "000",
        "epoch": 0,
        "bag_index": 0,
        "extraction_spec_sha256": _digest("extraction"),
        "plan_sha256": _digest("plan"),
        "prepared_plan_sha256": _digest("prepared"),
        "candidate_centers_sha256": _digest("centers"),
        "candidate_count": 100,
        "schedule_sha256": _digest("schedule"),
        "subject_epoch": 0,
        "subject_position": 0,
        "visit_rotation_index": 0,
    }


def test_parity_accepts_locked_pixel_tolerance_and_exact_metadata() -> None:
    reference, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=1, positions=8)
    changed = reference.source_patches.clone()
    changed.reshape(-1)[0] += 1e-7
    optimized = replace(reference, source_patches=changed)

    report = compare_reference_and_optimized_batches(
        reference,
        optimized,
        reference_plan=_plan_record(),
        optimized_plan=_plan_record(),
    )

    assert report["plan_sha256"] == _digest("plan")
    assert report["pixels"]["source_patches"]["passed"] is True  # type: ignore[index]
    assert all(report["metadata_exact"].values())  # type: ignore[union-attr]


def test_parity_fails_on_metadata_or_pixel_drift() -> None:
    reference, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=1, positions=8)
    bad_metadata = replace(
        reference,
        query_bag_ids=reference.query_bag_ids + 1,
    )
    with pytest.raises(A40ThroughputSmokeError, match="metadata"):
        compare_reference_and_optimized_batches(
            reference,
            bad_metadata,
            reference_plan=_plan_record(),
            optimized_plan=_plan_record(),
        )
    bad_pixels = replace(reference, target_patches=reference.target_patches + 0.1)
    with pytest.raises(A40ThroughputSmokeError, match="tolerance"):
        compare_reference_and_optimized_batches(
            reference,
            bad_pixels,
            reference_plan=_plan_record(),
            optimized_plan=_plan_record(),
        )


def test_fresh_absolute_step_replay_requires_exact_plan_and_batch_digest() -> None:
    batch, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=1, positions=8)
    plan = _plan_record()
    plan["absolute_step_index"] = REPLAY_ABSOLUTE_STEP_INDEX
    plan["completed_step"] = REPLAY_ABSOLUTE_STEP_INDEX + 1
    first = compare_absolute_batch_replay(
        absolute_step_index=REPLAY_ABSOLUTE_STEP_INDEX,
        continuous_batch_sha256=batch_semantic_sha256(batch),
        replayed_batch=batch,
        continuous_plan=plan,
        replayed_plan=plan,
    )

    assert first["exact"] is True
    changed = dict(plan)
    changed["bag_index"] = 1
    with pytest.raises(A40ThroughputSmokeError, match="plan identity"):
        compare_absolute_batch_replay(
            absolute_step_index=REPLAY_ABSOLUTE_STEP_INDEX,
            continuous_batch_sha256=first["continuous_batch_sha256"],  # type: ignore[arg-type]
            replayed_batch=batch,
            continuous_plan=plan,
            replayed_plan=changed,
        )


def test_steady_window_is_steps_9_through_64_and_enforces_floor() -> None:
    timestamps = {step: step / 2.0 for step in range(1, 65)}

    report = steady_throughput_report(timestamps)

    assert report["window"] == {
        "first_step": 9,
        "last_step": 64,
        "measured_steps": 56,
        "excluded_completed_steps": 8,
    }
    assert report["steps_per_second"] == 2.0
    with pytest.raises(A40ThroughputSmokeError, match="below"):
        steady_throughput_report({step: float(step) for step in range(1, 65)})


def test_a40_identity_is_exact() -> None:
    assert validate_a40_identity(
        name="NVIDIA A40",
        capability=(8, 6),
        visible_device_count=1,
    )["compute_capability"] == [8, 6]
    with pytest.raises(A40ThroughputSmokeError, match="A40"):
        validate_a40_identity(name="NVIDIA A100", capability=(8, 0), visible_device_count=1)


def test_first_64_steps_are_eight_distinct_subject_blocks() -> None:
    schedule = SubjectBalancedSchedule(tuple(_case(index) for index in range(1, 10)), seed=0)

    blocks = first_eight_subject_blocks(schedule)

    assert len(blocks) == 8
    assert len({block["subject_id"] for block in blocks}) == 8
    assert all(block["bag_indices"] == list(range(8)) for block in blocks)


def test_runtime_stats_require_exact_8_block_cache_access() -> None:
    stats = {
        "case_prefetch": {
            "submitted_count": 16,
            "consumed_count": 8,
            "ready_hit_count": 7,
            "stall_count": 1,
            "readiness_barrier_count": 1,
            "readiness_barrier_key_count": 16,
        },
        "host_case_cache": {"miss_count": 8, "hit_count": 56},
        "gpu_case_cache": {
            "miss_count": 8,
            "hit_count": 56,
            "resident_bytes": 100,
            "peak_resident_bytes": 200,
            "byte_budget": 1_000,
        },
    }

    validate_runtime_stats(stats)

    broken = dict(stats)
    broken["host_case_cache"] = {"miss_count": 9, "hit_count": 55}
    with pytest.raises(A40ThroughputSmokeError, match="access pattern"):
        validate_runtime_stats(broken)

    missing_barrier = dict(stats)
    missing_barrier["case_prefetch"] = {
        **stats["case_prefetch"],  # type: ignore[dict-item]
        "readiness_barrier_count": 0,
    }
    with pytest.raises(A40ThroughputSmokeError, match="sixteen"):
        validate_runtime_stats(missing_barrier)


def test_compile_counters_require_dynamo_and_inductor_execution() -> None:
    validate_compile_counters(
        {
            "dynamo_calls_captured": 10,
            "dynamo_unique_graphs": 2,
            "inductor_fxgraph_cache_events": 1,
            "inductor_generated_kernel_count": 0,
        }
    )
    with pytest.raises(A40ThroughputSmokeError, match="Dynamo"):
        validate_compile_counters(
            {
                "dynamo_calls_captured": 0,
                "dynamo_unique_graphs": 0,
                "inductor_fxgraph_cache_events": 1,
                "inductor_generated_kernel_count": 0,
            }
        )
    with pytest.raises(A40ThroughputSmokeError, match="Inductor"):
        validate_compile_counters(
            {
                "dynamo_calls_captured": 1,
                "dynamo_unique_graphs": 1,
                "inductor_fxgraph_cache_events": 0,
                "inductor_generated_kernel_count": 0,
            }
        )
