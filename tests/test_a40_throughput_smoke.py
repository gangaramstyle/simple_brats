from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from simple_brats.a40_throughput_smoke import (
    FIRST_POST_STARTUP_REFILL_COMPLETED_STEP,
    REPLAY_ABSOLUTE_STEP_INDEX,
    TOTAL_STEPS,
    A40ThroughputSmokeError,
    _assert_config,
    _fixed_target_probe_from_single_d_cycle,
    batch_semantic_sha256,
    compare_absolute_batch_replay,
    compare_reference_and_optimized_batches,
    scheduled_subject_blocks,
    steady_throughput_report,
    tail_interval_diagnostics,
    validate_a40_identity,
    validate_closed_plan_persistence,
    validate_compile_counters,
    validate_runtime_stats,
)
from simple_brats.config import ExperimentConfig, load_experiment_config
from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.long_run import SubjectBalancedSchedule
from simple_brats.training import make_synthetic_matching_batch


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_throughput_gate_accepts_both_exact_registered_scale_arms() -> None:
    _assert_config(load_experiment_config("configs/v0_cross_matching_small.toml"))
    _assert_config(load_experiment_config("configs/v0_cross_matching_small_8mm.toml"))
    with pytest.raises(A40ThroughputSmokeError, match="registered"):
        _assert_config(load_experiment_config("configs/v0_cross_matching.toml"))


def test_fixed_probe_joins_one_complete_single_d_cycle() -> None:
    batch, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=4, positions=32)
    tables = [
        (batch.target_patches[index : index + 1], batch.target_modality_ids[index : index + 1])
        for index in range(4)
    ]

    probe = _fixed_target_probe_from_single_d_cycle(
        tables,
        expected_modalities=range(4),
    )

    assert probe.target_patches.shape[:2] == (1, 128)
    assert probe.sample_count_by_modality == {0: 32, 1: 32, 2: 32, 3: 32}
    with pytest.raises(A40ThroughputSmokeError, match="every expected modality"):
        _fixed_target_probe_from_single_d_cycle(
            [*tables[:3], tables[0]],
            expected_modalities=range(4),
        )


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
    reference, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=1, positions=32)
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
    reference, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=1, positions=32)
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
    batch, _ = make_synthetic_matching_batch(ExperimentConfig(), batch_size=1, positions=32)
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


def test_steady_window_is_steps_65_through_160_and_enforces_floor() -> None:
    timestamps = {step: step / 2.0 for step in range(1, TOTAL_STEPS + 1)}

    report = steady_throughput_report(timestamps)

    assert report["window"] == {
        "first_step": 65,
        "last_step": 160,
        "measured_steps": 96,
        "excluded_completed_steps": 64,
    }
    assert report["steps_per_second"] == 2.0
    with pytest.raises(A40ThroughputSmokeError, match="below"):
        steady_throughput_report({step: float(step) for step in range(1, TOTAL_STEPS + 1)})


def test_tail_diagnostics_report_slow_gpfs_interval_without_a_per_step_gate() -> None:
    elapsed = 0.0
    timestamps: dict[int, float] = {}
    for step in range(1, TOTAL_STEPS + 1):
        elapsed += 3.0 if step == 100 else 0.25
        timestamps[step] = elapsed

    report = steady_throughput_report(timestamps)
    tail = tail_interval_diagnostics(timestamps)

    assert report["passed"] is True
    assert tail["maximum_seconds"] == pytest.approx(3.0)
    assert tail["above_two_seconds_count"] == 1
    assert tail["slowest_intervals"][0] == {  # type: ignore[index]
        "completed_step": 100,
        "seconds": pytest.approx(3.0),
    }
    assert tail["individual_interval_failure_threshold"] is None


def test_a40_identity_is_exact() -> None:
    assert validate_a40_identity(
        name="NVIDIA A40",
        capability=(8, 6),
        visible_device_count=1,
    )["compute_capability"] == [8, 6]
    with pytest.raises(A40ThroughputSmokeError, match="A40"):
        validate_a40_identity(name="NVIDIA A100", capability=(8, 0), visible_device_count=1)


def test_first_160_steps_are_twenty_distinct_subject_blocks() -> None:
    schedule = SubjectBalancedSchedule(tuple(_case(index) for index in range(1, 22)), seed=0)

    blocks = scheduled_subject_blocks(schedule)

    assert len(blocks) == 20
    assert len({block["subject_id"] for block in blocks}) == 20
    assert all(block["bag_indices"] == list(range(8)) for block in blocks)
    assert blocks[17]["start_step"] == FIRST_POST_STARTUP_REFILL_COMPLETED_STEP


def test_runtime_stats_require_exact_20_block_cache_and_refill_access() -> None:
    stats = {
        "case_prefetch": {
            "submitted_count": 33,
            "consumed_count": 20,
            "synchronous_consumed_count": 1,
            "startup_consumed_count": 16,
            "refill_consumed_count": 3,
            "ready_hit_count": 19,
            "stall_count": 1,
            "discarded_count": 0,
            "readiness_barrier_count": 1,
            "readiness_barrier_key_count": 16,
            "pending_count": 13,
            "ready_pending_count": 9,
            "failed_pending_count": 0,
            "running_pending_count": 4,
            "ready_prefix_count": 5,
        },
        "host_case_cache": {"miss_count": 20, "hit_count": 140},
        "gpu_case_cache": {
            "miss_count": 20,
            "hit_count": 140,
            "resident_bytes": 100,
            "peak_resident_bytes": 200,
            "byte_budget": 1_000,
        },
        "plan_artifact_writer": {
            "mode": "single_worker_ordered_bounded_async_atomic_create",
            "queue_depth": 64,
            "pending_count": 20,
            "maximum_pending_count": 64,
            "submitted_count": 160,
            "completed_count": 140,
            "persistence_seconds": 10.0,
            "backpressure_seconds": 1.0,
        },
    }

    validate_runtime_stats(stats)

    broken = dict(stats)
    broken["host_case_cache"] = {"miss_count": 21, "hit_count": 139}
    with pytest.raises(A40ThroughputSmokeError, match="access pattern"):
        validate_runtime_stats(broken)

    missing_barrier = dict(stats)
    missing_barrier["case_prefetch"] = {
        **stats["case_prefetch"],  # type: ignore[dict-item]
        "readiness_barrier_count": 0,
    }
    with pytest.raises(A40ThroughputSmokeError, match="sixteen"):
        validate_runtime_stats(missing_barrier)

    wrong_consumption_source = dict(stats)
    wrong_consumption_source["case_prefetch"] = {
        **stats["case_prefetch"],  # type: ignore[dict-item]
        "startup_consumed_count": 17,
        "refill_consumed_count": 2,
    }
    with pytest.raises(A40ThroughputSmokeError, match="provenance"):
        validate_runtime_stats(wrong_consumption_source)

    incomplete_refill = dict(stats)
    incomplete_refill["case_prefetch"] = {
        **stats["case_prefetch"],  # type: ignore[dict-item]
        "ready_pending_count": 9,
        "running_pending_count": 5,
    }
    with pytest.raises(A40ThroughputSmokeError, match="runway"):
        validate_runtime_stats(incomplete_refill)

    failed_refill = dict(stats)
    failed_refill["case_prefetch"] = {
        **stats["case_prefetch"],  # type: ignore[dict-item]
        "ready_pending_count": 9,
        "failed_pending_count": 1,
        "running_pending_count": 3,
    }
    with pytest.raises(A40ThroughputSmokeError, match="runway"):
        validate_runtime_stats(failed_refill)

    short_ready_prefix = dict(stats)
    short_ready_prefix["case_prefetch"] = {
        **stats["case_prefetch"],  # type: ignore[dict-item]
        "ready_prefix_count": 4,
    }
    with pytest.raises(A40ThroughputSmokeError, match="runway"):
        validate_runtime_stats(short_ready_prefix)

    unbounded_writer = dict(stats)
    unbounded_writer["plan_artifact_writer"] = {
        **stats["plan_artifact_writer"],  # type: ignore[dict-item]
        "pending_count": 65,
    }
    with pytest.raises(A40ThroughputSmokeError, match="bounded"):
        validate_runtime_stats(unbounded_writer)


def test_closed_plan_persistence_requires_sustained_two_pairs_per_second() -> None:
    stats = {
        "plan_artifact_writer": {
            "pending_count": 0,
            "completed_count": 160,
            "persistence_seconds": 80.0,
        }
    }
    report = validate_closed_plan_persistence(stats)
    assert report["plan_pairs_per_second"] == 2.0

    stats["plan_artifact_writer"]["persistence_seconds"] = 80.1
    with pytest.raises(A40ThroughputSmokeError, match="below"):
        validate_closed_plan_persistence(stats)


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
