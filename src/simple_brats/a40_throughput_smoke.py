"""Fail-closed real-data parity and throughput gate for the optimized A40 runtime."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.config import ExperimentConfig, load_experiment_config
from simple_brats.data.case_grids import load_case_grid_manifest
from simple_brats.data.manifest import canonical_json_bytes, load_manifest, sha256_file
from simple_brats.data.scheduled_cache import OptimizedRuntimeConfig
from simple_brats.data.splits import cases_for_splits, load_split, validate_split
from simple_brats.long_run import (
    SubjectBalancedBatchFactory,
    SubjectBalancedSchedule,
    configure_exact_resume_runtime,
)
from simple_brats.provenance import verify_git_sha
from simple_brats.training import (
    CheckpointManager,
    CheckpointPolicy,
    CollapseThresholds,
    FixedTargetPatchProbe,
    MatchingBatch,
    StepMetrics,
    apply_model_runtime,
    build_adamw_optimizer,
    build_matching_system,
    configure_training_runtime,
    run_matching_training,
    stats_by_modality,
)

CONFIG_SHA256_BY_ARM = {
    "32mm-prism_4mm-cube": "1ee8f45f2938c1d005fa975f20f3dcbeb8e378aada19b01b7d0dcc9fb28d847c",
    "64mm-prism_8mm-cube": "fdc89047dd0739c0108d077a9f9b38b611af8b241774a2f1a6bfb9c3aca568eb",
}
CONFIG_SHA256 = CONFIG_SHA256_BY_ARM["32mm-prism_4mm-cube"]
SCHEDULE_SHA256 = "4797321042581e25984038abc0ccb57dfe8859598f777502c96f02612c970912"
EXPECTED_TRAIN_CASES = 1_044
EXPECTED_TRAIN_SUBJECTS = 643
TOTAL_STEPS = 160
BAGS_PER_SUBJECT = 8
STEADY_START_STEP = 65
MINIMUM_STEPS_PER_SECOND = 2.0
STARTUP_PREFETCH_KEY_COUNT = 16
FIRST_POST_STARTUP_REFILL_COMPLETED_STEP = 137
PARITY_ATOL = 2e-6
PARITY_RTOL = 1e-5
REPLAY_ABSOLUTE_STEP_INDEX = 32
_LEARNING_RATE = 1e-4
_WEIGHT_DECAY = 0.05
_GRADIENT_CLIP_NORM = 10.0
_COLLAPSE_THRESHOLDS = CollapseThresholds(
    minimum_variance_ratio=0.10,
    minimum_effective_rank_ratio=0.25,
    maximum_off_diagonal_cosine=0.95,
)
_PLAN_IDENTITY_KEYS = (
    "absolute_step_index",
    "completed_step",
    "case_id",
    "subject_id",
    "visit_id",
    "epoch",
    "bag_index",
    "extraction_spec_sha256",
    "plan_sha256",
    "prepared_plan_sha256",
    "candidate_centers_sha256",
    "candidate_count",
    "schedule_sha256",
    "subject_epoch",
    "subject_position",
    "visit_rotation_index",
)


class A40ThroughputSmokeError(RuntimeError):
    """The optimized runtime failed parity, provenance, or throughput."""


def _fixed_target_probe_from_single_d_cycle(
    target_tables: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    expected_modalities: Sequence[int],
) -> FixedTargetPatchProbe:
    """Join one singleton-D bag per modality into an untimed fixed probe."""

    expected = tuple(int(modality_id) for modality_id in expected_modalities)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected probe modalities must be non-empty and unique")
    if len(target_tables) != len(expected):
        raise A40ThroughputSmokeError(
            "fixed-probe cycle must contain exactly one bag per expected modality"
        )
    patch_tables: list[torch.Tensor] = []
    modality_tables: list[torch.Tensor] = []
    observed: list[int] = []
    for patches, modality_ids in target_tables:
        if modality_ids.ndim != 2 or patches.ndim != modality_ids.ndim + 3:
            raise A40ThroughputSmokeError("fixed-probe tables have invalid ranks")
        if patches.shape[:2] != modality_ids.shape or patches.shape[0] != 1:
            raise A40ThroughputSmokeError(
                "each fixed-probe cycle entry must be one aligned singleton bag"
            )
        unique = modality_ids.detach().reshape(-1).unique().to(device="cpu")
        if unique.numel() != 1:
            raise A40ThroughputSmokeError(
                "each fixed-probe cycle bag must contain one target modality D"
            )
        observed.append(int(unique.item()))
        patch_tables.append(patches.detach().to(device="cpu"))
        modality_tables.append(modality_ids.detach().to(device="cpu"))
    if sorted(observed) != sorted(expected):
        raise A40ThroughputSmokeError(
            "fixed-probe cycle must cover every expected modality exactly once"
        )
    return FixedTargetPatchProbe(
        torch.cat(patch_tables, dim=1),
        torch.cat(modality_tables, dim=1),
    )


def _resolve_file(path: str | os.PathLike[str], description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise A40ThroughputSmokeError(f"{description} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise A40ThroughputSmokeError(f"{description} is unavailable: {path}") from error
    if not resolved.is_file():
        raise A40ThroughputSmokeError(f"{description} must be a regular file")
    return resolved


def _write_result(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    atomic_create_bytes(path, payload)
    if path.read_bytes() != payload:
        raise A40ThroughputSmokeError("throughput result changed after atomic publication")
    return hashlib.sha256(payload).hexdigest()


def _assert_config(config: ExperimentConfig) -> None:
    arm = config.registered_single_d_arm
    if arm is None or config.sha256 != CONFIG_SHA256_BY_ARM.get(arm):
        raise A40ThroughputSmokeError("smoke requires an exact registered single-D scale arm")
    observed = {
        "footprint_mm": config.patch.footprint_mm,
        "thin_mm": config.patch.thin_mm,
        "tensor_shape": config.patch.tensor_shape,
        "width": config.model.width,
        "depth": config.model.depth,
        "heads": config.model.heads,
        "prism_extent_mm": config.task.prism_extent_mm,
        "target_patches_per_bag": config.task.target_patches_per_bag,
        "context_patches_per_nontarget_modality": (
            config.task.context_patches_per_nontarget_modality
        ),
        "context_patches_target_modality": config.task.context_patches_target_modality,
        "source_patches_per_bag": config.task.source_patches_per_bag,
        "modalities": config.task.modalities,
        "objective": config.task.objective,
        "target_elsewhere": config.task.allow_target_modality_elsewhere,
        "target_at_target": config.task.allow_target_modality_at_target,
        "teacher_statistics": config.task.pass_scan_statistics_to_teacher,
    }
    expected = {
        "footprint_mm": config.patch.footprint_mm,
        "thin_mm": config.patch.footprint_mm,
        "tensor_shape": (16, 16, 16),
        "width": 256,
        "depth": 8,
        "heads": 4,
        "prism_extent_mm": config.task.prism_extent_mm,
        "target_patches_per_bag": 32,
        "context_patches_per_nontarget_modality": 30,
        "context_patches_target_modality": 6,
        "source_patches_per_bag": 96,
        "modalities": ("t1n", "t1c", "t2w", "t2f"),
        "objective": "match",
        "target_elsewhere": True,
        "target_at_target": False,
        "teacher_statistics": False,
    }
    if observed != expected:
        raise A40ThroughputSmokeError(f"registered scientific contract drifted for {arm}")


def validate_a40_identity(
    *,
    name: str,
    capability: tuple[int, int],
    visible_device_count: int,
) -> dict[str, object]:
    if not isinstance(name, str) or "A40" not in name.upper():
        raise A40ThroughputSmokeError(f"expected NVIDIA A40, observed {name!r}")
    if capability != (8, 6):
        raise A40ThroughputSmokeError(f"expected A40 compute capability 8.6, observed {capability}")
    if visible_device_count != 1:
        raise A40ThroughputSmokeError(
            f"smoke requires exactly one visible A40, observed {visible_device_count}"
        )
    return {
        "name": name,
        "compute_capability": list(capability),
        "visible_device_count": visible_device_count,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def batch_semantic_sha256(batch: MatchingBatch) -> str:
    if not isinstance(batch, MatchingBatch):
        raise TypeError("batch must be MatchingBatch")
    digest = hashlib.sha256(b"simple-brats.matching-batch-semantic-v1\0")
    for field in fields(MatchingBatch):
        digest.update(field.name.encode())
        digest.update(b"\0")
        value = getattr(batch, field.name)
        if value is None:
            digest.update(b"none\0")
        elif isinstance(value, torch.Tensor):
            digest.update(_tensor_sha256(value).encode())
            digest.update(b"\0")
        else:  # pragma: no cover - MatchingBatch currently has only Tensor | None fields
            raise A40ThroughputSmokeError(f"unsupported batch field {field.name}")
    return digest.hexdigest()


def compare_reference_and_optimized_batches(
    reference: MatchingBatch,
    optimized: MatchingBatch,
    *,
    reference_plan: Mapping[str, object],
    optimized_plan: Mapping[str, object],
    atol: float = PARITY_ATOL,
    rtol: float = PARITY_RTOL,
) -> dict[str, object]:
    if not isinstance(reference, MatchingBatch) or not isinstance(optimized, MatchingBatch):
        raise TypeError("parity requires two MatchingBatch values")
    for value, name in ((atol, "atol"), (rtol, "rtol")):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if any(reference_plan.get(key) != optimized_plan.get(key) for key in _PLAN_IDENTITY_KEYS):
        raise A40ThroughputSmokeError("reference and optimized plans are not identical")

    pixel_fields = {"source_patches", "target_patches"}
    metadata_checks: dict[str, bool] = {}
    pixel_checks: dict[str, object] = {}
    for field in fields(MatchingBatch):
        reference_value = getattr(reference, field.name)
        optimized_value = getattr(optimized, field.name)
        if field.name == "source_padding_mask":
            passed = reference_value is None and optimized_value is None
            metadata_checks[field.name] = passed
            if not passed:
                raise A40ThroughputSmokeError("optimized parity changed source padding")
            continue
        if not isinstance(reference_value, torch.Tensor) or not isinstance(
            optimized_value, torch.Tensor
        ):
            raise A40ThroughputSmokeError(f"batch field {field.name} is not a tensor")
        optimized_cpu = optimized_value.detach().to(device="cpu")
        reference_cpu = reference_value.detach().to(device="cpu")
        if field.name not in pixel_fields:
            passed = torch.equal(reference_cpu, optimized_cpu)
            metadata_checks[field.name] = passed
            if not passed:
                raise A40ThroughputSmokeError(
                    f"optimized parity changed metadata field {field.name}"
                )
            continue
        difference = (reference_cpu.float() - optimized_cpu.float()).abs()
        passed = torch.allclose(reference_cpu, optimized_cpu, atol=atol, rtol=rtol)
        pixel_checks[field.name] = {
            "passed": passed,
            "shape": list(reference_cpu.shape),
            "reference_sha256": _tensor_sha256(reference_cpu),
            "optimized_sha256": _tensor_sha256(optimized_cpu),
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        }
        if not passed:
            raise A40ThroughputSmokeError(
                f"optimized pixels exceeded parity tolerance for {field.name}"
            )
    return {
        "schema": "simple-brats.a40-throughput-parity",
        "schema_version": 1,
        "atol": atol,
        "rtol": rtol,
        "plan_sha256": reference_plan["plan_sha256"],
        "prepared_plan_sha256": reference_plan["prepared_plan_sha256"],
        "metadata_exact": metadata_checks,
        "pixels": pixel_checks,
    }


def compare_absolute_batch_replay(
    *,
    absolute_step_index: int,
    continuous_batch_sha256: str,
    replayed_batch: MatchingBatch,
    continuous_plan: Mapping[str, object],
    replayed_plan: Mapping[str, object],
) -> dict[str, object]:
    if absolute_step_index != REPLAY_ABSOLUTE_STEP_INDEX:
        raise A40ThroughputSmokeError("fresh replay must use the registered absolute step 32")
    if continuous_plan.get("absolute_step_index") != absolute_step_index or any(
        continuous_plan.get(key) != replayed_plan.get(key) for key in _PLAN_IDENTITY_KEYS
    ):
        raise A40ThroughputSmokeError("fresh replay plan identity differs from the continuous run")
    replayed_sha256 = batch_semantic_sha256(replayed_batch)
    if replayed_sha256 != continuous_batch_sha256:
        raise A40ThroughputSmokeError("fresh replay batch bytes differ from the continuous run")
    return {
        "absolute_step_index": absolute_step_index,
        "completed_step": absolute_step_index + 1,
        "plan_sha256": continuous_plan["plan_sha256"],
        "prepared_plan_sha256": continuous_plan["prepared_plan_sha256"],
        "continuous_batch_sha256": continuous_batch_sha256,
        "fresh_replayed_batch_sha256": replayed_sha256,
        "exact": True,
    }


def scheduled_subject_blocks(schedule: SubjectBalancedSchedule) -> list[dict[str, object]]:
    if not isinstance(schedule, SubjectBalancedSchedule):
        raise TypeError("schedule must be SubjectBalancedSchedule")
    records: list[dict[str, object]] = []
    subject_ids: list[str] = []
    for block_index in range(TOTAL_STEPS // BAGS_PER_SUBJECT):
        assignments = tuple(
            schedule.assignment_for_step(block_index * BAGS_PER_SUBJECT + offset)
            for offset in range(BAGS_PER_SUBJECT)
        )
        if (
            len({item.subject_id for item in assignments}) != 1
            or len({item.case_id for item in assignments}) != 1
            or tuple(item.bag_index for item in assignments) != tuple(range(BAGS_PER_SUBJECT))
        ):
            raise A40ThroughputSmokeError("160-step schedule is not twenty exact subject blocks")
        subject_ids.append(assignments[0].subject_id)
        records.append(
            {
                "block_index": block_index,
                "start_step": block_index * BAGS_PER_SUBJECT + 1,
                "end_step": (block_index + 1) * BAGS_PER_SUBJECT,
                "subject_id": assignments[0].subject_id,
                "case_id": assignments[0].case_id,
                "visit_id": assignments[0].visit_id,
                "subject_epoch": assignments[0].subject_epoch,
                "case_index": assignments[0].case_index,
                "bag_indices": list(range(BAGS_PER_SUBJECT)),
            }
        )
    if len(set(subject_ids)) != len(subject_ids):
        raise A40ThroughputSmokeError("first twenty schedule blocks must use distinct subjects")
    return records


def _validated_throughput_timestamps(
    completed_step_seconds: Mapping[int, float],
) -> list[float]:
    expected = set(range(1, TOTAL_STEPS + 1))
    if set(completed_step_seconds) != expected:
        raise A40ThroughputSmokeError(
            "throughput timestamps must exactly cover steps 1 through 160"
        )
    values = [float(completed_step_seconds[step]) for step in range(1, TOTAL_STEPS + 1)]
    if any(not math.isfinite(value) or value <= 0 for value in values) or any(
        second <= first for first, second in zip(values[:-1], values[1:], strict=True)
    ):
        raise A40ThroughputSmokeError("throughput timestamps must be finite and increasing")
    return values


def tail_interval_diagnostics(
    completed_step_seconds: Mapping[int, float],
) -> dict[str, object]:
    """Describe slow synchronized steps without imposing a per-step failure threshold."""

    values = _validated_throughput_timestamps(completed_step_seconds)
    intervals = [
        (completed_step, values[completed_step - 1] - values[completed_step - 2])
        for completed_step in range(STEADY_START_STEP, TOTAL_STEPS + 1)
    ]
    seconds = np.asarray([interval for _, interval in intervals], dtype=np.float64)
    slowest = sorted(intervals, key=lambda item: (-item[1], item[0]))[:10]
    return {
        "window_first_step": STEADY_START_STEP,
        "window_last_step": TOTAL_STEPS,
        "interval_count": len(intervals),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.quantile(seconds, 0.95)),
        "p99_seconds": float(np.quantile(seconds, 0.99)),
        "maximum_seconds": float(seconds.max()),
        "above_one_second_count": int(np.count_nonzero(seconds > 1.0)),
        "above_two_seconds_count": int(np.count_nonzero(seconds > 2.0)),
        "slowest_intervals": [
            {"completed_step": completed_step, "seconds": interval}
            for completed_step, interval in slowest
        ],
        "individual_interval_failure_threshold": None,
    }


def steady_throughput_report(
    completed_step_seconds: Mapping[int, float],
    *,
    minimum_steps_per_second: float = MINIMUM_STEPS_PER_SECOND,
) -> dict[str, object]:
    values = _validated_throughput_timestamps(completed_step_seconds)
    if not math.isfinite(minimum_steps_per_second) or minimum_steps_per_second <= 0:
        raise ValueError("minimum_steps_per_second must be finite and positive")
    excluded_completed_steps = STEADY_START_STEP - 1
    measured_steps = TOTAL_STEPS - excluded_completed_steps
    elapsed = values[-1] - values[excluded_completed_steps - 1]
    rate = measured_steps / elapsed
    if rate < minimum_steps_per_second:
        raise A40ThroughputSmokeError(
            f"steady throughput {rate:.6f} steps/s is below {minimum_steps_per_second:.6f}"
        )
    return {
        "window": {
            "first_step": STEADY_START_STEP,
            "last_step": TOTAL_STEPS,
            "measured_steps": measured_steps,
            "excluded_completed_steps": excluded_completed_steps,
        },
        "elapsed_seconds": elapsed,
        "steps_per_second": rate,
        "minimum_steps_per_second": minimum_steps_per_second,
        "tail_interval_diagnostics": tail_interval_diagnostics(completed_step_seconds),
        "passed": True,
    }


def validate_runtime_stats(stats: Mapping[str, object]) -> None:
    prefetch = stats.get("case_prefetch")
    host = stats.get("host_case_cache")
    gpu = stats.get("gpu_case_cache")
    plan_writer = stats.get("plan_artifact_writer")
    if (
        not isinstance(prefetch, Mapping)
        or not isinstance(host, Mapping)
        or not isinstance(gpu, Mapping)
        or not isinstance(plan_writer, Mapping)
    ):
        raise A40ThroughputSmokeError("optimized cache statistics are incomplete")
    consumed = prefetch.get("consumed_count")
    if (
        consumed != 20
        or prefetch.get("submitted_count") != 33
        or prefetch.get("discarded_count") != 0
    ):
        raise A40ThroughputSmokeError("prefetch did not consume exactly twenty scheduled cases")
    if prefetch.get("ready_hit_count") != 19 or prefetch.get("stall_count") != 1:
        raise A40ThroughputSmokeError(
            "only synchronous step-zero calibration may stall exact case consumption"
        )
    consumed_by_source = (
        prefetch.get("synchronous_consumed_count"),
        prefetch.get("startup_consumed_count"),
        prefetch.get("refill_consumed_count"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in consumed_by_source
    ) or consumed_by_source != (1, 16, 3):
        raise A40ThroughputSmokeError(
            "prefetch consumption did not directly prove calibration/startup/refill provenance"
        )
    if (
        prefetch.get("readiness_barrier_count") != 1
        or prefetch.get("readiness_barrier_key_count") != STARTUP_PREFETCH_KEY_COUNT
    ):
        raise A40ThroughputSmokeError(
            "startup did not make all sixteen exact lookahead cases ready"
        )
    ready_pending = prefetch.get("ready_pending_count")
    failed_pending = prefetch.get("failed_pending_count")
    running_pending = prefetch.get("running_pending_count")
    ready_prefix = prefetch.get("ready_prefix_count")
    if (
        prefetch.get("pending_count") != 13
        or isinstance(ready_pending, bool)
        or not isinstance(ready_pending, int)
        or ready_pending < 0
        or isinstance(failed_pending, bool)
        or not isinstance(failed_pending, int)
        or failed_pending != 0
        or isinstance(running_pending, bool)
        or not isinstance(running_pending, int)
        or running_pending < 0
        or ready_pending + failed_pending + running_pending != 13
        or isinstance(ready_prefix, bool)
        or not isinstance(ready_prefix, int)
        or not 5 <= ready_prefix <= ready_pending
    ):
        raise A40ThroughputSmokeError(
            "low-watermark replenishment did not provide a verified ready runway"
        )
    for record, name in ((host, "host"), (gpu, "gpu")):
        if record.get("miss_count") != 20 or record.get("hit_count") != 140:
            raise A40ThroughputSmokeError(
                f"{name} cache did not observe the exact 20-block/160-bag access pattern"
            )
    if (
        not isinstance(gpu.get("resident_bytes"), int)
        or gpu["resident_bytes"] <= 0
        or gpu["resident_bytes"] > gpu.get("byte_budget", 0)
        or gpu.get("peak_resident_bytes", 0) > gpu.get("byte_budget", 0)
    ):
        raise A40ThroughputSmokeError("GPU cache violated its byte budget")
    pending_plans = plan_writer.get("pending_count")
    completed_plans = plan_writer.get("completed_count")
    maximum_pending_plans = plan_writer.get("maximum_pending_count")
    if (
        plan_writer.get("mode") != "single_worker_ordered_bounded_async_atomic_create"
        or plan_writer.get("queue_depth") != 64
        or plan_writer.get("submitted_count") != TOTAL_STEPS
        or isinstance(pending_plans, bool)
        or not isinstance(pending_plans, int)
        or not 0 <= pending_plans <= 64
        or isinstance(completed_plans, bool)
        or not isinstance(completed_plans, int)
        or completed_plans + pending_plans != TOTAL_STEPS
        or isinstance(maximum_pending_plans, bool)
        or not isinstance(maximum_pending_plans, int)
        or not pending_plans <= maximum_pending_plans <= 64
        or any(
            isinstance(plan_writer.get(name), bool)
            or not isinstance(plan_writer.get(name), (int, float))
            or not math.isfinite(float(plan_writer[name]))
            or float(plan_writer[name]) < 0
            for name in ("persistence_seconds", "backpressure_seconds")
        )
    ):
        raise A40ThroughputSmokeError(
            "plan persistence did not prove bounded ordered asynchronous publication"
        )


def validate_closed_plan_persistence(stats: Mapping[str, object]) -> dict[str, object]:
    writer = stats.get("plan_artifact_writer")
    if not isinstance(writer, Mapping):
        raise A40ThroughputSmokeError("closed plan persistence statistics are missing")
    persistence_seconds = writer.get("persistence_seconds")
    if (
        writer.get("pending_count") != 0
        or writer.get("completed_count") != TOTAL_STEPS
        or isinstance(persistence_seconds, bool)
        or not isinstance(persistence_seconds, (int, float))
        or not math.isfinite(float(persistence_seconds))
        or float(persistence_seconds) <= 0
    ):
        raise A40ThroughputSmokeError("plan persistence did not fully drain after training")
    rate = TOTAL_STEPS / float(persistence_seconds)
    if rate < MINIMUM_STEPS_PER_SECOND:
        raise A40ThroughputSmokeError(
            f"plan persistence {rate:.6f} steps/s is below {MINIMUM_STEPS_PER_SECOND:.6f}"
        )
    return {
        "completed_plan_pairs": TOTAL_STEPS,
        "persistence_seconds": float(persistence_seconds),
        "plan_pairs_per_second": rate,
        "minimum_pairs_per_second": MINIMUM_STEPS_PER_SECOND,
        "passed": True,
    }


def compile_counter_report() -> dict[str, object]:
    """Snapshot Dynamo/Inductor evidence that compiled graphs really executed."""

    from torch._dynamo.utils import counters
    from torch._inductor import metrics as inductor_metrics

    groups = {
        str(group): {str(key): int(value) for key, value in values.items()}
        for group, values in counters.items()
    }
    stats = groups.get("stats", {})
    inductor = groups.get("inductor", {})
    cache_events = sum(value for key, value in inductor.items() if "fxgraph_cache" in key)
    return {
        "dynamo_calls_captured": int(stats.get("calls_captured", 0)),
        "dynamo_unique_graphs": int(stats.get("unique_graphs", 0)),
        "inductor_fxgraph_cache_events": cache_events,
        "inductor_generated_kernel_count": int(
            getattr(inductor_metrics, "generated_kernel_count", 0)
        ),
        "counter_groups": groups,
    }


def validate_compile_counters(report: Mapping[str, object]) -> None:
    for key in (
        "dynamo_calls_captured",
        "dynamo_unique_graphs",
        "inductor_fxgraph_cache_events",
        "inductor_generated_kernel_count",
    ):
        if isinstance(report.get(key), bool) or not isinstance(report.get(key), int):
            raise A40ThroughputSmokeError("compile counter report is malformed")
    if report["dynamo_calls_captured"] <= 0 or report["dynamo_unique_graphs"] <= 0:
        raise A40ThroughputSmokeError("Dynamo captured no compiled production graph")
    if (
        report["inductor_fxgraph_cache_events"] <= 0
        and report["inductor_generated_kernel_count"] <= 0
    ):
        raise A40ThroughputSmokeError("Inductor neither generated nor loaded a compiled graph")


def _capture_rng() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def _model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(_tensor_sha256(state[name]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise A40ThroughputSmokeError("throughput smoke requires CUDA")
    exact_runtime = configure_exact_resume_runtime(device)
    gpu_record = validate_a40_identity(
        name=torch.cuda.get_device_name(device),
        capability=torch.cuda.get_device_capability(device),
        visible_device_count=torch.cuda.device_count(),
    )
    training_runtime = configure_training_runtime(device)
    training_record = training_runtime.to_dict()
    if (
        training_record["autocast"]
        != {
            "enabled": True,
            "dtype": "bfloat16",
            "gradient_scaler_enabled": False,
        }
        or training_record["optimizer"]
        != {"name": "AdamW", "fused": True, "foreach": False, "capturable": False}
        or not training_record["compile"]["enabled"]  # type: ignore[index]
    ):
        raise A40ThroughputSmokeError("production BF16/compile/fused runtime contract drifted")

    repo = Path(args.repo_root).expanduser().resolve(strict=True)
    launch_sha = verify_git_sha(args.expected_git_sha, repo)
    manifest_file = _resolve_file(args.manifest, "filtered manifest")
    split_file = _resolve_file(args.split, "subject split")
    grids_file = _resolve_file(args.case_grid_manifest, "case-grid manifest")
    config_file = _resolve_file(args.config, "experiment config")
    manifest = load_manifest(manifest_file, expected_sha256=args.expected_manifest_sha256)
    split = load_split(split_file, expected_sha256=args.expected_split_sha256)
    case_grids = load_case_grid_manifest(
        grids_file,
        expected_sha256=args.expected_case_grid_manifest_sha256,
    )
    config = load_experiment_config(config_file)
    _assert_config(config)
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    train_cases = cases_for_splits(manifest, split, ("train",))
    heldout_cases = cases_for_splits(manifest, split, ("validation", "test"))
    schedule = SubjectBalancedSchedule(
        train_cases,
        seed=config.seed,
        bags_per_subject=BAGS_PER_SUBJECT,
    )
    if (
        schedule.case_count != EXPECTED_TRAIN_CASES
        or schedule.subject_count != EXPECTED_TRAIN_SUBJECTS
        or schedule.sha256 != SCHEDULE_SHA256
    ):
        raise A40ThroughputSmokeError("full train schedule does not match the registered cohort")
    blocks = scheduled_subject_blocks(schedule)
    heldout_ids = {case.case_id for case in heldout_cases}
    accessed_ids = {str(block["case_id"]) for block in blocks}
    if accessed_ids & heldout_ids:
        raise A40ThroughputSmokeError("throughput schedule crossed into held-out cases")

    requested = Path(args.output_dir).expanduser()
    output = requested.parent.resolve(strict=True) / requested.name
    output.mkdir(mode=0o700, exist_ok=False)
    reference_plans = output / "parity-reference-plans"
    production_plans = output / "production-plans"
    replay_plans = output / "fresh-replay-plans"
    reference_plans.mkdir(mode=0o700)
    production_plans.mkdir(mode=0o700)
    replay_plans.mkdir(mode=0o700)

    reference_factory: SubjectBalancedBatchFactory | None = None
    production_factory: SubjectBalancedBatchFactory | None = None
    replay_factory: SubjectBalancedBatchFactory | None = None
    success_payload: dict[str, object] | None = None
    try:
        reference_factory = SubjectBalancedBatchFactory(
            schedule=schedule,
            data_root=args.data_root,
            manifest=manifest,
            case_grids=case_grids,
            config=config,
            plans_dir=reference_plans,
            candidate_pool_size=512,
            max_plan_attempts=8,
            replay_existing=True,
        )
        optimized_config = OptimizedRuntimeConfig(
            prefetch_workers=8,
            prefetch_depth=16,
            prefetch_refill_batch_size=4,
            gpu_cache_bytes=4 * 1024**3,
            batched_gpu_extraction=True,
        )
        production_factory = SubjectBalancedBatchFactory(
            schedule=schedule,
            data_root=args.data_root,
            manifest=manifest,
            case_grids=case_grids,
            config=config,
            plans_dir=production_plans,
            candidate_pool_size=512,
            max_plan_attempts=8,
            replay_existing=True,
            optimized_runtime=optimized_config,
            optimized_device=device,
        )
        data_contract = production_factory.runtime_contract
        if data_contract != {
            **optimized_config.to_dict(),
            "optimized": True,
            "schedule_selects_samples": True,
            "cache_selects_samples": False,
            "plan_artifact_persistence": (
                "single_worker_ordered_bounded_async_atomic_create_flush_before_checkpoint"
            ),
            "plan_artifact_queue_depth": 64,
        }:
            raise A40ThroughputSmokeError("optimized data runtime contract drifted")

        reference_batch = reference_factory.materialize(0, prime_lookahead=False)
        reference_record = dict(reference_factory.last_record or {})
        production_batch = production_factory.materialize(0, prime_lookahead=False)
        production_record = dict(production_factory.last_record or {})
        # Match production startup: overlap exact cold lookahead with model
        # compilation, then enforce readiness before the timed optimizer path.
        production_factory.prime(0)
        parity = compare_reference_and_optimized_batches(
            reference_batch,
            production_batch,
            reference_plan=reference_record,
            optimized_plan=production_record,
        )

        from torch._dynamo import reset as reset_dynamo
        from torch._dynamo.utils import counters as dynamo_counters
        from torch._inductor import metrics as inductor_metrics

        reset_dynamo()
        dynamo_counters.clear()
        inductor_metrics.reset()
        torch.cuda.reset_peak_memory_stats(device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        system = build_matching_system(config).to(device).train()
        apply_model_runtime(system, training_runtime)
        initial_state = {
            name: value.detach().to(device="cpu").clone()
            for name, value in system.state_dict().items()
        }
        initial_model_sha256 = _model_state_sha256(initial_state)
        rng_before_warmup = _capture_rng()
        compile_start = time.perf_counter()
        system.zero_grad(set_to_none=True)
        with training_runtime.autocast(device):
            warmup_output = system(production_batch)
        if not bool(torch.isfinite(warmup_output.loss)):
            raise A40ThroughputSmokeError("compile warmup produced non-finite loss")
        warmup_output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            system.parameters(),
            float("inf"),
            error_if_nonfinite=True,
            foreach=False,
        )
        torch.cuda.synchronize(device)
        compile_seconds = time.perf_counter() - compile_start
        system.zero_grad(set_to_none=True)
        system.load_state_dict(initial_state, strict=True)
        _restore_rng(rng_before_warmup)
        compile_after_warmup = compile_counter_report()
        validate_compile_counters(compile_after_warmup)
        restored_state = {
            name: value.detach().to(device="cpu") for name, value in system.state_dict().items()
        }
        if _model_state_sha256(restored_state) != initial_model_sha256:
            raise A40ThroughputSmokeError("compile warmup changed the production initial state")

        # Ordinary training bags intentionally contain targets for only one D.
        # Build the collapse probe from one complete, deterministic four-bag D
        # cycle on the untimed reference path so production cache accounting and
        # the 160-step measurement remain untouched.
        probe_tables = [(reference_batch.target_patches, reference_batch.target_modality_ids)]
        for probe_absolute_index in range(1, len(config.task.modalities)):
            probe_batch = reference_factory.materialize(
                probe_absolute_index,
                prime_lookahead=False,
            )
            probe_tables.append((probe_batch.target_patches, probe_batch.target_modality_ids))
        probe = _fixed_target_probe_from_single_d_cycle(
            probe_tables,
            expected_modalities=range(len(config.task.modalities)),
        )
        calibration_rng = _capture_rng()
        calibration_start = time.perf_counter()
        with torch.no_grad(), training_runtime.autocast(device):
            calibration_output = system(production_batch)
            calibration_targets = system.target_teacher(probe.target_patches.to(device))
        torch.cuda.synchronize(device)
        calibration_seconds = time.perf_counter() - calibration_start
        _restore_rng(calibration_rng)
        references = stats_by_modality(
            calibration_targets,
            probe.target_modality_ids.to(device),
        )
        if set(references) != {0, 1, 2, 3} or any(
            value.variance <= 0 for value in references.values()
        ):
            raise A40ThroughputSmokeError("calibration references are incomplete or degenerate")

        optimizer = build_adamw_optimizer(
            system,
            learning_rate=_LEARNING_RATE,
            weight_decay=_WEIGHT_DECAY,
            policy=training_runtime,
        )
        if optimizer.state:
            raise A40ThroughputSmokeError("optimizer must start empty after compile/calibration")
        production_factory.wait_for_prefetch()
        manager = CheckpointManager(
            output / "checkpoints",
            policy=CheckpointPolicy(
                checkpoint_every_steps=1_000,
                artifact_every_steps=5_000,
            ),
            artifact_sink=None,
        )
        timestamps: dict[int, float] = {}
        metrics_records: list[dict[str, object]] = []
        diagnostic_steps: list[int] = []
        continuous_replay_capture: dict[str, object] = {}
        batch_stage_seconds: dict[int, dict[str, object]] = {}
        benchmark_origin = 0.0

        def timed_batches(absolute_step_index: int) -> MatchingBatch:
            batch = production_factory(absolute_step_index)
            record = production_factory.last_record or {}
            timings = record.get("runtime_stage_seconds")
            if isinstance(timings, Mapping):
                batch_stage_seconds[absolute_step_index + 1] = dict(timings)
            if absolute_step_index == REPLAY_ABSOLUTE_STEP_INDEX:
                # Retain the continuously consumed batch, but defer its CPU
                # transfer and hashing until after the measured window.
                continuous_replay_capture["batch"] = batch
                continuous_replay_capture["plan"] = dict(production_factory.last_record or {})
            return batch

        def on_step(metrics: StepMetrics) -> None:
            torch.cuda.synchronize(device)
            timestamps[metrics.step] = time.perf_counter() - benchmark_origin
            if metrics.diagnostics_measured:
                diagnostic_steps.append(metrics.step)
            metrics_records.append(
                {
                    "step": metrics.step,
                    "loss": metrics.loss,
                    "accuracy": metrics.accuracy,
                    "chance": metrics.chance,
                    "ema_update_count": metrics.ema_update_count,
                    "diagnostics_measured": metrics.diagnostics_measured,
                }
            )

        provenance = {
            "schema": "simple-brats.a40-optimized-throughput-smoke",
            "schema_version": 1,
            "launch_sha": launch_sha,
            "manifest_sha256": manifest.sha256,
            "split_sha256": split.sha256,
            "case_grid_manifest_sha256": case_grids.sha256,
            "config_sha256": config.sha256,
            "schedule_sha256": schedule.sha256,
            "training_split": "train",
            "validation_or_test_consumed": False,
            "training_runtime": training_record,
            "data_runtime": data_contract,
        }
        torch.cuda.synchronize(device)
        benchmark_origin = time.perf_counter()
        result = run_matching_training(
            system,
            optimizer,
            timed_batches,
            manager,
            provenance,
            total_steps=TOTAL_STEPS,
            max_steps=TOTAL_STEPS,
            collapse_probe=probe,
            collapse_reference=references,
            collapse_thresholds=_COLLAPSE_THRESHOLDS,
            collapse_warmup_steps=TOTAL_STEPS,
            gradient_clip_norm=_GRADIENT_CLIP_NORM,
            runtime_policy=training_runtime,
            on_step=on_step,
        )
        torch.cuda.synchronize(device)
        benchmark_total_seconds = time.perf_counter() - benchmark_origin
        compile_after_training = compile_counter_report()
        validate_compile_counters(compile_after_training)
        if (
            result.start_step != 0
            or result.end_step != TOTAL_STEPS
            or result.ema_update_count != TOTAL_STEPS
            or result.latest_checkpoint is not None
            or len(metrics_records) != TOTAL_STEPS
            or diagnostic_steps != [1, 50, 100, 150, 160]
        ):
            raise A40ThroughputSmokeError("160-step production runner contract was not exact")
        stats_before_close = production_factory.runtime_stats()
        candidate_elapsed = timestamps[TOTAL_STEPS] - timestamps[STEADY_START_STEP - 1]
        candidate_rate = (TOTAL_STEPS - STEADY_START_STEP + 1) / candidate_elapsed
        candidate_tail = tail_interval_diagnostics(timestamps)
        raw_slowest = candidate_tail.get("slowest_intervals")
        if not isinstance(raw_slowest, list):
            raise A40ThroughputSmokeError("tail interval diagnostics are malformed")
        slow_stage_diagnostics: list[dict[str, object]] = []
        for item in raw_slowest:
            if not isinstance(item, Mapping) or not isinstance(item.get("completed_step"), int):
                raise A40ThroughputSmokeError("slow interval diagnostics are malformed")
            completed_step = int(item["completed_step"])
            slow_stage_diagnostics.append(
                {
                    **dict(item),
                    "batch_stage_seconds": batch_stage_seconds.get(completed_step, {}),
                }
            )
        print(
            canonical_json_bytes(
                {
                    "schema": "simple-brats.a40-throughput-prethreshold",
                    "steps_per_second": candidate_rate,
                    "tail_interval_diagnostics": candidate_tail,
                    "slow_step_stage_diagnostics": slow_stage_diagnostics,
                    "data_runtime_stats": stats_before_close,
                }
            ).decode(),
            flush=True,
        )
        validate_runtime_stats(stats_before_close)
        throughput = steady_throughput_report(timestamps)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_before_replay = {
            "current_allocated_bytes": torch.cuda.memory_allocated(device),
            "current_reserved_bytes": torch.cuda.memory_reserved(device),
            "maximum_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "maximum_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": total_memory,
        }
        if (
            memory_before_replay["maximum_allocated_bytes"] <= 0
            or memory_before_replay["maximum_reserved_bytes"] <= 0
            or memory_before_replay["maximum_reserved_bytes"] > total_memory
        ):
            raise A40ThroughputSmokeError("CUDA memory accounting is invalid")

        # Tensor hashing and replay verification are audit-only and remain
        # strictly outside the measured 160-step production window.
        continuous_replayed_batch = continuous_replay_capture.get("batch")
        continuous_replay_plan = continuous_replay_capture.get("plan")
        if not isinstance(continuous_replayed_batch, MatchingBatch) or not isinstance(
            continuous_replay_plan, Mapping
        ):
            raise A40ThroughputSmokeError(
                "timed run did not retain its absolute-step-32 batch and plan"
            )
        continuous_replay_batch_sha256 = batch_semantic_sha256(continuous_replayed_batch)
        continuous_replay_capture.clear()

        # A new factory with no mutable cursor must reconstruct the exact batch
        # directly from the registered absolute step, as a resumed run would.
        production_factory.close()
        stats_after_close = production_factory.runtime_stats()
        plan_persistence_throughput = validate_closed_plan_persistence(stats_after_close)
        production_factory = None
        torch.cuda.empty_cache()
        replay_factory = SubjectBalancedBatchFactory(
            schedule=schedule,
            data_root=args.data_root,
            manifest=manifest,
            case_grids=case_grids,
            config=config,
            plans_dir=replay_plans,
            candidate_pool_size=512,
            max_plan_attempts=8,
            replay_existing=True,
            optimized_runtime=optimized_config,
            optimized_device=device,
        )
        if replay_factory.runtime_contract != data_contract:
            raise A40ThroughputSmokeError("fresh replay runtime differs from production")
        replayed_batch = replay_factory.materialize(
            REPLAY_ABSOLUTE_STEP_INDEX,
            prime_lookahead=False,
        )
        replay = compare_absolute_batch_replay(
            absolute_step_index=REPLAY_ABSOLUTE_STEP_INDEX,
            continuous_batch_sha256=continuous_replay_batch_sha256,
            replayed_batch=replayed_batch,
            continuous_plan=continuous_replay_plan,
            replayed_plan=dict(replay_factory.last_record or {}),
        )
        replay_runtime_stats = replay_factory.runtime_stats()
        torch.cuda.synchronize(device)
        memory_after_replay = {
            "current_allocated_bytes": torch.cuda.memory_allocated(device),
            "current_reserved_bytes": torch.cuda.memory_reserved(device),
            "maximum_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "maximum_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": total_memory,
        }
        if memory_after_replay["maximum_reserved_bytes"] > total_memory:
            raise A40ThroughputSmokeError("CUDA peak reservation exceeds device memory")
        success_payload = {
            "schema": "simple-brats.a40-optimized-throughput-smoke-result",
            "schema_version": 1,
            "status": "passed",
            "launch_sha": launch_sha,
            "gpu": gpu_record,
            "pins": {
                "manifest_sha256": manifest.sha256,
                "split_sha256": split.sha256,
                "case_grid_manifest_sha256": case_grids.sha256,
                "config_sha256": config.sha256,
                "config_file_sha256": sha256_file(config_file),
                "schedule_sha256": schedule.sha256,
                "train_case_count": schedule.case_count,
                "train_subject_count": schedule.subject_count,
            },
            "access_boundary": {
                "source_split": "train",
                "scheduled_case_ids": sorted(accessed_ids),
                "heldout_case_ids_loaded": [],
                "validation_image_access": False,
                "test_image_access": False,
                "segmentation_image_access": False,
            },
            "runtime": {
                "exact_resume": exact_runtime,
                "training": training_record,
                "data": data_contract,
            },
            "parity": parity,
            "absolute_step_fresh_replay": replay,
            "schedule_blocks": blocks,
            "excluded_timing": {
                "compile_warmup_seconds": compile_seconds,
                "calibration_seconds": calibration_seconds,
                "parity_and_case_preparation_excluded": True,
                "optimizer_constructed_after_compile_and_calibration": True,
                "initial_model_sha256": initial_model_sha256,
                "calibration_loss": float(calibration_output.loss),
                "compile_counters_after_warmup": compile_after_warmup,
            },
            "throughput": {
                **throughput,
                "all_160_steps_seconds": benchmark_total_seconds,
                "diagnostic_steps": diagnostic_steps,
                "completed_step_seconds": {
                    str(step): timestamps[step] for step in sorted(timestamps)
                },
                "slow_step_stage_diagnostics": slow_stage_diagnostics,
            },
            "plan_artifact_persistence_throughput": plan_persistence_throughput,
            "post_startup_refill": {
                "startup_key_count": STARTUP_PREFETCH_KEY_COUNT,
                "first_consumed_completed_step": FIRST_POST_STARTUP_REFILL_COMPLETED_STEP,
                "consumed_key_count": 3,
                "verified_by_exact_prefetch_accounting": True,
            },
            "runner": {
                "end_step": result.end_step,
                "ema_update_count": result.ema_update_count,
                "runner_contract_sha256": result.runner_contract_sha256,
                "metrics": metrics_records,
            },
            "compile_execution": compile_after_training,
            "data_runtime_stats_before_close": stats_before_close,
            "data_runtime_stats_after_close": stats_after_close,
            "fresh_replay_data_runtime_stats": replay_runtime_stats,
            "cuda_memory": {
                "production_before_fresh_replay": memory_before_replay,
                "after_fresh_replay": memory_after_replay,
            },
        }
    finally:
        if reference_factory is not None:
            reference_factory.close()
        if production_factory is not None:
            production_factory.close()
        if replay_factory is not None:
            replay_factory.close()

    assert success_payload is not None
    if production_factory is not None:
        success_payload["data_runtime_stats_after_close"] = production_factory.runtime_stats()
    result_sha256 = _write_result(output / "result.json", success_payload)
    print(canonical_json_bytes({**success_payload, "result_sha256": result_sha256}).decode())
    return success_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", default="configs/v0_cross_matching_small.toml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A40ThroughputSmokeError",
    "FIRST_POST_STARTUP_REFILL_COMPLETED_STEP",
    "MINIMUM_STEPS_PER_SECOND",
    "PARITY_ATOL",
    "PARITY_RTOL",
    "REPLAY_ABSOLUTE_STEP_INDEX",
    "STARTUP_PREFETCH_KEY_COUNT",
    "TOTAL_STEPS",
    "batch_semantic_sha256",
    "compare_absolute_batch_replay",
    "compare_reference_and_optimized_batches",
    "scheduled_subject_blocks",
    "steady_throughput_report",
    "tail_interval_diagnostics",
    "validate_a40_identity",
    "validate_compile_counters",
    "validate_runtime_stats",
]
