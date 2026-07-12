"""Pinned short real-data hard-matching training run.

This is the first experiment entrypoint, not a benchmark trainer.  It uses
only the subject split's training cases, reuses each prepared case for a small
block of bags, materializes every stochastic plan, and records both teacher
and prediction diagnostics for every optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.config import ExperimentConfig, load_experiment_config
from simple_brats.data.case_grids import CaseGridManifest, load_case_grid_manifest
from simple_brats.data.manifest import (
    CaseRecord,
    DatasetManifest,
    canonical_json_bytes,
    load_manifest,
    sha256_file,
)
from simple_brats.data.pipeline import (
    CachedNiftiPatchExtractor,
    PreparedCaseCandidateUniverse,
    materialize_case_matching_plan_record,
    prepare_case_candidate_universe,
)
from simple_brats.data.real_batches import assemble_matching_batch
from simple_brats.data.scheduled_cache import (
    ByteBoundedGpuCaseCache,
    OptimizedRuntimeConfig,
    ScheduleKeyedPrefetcher,
    assemble_batched_gpu_matching_batch,
)
from simple_brats.data.splits import cases_for_splits, load_split, validate_split
from simple_brats.provenance import verify_git_sha
from simple_brats.sampling import SlabGeometry, save_patch_plan
from simple_brats.training import (
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
    CheckpointManager,
    CheckpointPolicy,
    CollapseThresholds,
    FixedTargetPatchProbe,
    StepMetrics,
    WandbArtifactSink,
    build_matching_system,
    optimizer_parameter_groups,
    run_matching_training,
    stats_by_modality,
)

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CACHED_CASES = 4
_FIXED_PROBE_CASE_COUNT = 4
_MIN_FIXED_PROBE_SAMPLES_PER_MODALITY = 64
_PLAN_PERSISTENCE_QUEUE_DEPTH = 64
_DEFAULT_THRESHOLDS = CollapseThresholds(
    minimum_variance_ratio=0.10,
    minimum_effective_rank_ratio=0.25,
    maximum_off_diagonal_cosine=0.95,
)


class ShortRunError(RuntimeError):
    """The short run cannot satisfy its pinned experiment contract."""


class _OrderedPlanArtifactWriter:
    """Persist immutable plan/audit pairs off the optimizer critical path."""

    def __init__(self, *, queue_depth: int = _PLAN_PERSISTENCE_QUEUE_DEPTH) -> None:
        if isinstance(queue_depth, bool) or not isinstance(queue_depth, int) or queue_depth <= 0:
            raise ValueError("plan persistence queue_depth must be a positive integer")
        self.queue_depth = queue_depth
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="simple-brats-plan-persistence",
        )
        self._pending: deque[Future[float]] = deque()
        self._failed = threading.Event()
        self._closed = False
        self.submitted_count = 0
        self.completed_count = 0
        self.maximum_pending_count = 0
        self.persistence_seconds = 0.0
        self.backpressure_seconds = 0.0

    @staticmethod
    def _persist(
        *,
        prepared: Any,
        plan_path: Path,
        audit_path: Path,
        write_plan: bool,
        write_audit: bool,
        failed: threading.Event,
    ) -> float:
        if failed.is_set():
            raise ShortRunError("plan persistence halted after an earlier failure")
        started = time.perf_counter()
        try:
            if write_plan:
                save_patch_plan(prepared.plan, plan_path, overwrite=False)
            if write_audit:
                audit_sha256 = _write_new_canonical(audit_path, prepared.to_dict())
                if audit_sha256 != prepared.sha256:
                    raise ShortRunError("prepared plan audit does not match its canonical SHA")
        except BaseException:
            failed.set()
            raise
        return time.perf_counter() - started

    def _consume_oldest(self) -> None:
        future = self._pending.popleft()
        elapsed = future.result()
        self.completed_count += 1
        self.persistence_seconds += elapsed

    def _drain_completed(self) -> None:
        while self._pending and self._pending[0].done():
            self._consume_oldest()

    def submit(
        self,
        *,
        prepared: Any,
        plan_path: Path,
        audit_path: Path,
        write_plan: bool,
        write_audit: bool,
    ) -> None:
        if self._closed:
            raise ShortRunError("plan persistence writer is closed")
        if not write_plan and not write_audit:
            return
        self._drain_completed()
        if len(self._pending) >= self.queue_depth:
            started = time.perf_counter()
            try:
                self._consume_oldest()
            finally:
                self.backpressure_seconds += time.perf_counter() - started
        if self._failed.is_set():
            self._drain_completed()
            raise ShortRunError("plan persistence halted after an earlier failure")
        self._pending.append(
            self._executor.submit(
                self._persist,
                prepared=prepared,
                plan_path=plan_path,
                audit_path=audit_path,
                write_plan=write_plan,
                write_audit=write_audit,
                failed=self._failed,
            )
        )
        self.submitted_count += 1
        self.maximum_pending_count = max(self.maximum_pending_count, len(self._pending))

    def flush(self) -> None:
        while self._pending:
            self._consume_oldest()

    def stats(self) -> dict[str, int | float | str]:
        self._drain_completed()
        return {
            "mode": "single_worker_ordered_bounded_async_atomic_create",
            "queue_depth": self.queue_depth,
            "pending_count": len(self._pending),
            "maximum_pending_count": self.maximum_pending_count,
            "submitted_count": self.submitted_count,
            "completed_count": self.completed_count,
            "persistence_seconds": self.persistence_seconds,
            "backpressure_seconds": self.backpressure_seconds,
        }

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self.flush()
        except BaseException as caught:
            error = caught
            self._failed.set()
            for future in self._pending:
                future.cancel()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True
        if error is not None:
            raise error


@dataclass(frozen=True, slots=True)
class StepAssignment:
    """Deterministic mapping from an absolute optimizer index to one case/bag."""

    case_index: int
    epoch: int
    bag_index: int


@dataclass(frozen=True, slots=True)
class _CaseSamplingState:
    extractor: CachedNiftiPatchExtractor
    candidate_universe: PreparedCaseCandidateUniverse


@dataclass(frozen=True, slots=True)
class _FixedProbeBuild:
    probe: FixedTargetPatchProbe
    records: tuple[dict[str, object], ...]
    bags_per_case: int


def assignment_for_step(
    absolute_step_index: int,
    *,
    case_count: int,
    bags_per_case: int,
) -> StepAssignment:
    """Assign consecutive bag blocks to cases without mutable sampler state."""

    for value, name in (
        (absolute_step_index, "absolute_step_index"),
        (case_count, "case_count"),
        (bags_per_case, "bags_per_case"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if absolute_step_index < 0 or case_count <= 0 or bags_per_case <= 0:
        raise ValueError("step must be non-negative and case/bag counts must be positive")
    block = absolute_step_index // bags_per_case
    return StepAssignment(
        case_index=block % case_count,
        epoch=block // case_count,
        bag_index=absolute_step_index % bags_per_case,
    )


def run_classification(*, total_steps: int, checkpoint_every_steps: int) -> str:
    """Distinguish a no-checkpoint stability diagnostic from a reusable run."""

    if total_steps < checkpoint_every_steps:
        return "optimization_stability_diagnostic_not_representation_result"
    return "checkpointed_representation_pretraining"


def _wandb_for_schedule(*, total_steps: int, artifact_every_steps: int) -> Any | None:
    """Load W&B when available and require it before its first artifact step."""

    try:
        import wandb
    except Exception as error:
        if total_steps >= artifact_every_steps:
            raise ShortRunError(
                "W&B must be functional before a run can reach its artifact cadence"
            ) from error
        return None
    if not callable(getattr(wandb, "init", None)):
        if total_steps >= artifact_every_steps:
            raise ShortRunError(
                "W&B must expose callable init before a run can reach its artifact cadence"
            )
        return None
    return wandb


def _write_new_canonical(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    atomic_create_bytes(path, payload)
    if path.read_bytes() != payload:
        raise ShortRunError(f"artifact changed after canonical write: {path}")
    return hashlib.sha256(payload).hexdigest()


def _require_canonical(path: Path, value: Mapping[str, object], description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ShortRunError(f"{description} must be an existing non-symlink file: {path}")
    if path.read_bytes() != canonical_json_bytes(value):
        raise ShortRunError(f"{description} is not canonical on disk: {path}")


def _resolve_file(path: str | os.PathLike[str], description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ShortRunError(f"{description} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ShortRunError(f"{description} is unavailable: {path}") from error
    if not resolved.is_file():
        raise ShortRunError(f"{description} must be a regular file")
    return resolved


def _ordered_all_train_cases(
    manifest: DatasetManifest,
    split: object,
    *,
    seed: int,
) -> tuple[CaseRecord, ...]:
    train_cases = cases_for_splits(manifest, split, ("train",))  # type: ignore[arg-type]

    def key(case: CaseRecord) -> tuple[str, str]:
        identity = "\0".join(
            (case.source, case.release, case.subject_id, case.visit_id, case.case_id)
        )
        digest = hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()
        return digest, identity

    return tuple(sorted(train_cases, key=key))


def _ordered_train_cases(
    manifest: DatasetManifest,
    split: object,
    *,
    seed: int,
    max_cases: int,
) -> tuple[CaseRecord, ...]:
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    ordered = _ordered_all_train_cases(manifest, split, seed=seed)
    if len(ordered) < max_cases:
        raise ShortRunError(
            f"split has only {len(ordered)} training cases but max_cases={max_cases}"
        )
    return ordered[:max_cases]


def _held_out_probe_cases(
    manifest: DatasetManifest,
    split: object,
    *,
    seed: int,
    optimization_cases: Sequence[CaseRecord],
    case_count: int,
) -> tuple[CaseRecord, ...]:
    """Select deterministic training cases held out at the subject level."""

    if case_count <= 0:
        raise ValueError("probe case_count must be positive")
    optimization_subjects = {case.subject_id for case in optimization_cases}
    probe_subjects: set[str] = set()
    selected: list[CaseRecord] = []
    for case in _ordered_all_train_cases(manifest, split, seed=seed):
        if case.subject_id in optimization_subjects or case.subject_id in probe_subjects:
            continue
        selected.append(case)
        probe_subjects.add(case.subject_id)
        if len(selected) == case_count:
            return tuple(selected)
    raise ShortRunError(
        "training split does not contain enough subject-disjoint cases for the fixed probe"
    )


class DeterministicRealBatchFactory:
    """Absolute-step factory with a bounded cache of prepared case state."""

    def __init__(
        self,
        *,
        data_root: str | os.PathLike[str],
        manifest: DatasetManifest,
        case_grids: CaseGridManifest,
        cases: tuple[CaseRecord, ...],
        config: ExperimentConfig,
        plans_dir: Path,
        bags_per_case: int,
        candidate_pool_size: int,
        max_plan_attempts: int,
        replay_existing: bool = False,
        optimized_runtime: OptimizedRuntimeConfig | None = None,
        optimized_device: str | torch.device | None = None,
    ) -> None:
        self.data_root = data_root
        self.manifest = manifest
        self.case_grids = case_grids
        self.cases = cases
        self.config = config
        self.plans_dir = plans_dir
        self.bags_per_case = bags_per_case
        self.candidate_pool_size = candidate_pool_size
        self.max_plan_attempts = max_plan_attempts
        self.replay_existing = replay_existing
        if optimized_runtime is not None and not isinstance(
            optimized_runtime, OptimizedRuntimeConfig
        ):
            raise TypeError("optimized_runtime must be OptimizedRuntimeConfig or None")
        self.optimized_runtime = optimized_runtime
        self.optimized_device = (
            torch.device(optimized_device) if optimized_device is not None else None
        )
        if (
            optimized_runtime is not None
            and optimized_runtime.batched_gpu_extraction
            and (self.optimized_device is None or self.optimized_device.type != "cuda")
        ):
            raise ShortRunError("batched optimized extraction requires an explicit CUDA device")
        self.geometry = SlabGeometry(
            in_plane_footprint_mm=config.patch.footprint_mm,
            thin_extent_mm=config.patch.thin_mm,
            model_shape=config.patch.tensor_shape,
        )
        self._case_cache: OrderedDict[int, _CaseSamplingState] = OrderedDict()
        self._case_cache_lock = threading.RLock()
        self._case_cache_hit_count = 0
        self._case_cache_miss_count = 0
        self._case_cache_eviction_count = 0
        self._case_prefetcher: ScheduleKeyedPrefetcher[int, _CaseSamplingState] | None = None
        self._gpu_case_cache: ByteBoundedGpuCaseCache | None = None
        self._plan_artifact_writer: _OrderedPlanArtifactWriter | None = None
        if optimized_runtime is not None:
            self._case_prefetcher = ScheduleKeyedPrefetcher(
                self._load_case_state,
                workers=optimized_runtime.prefetch_workers,
                depth=optimized_runtime.prefetch_depth,
                refill_batch_size=optimized_runtime.prefetch_refill_batch_size,
                thread_name_prefix="simple-brats-case",
            )
            if optimized_runtime.batched_gpu_extraction:
                self._gpu_case_cache = ByteBoundedGpuCaseCache(
                    byte_budget=optimized_runtime.gpu_cache_bytes
                )
            self._plan_artifact_writer = _OrderedPlanArtifactWriter()
        self._cached_step: int | None = None
        self._cached_batch: Any | None = None
        self._initial_lookahead_primed = False
        self._startup_prefetch_keys: set[int] = set()
        self._refill_prefetch_keys: set[int] = set()
        self._synchronous_consumed_count = 0
        self._startup_consumed_count = 0
        self._refill_consumed_count = 0
        self.last_record: dict[str, object] | None = None
        # Wall-clock timings are operational telemetry, not deterministic
        # sample provenance.  Keep them separate so replayed batches retain an
        # identical provenance record across processes and allocations.
        self.last_runtime_stage_seconds: dict[str, float] | None = None

    def _load_case_state(self, case_index: int) -> _CaseSamplingState:
        if (
            isinstance(case_index, bool)
            or not isinstance(case_index, int)
            or not (0 <= case_index < len(self.cases))
        ):
            raise ShortRunError("prefetched case index is outside the exact case table")
        with self._case_cache_lock:
            cached = self._case_cache.get(case_index)
        if cached is not None:
            return cached
        case = self.cases[case_index]
        # The case-grid catalog owns scan-specific shape/origin.  Patch
        # footprint/model shape are supplied by its extraction policy.
        spec = self.case_grids.extraction_spec_for_case(
            case,
            patch_config=self.config.patch,
        )
        extractor = CachedNiftiPatchExtractor(
            data_root=self.data_root,
            manifest=self.manifest,
            data_manifest_sha256=self.manifest.sha256,
            extraction_spec=spec,
            max_cached_volumes=4,
        )
        candidate_universe = prepare_case_candidate_universe(
            extractor,
            case,
            geometry=self.geometry,
        )
        return _CaseSamplingState(
            extractor=extractor,
            candidate_universe=candidate_universe,
        )

    def _consume_case_future(self, case_index: int) -> _CaseSamplingState:
        """Consume one exact future and account for its submission origin."""

        if self._case_prefetcher is None:
            return self._load_case_state(case_index)
        was_pending = case_index in self._case_prefetcher.pending_keys
        state = self._case_prefetcher.get(case_index)
        if not was_pending:
            self._synchronous_consumed_count += 1
        elif case_index in self._startup_prefetch_keys:
            self._startup_prefetch_keys.remove(case_index)
            self._startup_consumed_count += 1
        elif case_index in self._refill_prefetch_keys:
            self._refill_prefetch_keys.remove(case_index)
            self._refill_consumed_count += 1
        else:
            raise ShortRunError("pending case future has no registered submission origin")
        return state

    def _activate(self, case_index: int) -> _CaseSamplingState:
        with self._case_cache_lock:
            cached = self._case_cache.get(case_index)
            if cached is not None:
                self._case_cache_hit_count += 1
                self._case_cache.move_to_end(case_index)
            else:
                self._case_cache_miss_count += 1
        pending = (
            self._case_prefetcher is not None and case_index in self._case_prefetcher.pending_keys
        )
        if cached is not None and not pending:
            return cached
        state = self._consume_case_future(case_index)
        with self._case_cache_lock:
            raced = self._case_cache.get(case_index)
            if raced is not None:
                if raced is not state:
                    raise ShortRunError(
                        "prefetched case state differs from resident cache identity"
                    )
                self._case_cache.move_to_end(case_index)
                return raced
            self._case_cache[case_index] = state
            self._case_cache.move_to_end(case_index)
            while len(self._case_cache) > min(_MAX_CACHED_CASES, len(self.cases)):
                self._case_cache.popitem(last=False)
                self._case_cache_eviction_count += 1
            return state

    def _assignment_for_absolute_step(self, absolute_step_index: int) -> StepAssignment:
        return assignment_for_step(
            absolute_step_index,
            case_count=len(self.cases),
            bags_per_case=self.bags_per_case,
        )

    def _prefetch_scan_block_horizon(self) -> int:
        """Bound schedule inspection while allowing subclasses to tighten it.

        A full case table is a safe generic horizon for the cyclic base
        schedule and for subject-balanced schedules whose cases are a superset
        of one subject epoch.  The scan normally exits after only the handful
        of distinct cases needed for one refill.
        """

        depth = self.optimized_runtime.prefetch_depth if self.optimized_runtime else 1
        return max(len(self.cases), depth)

    def prime(self, absolute_step_index: int) -> tuple[int, ...]:
        """Prefetch exact upcoming scheduled cases without choosing any sample."""

        if self._case_prefetcher is None or self.optimized_runtime is None:
            return ()
        pending = set(self._case_prefetcher.pending_keys)
        if len(pending) > self.optimized_runtime.prefetch_refill_low_watermark:
            return ()
        missing = self.optimized_runtime.prefetch_depth - len(pending)
        if missing <= 0:
            return ()
        ordered: list[int] = []
        with self._case_cache_lock:
            current_case_index = self._assignment_for_absolute_step(absolute_step_index).case_index
            current_is_resident = current_case_index in self._case_cache
        seen: set[int] = set(pending)
        if current_is_resident:
            seen.add(current_case_index)
        block_horizon = self._prefetch_scan_block_horizon()
        if (
            isinstance(block_horizon, bool)
            or not isinstance(block_horizon, int)
            or block_horizon <= 0
        ):
            raise ShortRunError("prefetch scan block horizon must be a positive integer")
        # Inspect exact absolute-step assignments rather than deriving a second
        # schedule.  The bounded horizon handles resident/pending keys and rare
        # epoch-boundary duplicates without allowing cache state to select data.
        stop = absolute_step_index + block_horizon * max(self.bags_per_case, 1)
        for step in range(absolute_step_index, stop):
            case_index = self._assignment_for_absolute_step(step).case_index
            if case_index not in seen:
                ordered.append(case_index)
                seen.add(case_index)
                if len(ordered) == missing:
                    break
        if not self._initial_lookahead_primed:
            submitted: list[int] = []
            remaining = ordered
            while remaining:
                batch = self._case_prefetcher.prime(remaining)
                if not batch:
                    break
                submitted.extend(batch)
                submitted_set = set(batch)
                remaining = [key for key in remaining if key not in submitted_set]
            self._initial_lookahead_primed = True
            if self._startup_prefetch_keys or self._refill_prefetch_keys:
                raise ShortRunError("initial prefetch origin tables must be empty")
            self._startup_prefetch_keys.update(submitted)
            return tuple(submitted)
        submitted = self._case_prefetcher.prime(ordered)
        if set(submitted) & (self._startup_prefetch_keys | self._refill_prefetch_keys):
            raise ShortRunError("refill prefetch key already has a submission origin")
        self._refill_prefetch_keys.update(submitted)
        return submitted

    @property
    def runtime_contract(self) -> dict[str, object]:
        if self.optimized_runtime is None:
            return {
                "schema": "simple-brats.reference-data-runtime",
                "schema_version": 1,
                "optimized": False,
            }
        return {
            **self.optimized_runtime.to_dict(),
            "optimized": True,
            "schedule_selects_samples": True,
            "cache_selects_samples": False,
            "plan_artifact_persistence": (
                "single_worker_ordered_bounded_async_atomic_create_flush_before_checkpoint"
            ),
            "plan_artifact_queue_depth": _PLAN_PERSISTENCE_QUEUE_DEPTH,
        }

    def runtime_stats(self) -> dict[str, object]:
        prefetch_stats = self._case_prefetcher.to_dict() if self._case_prefetcher else None
        if prefetch_stats is not None:
            prefetch_stats.update(
                {
                    "synchronous_consumed_count": self._synchronous_consumed_count,
                    "startup_consumed_count": self._startup_consumed_count,
                    "refill_consumed_count": self._refill_consumed_count,
                    "unconsumed_startup_prefetch_count": len(self._startup_prefetch_keys),
                }
            )
        return {
            "case_prefetch": prefetch_stats,
            "host_case_cache": {
                "resident_case_count": len(self._case_cache),
                "hit_count": self._case_cache_hit_count,
                "miss_count": self._case_cache_miss_count,
                "eviction_count": self._case_cache_eviction_count,
            },
            "gpu_case_cache": (
                self._gpu_case_cache.to_dict() if self._gpu_case_cache is not None else None
            ),
            "plan_artifact_writer": (
                self._plan_artifact_writer.stats()
                if self._plan_artifact_writer is not None
                else None
            ),
        }

    def flush_plan_artifacts(self) -> None:
        """Make every submitted plan/audit pair durable before a checkpoint."""

        if self._plan_artifact_writer is not None:
            self._plan_artifact_writer.flush()

    def wait_for_prefetch(self) -> tuple[int, ...]:
        """Complete submitted exact lookahead while retaining it for consumption."""

        return self._case_prefetcher.wait_pending() if self._case_prefetcher is not None else ()

    def discard_prefetch(self) -> tuple[int, ...]:
        """Discard unused lookahead when no overlapping keys will be re-primed."""

        discarded = (
            self._case_prefetcher.discard_pending() if self._case_prefetcher is not None else ()
        )
        self._startup_prefetch_keys.difference_update(discarded)
        self._refill_prefetch_keys.difference_update(discarded)
        return discarded

    def close(self) -> None:
        persistence_error: BaseException | None = None
        try:
            if self._plan_artifact_writer is not None:
                self._plan_artifact_writer.close()
        except BaseException as error:
            persistence_error = error
        finally:
            if self._case_prefetcher is not None:
                self._case_prefetcher.close(cancel_pending=True)
        if persistence_error is not None:
            raise persistence_error

    def materialize(
        self,
        absolute_step_index: int,
        *,
        prime_lookahead: bool = True,
    ) -> Any:
        """Materialize one exact step, optionally without scheduling lookahead."""

        if not isinstance(prime_lookahead, bool):
            raise TypeError("prime_lookahead must be boolean")
        if self._cached_step == absolute_step_index:
            return self._cached_batch
        if prime_lookahead:
            self.prime(absolute_step_index)
        assignment = self._assignment_for_absolute_step(absolute_step_index)
        return self._batch_for_assignment(absolute_step_index, assignment)

    def __call__(self, absolute_step_index: int) -> Any:
        return self.materialize(absolute_step_index)

    def _batch_for_assignment(
        self,
        absolute_step_index: int,
        assignment: StepAssignment,
    ) -> Any:
        """Materialize one validated absolute assignment.

        Long-running schedulers reuse this operation while supplying their own
        stateless subject/case ordering.  The assignment remains fully bound to
        the absolute optimizer index by the caller's provenance contract.
        """

        if self._cached_step == absolute_step_index:
            return self._cached_batch
        if not isinstance(assignment, StepAssignment):
            raise TypeError("assignment must be a StepAssignment")
        if not 0 <= assignment.case_index < len(self.cases):
            raise ValueError("assignment case index is outside the factory case table")
        materialization_started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        case = self.cases[assignment.case_index]
        stage_started = time.perf_counter()
        state = self._activate(assignment.case_index)
        stage_seconds["case_activation"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        prepared = materialize_case_matching_plan_record(
            state.extractor,
            case,
            state.candidate_universe,
            epoch=assignment.epoch,
            bag_index=assignment.bag_index,
            experiment_seed=self.config.seed,
            geometry=self.geometry,
            prism_extent_mm=self.config.task.prism_extent_mm,
            target_count=self.config.task.target_patches_per_bag,
            candidate_pool_size=self.candidate_pool_size,
            max_attempts=self.max_plan_attempts,
        )
        stage_seconds["plan_materialization"] = time.perf_counter() - stage_started
        stem = f"step-{absolute_step_index + 1:09d}"
        plan_path = self.plans_dir / f"{stem}.plan.json"
        audit_path = self.plans_dir / f"{stem}.prepared.json"
        plan_present = os.path.lexists(plan_path)
        audit_present = os.path.lexists(audit_path)
        if not self.replay_existing and (plan_present or audit_present):
            raise ShortRunError("fresh plan persistence refuses existing filesystem entries")
        if self.replay_existing and audit_present and not plan_present:
            raise ShortRunError("replayed prepared-plan audit exists without its patch plan")
        plan_exists = self.replay_existing and plan_present
        audit_exists = self.replay_existing and audit_present
        stage_started = time.perf_counter()
        if plan_exists:
            _require_canonical(plan_path, prepared.plan.to_dict(), "replayed patch plan")
        if audit_exists:
            _require_canonical(audit_path, prepared.to_dict(), "replayed prepared-plan audit")
            if hashlib.sha256(audit_path.read_bytes()).hexdigest() != prepared.sha256:
                raise ShortRunError("prepared plan audit does not match its canonical SHA")
        if self._plan_artifact_writer is not None:
            self._plan_artifact_writer.submit(
                prepared=prepared,
                plan_path=plan_path,
                audit_path=audit_path,
                write_plan=not plan_exists,
                write_audit=not audit_exists,
            )
        else:
            if not plan_exists:
                save_patch_plan(prepared.plan, plan_path, overwrite=False)
            if not audit_exists:
                audit_sha256 = _write_new_canonical(audit_path, prepared.to_dict())
                if audit_sha256 != prepared.sha256:
                    raise ShortRunError("prepared plan audit does not match its canonical SHA")
        stage_seconds["plan_persistence_submission"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        if self._gpu_case_cache is None:
            batch = assemble_matching_batch(
                case,
                prepared.plan,
                state.extractor,
                data_manifest_sha256=self.manifest.sha256,
                plan_sha256=prepared.plan.sha256,
                extraction_spec_sha256=state.extractor.extraction_spec_sha256,
            )
            stage_seconds["reference_batch_assembly"] = time.perf_counter() - stage_started
        else:
            assert self.optimized_device is not None
            volumes = state.extractor.canonical_volumes_for_case(case)
            stage_seconds["canonical_volume_access"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            gpu_case = self._gpu_case_cache.get_or_upload(
                case=case,
                extraction_spec=state.extractor.extraction_spec,
                canonical_volumes=volumes,
                candidate_universe=state.candidate_universe,
                device=self.optimized_device,
            )
            stage_seconds["gpu_case_cache"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            batch = assemble_batched_gpu_matching_batch(
                case=case,
                plan=prepared.plan,
                extractor=state.extractor,
                gpu_case=gpu_case,
                extraction_spec=state.extractor.extraction_spec,
                candidate_universe=state.candidate_universe,
                data_manifest_sha256=self.manifest.sha256,
                plan_sha256=prepared.plan.sha256,
                extraction_spec_sha256=state.extractor.extraction_spec_sha256,
            )
            stage_seconds["gpu_batch_assembly"] = time.perf_counter() - stage_started
        if (
            batch.source_patches.shape[1] != self.config.task.source_patches_per_bag
            or batch.target_patches.shape[1] != self.config.task.target_patches_per_bag
        ):
            raise ShortRunError(
                "materialized batch violates the registered 96-source/32-target shape"
            )
        self.last_record = {
            "absolute_step_index": absolute_step_index,
            "completed_step": absolute_step_index + 1,
            "case_id": case.case_id,
            "subject_id": case.subject_id,
            "visit_id": case.visit_id,
            "epoch": assignment.epoch,
            "bag_index": assignment.bag_index,
            "extraction_spec_sha256": state.extractor.extraction_spec_sha256,
            "plan_sha256": prepared.plan.sha256,
            "prepared_plan_sha256": prepared.sha256,
            "candidate_centers_sha256": prepared.candidate_centers_sha256,
            "candidate_count": prepared.candidate_count,
            "plan_file": plan_path.name,
            "prepared_file": audit_path.name,
        }
        self.last_runtime_stage_seconds = {
            **stage_seconds,
            "total_batch_materialization": time.perf_counter() - materialization_started,
        }
        self._cached_step = absolute_step_index
        self._cached_batch = batch
        return batch


def _build_fixed_target_probe(
    *,
    data_root: str | os.PathLike[str],
    manifest: DatasetManifest,
    case_grids: CaseGridManifest,
    cases: tuple[CaseRecord, ...],
    config: ExperimentConfig,
    plans_dir: Path,
    candidate_pool_size: int,
    max_plan_attempts: int,
    replay_existing: bool = False,
) -> _FixedProbeBuild:
    """Materialize a subject-held-out, multi-case fixed target-patch probe."""

    if len(cases) < 2:
        raise ShortRunError("fixed collapse probe must span at least two cases")
    # A corrected bag has one target modality D.  The planner guarantees one
    # balanced random permutation of all modalities per four consecutive bag
    # indices, so materialize only complete cycles and size them to the fixed
    # per-modality probe minimum across the selected cases.
    cycles_per_case = math.ceil(
        _MIN_FIXED_PROBE_SAMPLES_PER_MODALITY / (len(cases) * config.task.target_patches_per_bag)
    )
    bags_per_case = len(config.task.modalities) * cycles_per_case
    factory = DeterministicRealBatchFactory(
        data_root=data_root,
        manifest=manifest,
        case_grids=case_grids,
        cases=cases,
        config=config,
        plans_dir=plans_dir,
        bags_per_case=bags_per_case,
        candidate_pool_size=candidate_pool_size,
        max_plan_attempts=max_plan_attempts,
        replay_existing=replay_existing,
    )
    patch_tables: list[torch.Tensor] = []
    modality_tables: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    for probe_index in range(len(cases) * bags_per_case):
        batch = factory(probe_index)
        if batch.target_patches.shape[0] != 1 or batch.target_modality_ids.shape[0] != 1:
            raise ShortRunError("real fixed-probe bags must have singleton batch dimension")
        patch_tables.append(batch.target_patches.detach().cpu())
        modality_tables.append(batch.target_modality_ids.detach().cpu())
        if factory.last_record is None:
            raise ShortRunError("fixed probe bag is missing materialized provenance")
        records.append({"probe_bag_index": probe_index, **dict(factory.last_record)})

    probe = FixedTargetPatchProbe(
        torch.cat(patch_tables, dim=1),
        torch.cat(modality_tables, dim=1),
    )
    if min(probe.sample_count_by_modality.values()) < _MIN_FIXED_PROBE_SAMPLES_PER_MODALITY:
        raise ShortRunError("fixed probe did not reach its minimum samples per modality")
    return _FixedProbeBuild(
        probe=probe,
        records=tuple(records),
        bags_per_case=bags_per_case,
    )


def _stats_record(stats: Mapping[int, Any]) -> dict[str, object]:
    return {str(key): value.to_dict() for key, value in sorted(stats.items())}


class _MetricsLogger:
    _WANDB_SCALAR_EVERY_STEPS = 10

    def __init__(
        self,
        path: Path,
        factory: DeterministicRealBatchFactory,
        wandb_run: Any,
        *,
        schema: str = "simple-brats.short-run-step",
    ) -> None:
        self._handle = path.open("xb")
        self._factory = factory
        self._wandb_run = wandb_run
        self._schema = schema
        self._previous_wandb_step: int | None = None
        self._previous_wandb_time: float | None = None

    def close(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def _runtime_telemetry(self) -> dict[str, int | float]:
        runtime_stats = getattr(self._factory, "runtime_stats", None)
        if not callable(runtime_stats):
            return {}
        stats = runtime_stats()
        if not isinstance(stats, Mapping):
            raise ShortRunError("data runtime statistics must be a mapping")
        result: dict[str, int | float] = {}
        for section, prefix in (
            (stats.get("case_prefetch"), "runtime/prefetch"),
            (stats.get("host_case_cache"), "runtime/host_cache"),
            (stats.get("gpu_case_cache"), "runtime/gpu_cache"),
            (stats.get("plan_artifact_writer"), "runtime/plan_persistence"),
        ):
            if section is None:
                continue
            if not isinstance(section, Mapping):
                raise ShortRunError(f"{prefix} statistics must be a mapping or null")
            for key, value in section.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[f"{prefix}/{key}"] = value
        return result

    def __call__(self, metrics: StepMetrics) -> None:
        if self._factory.last_record is None:
            raise ShortRunError("step metrics arrived without a batch provenance record")
        batch_record = dict(self._factory.last_record)
        runtime_stage_seconds = getattr(
            self._factory,
            "last_runtime_stage_seconds",
            None,
        )
        if runtime_stage_seconds is not None:
            if not isinstance(runtime_stage_seconds, Mapping):
                raise ShortRunError("batch runtime stage telemetry must be a mapping or null")
            batch_record["runtime_stage_seconds"] = dict(runtime_stage_seconds)
        streams = {
            stream: _stats_record(values)
            for stream, values in sorted(metrics.diagnostics_by_stream.items())
        }
        record: dict[str, object] = {
            "schema": self._schema,
            "schema_version": 3,
            "step": metrics.step,
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "chance": metrics.chance,
            "ema_update_count": metrics.ema_update_count,
            "batch": batch_record,
            "diagnostics_measured": metrics.diagnostics_measured,
            "diagnostics_by_stream": streams,
        }
        self._handle.write(canonical_json_bytes(record) + b"\n")
        self._handle.flush()
        if self._wandb_run is not None and (
            metrics.step == 1
            or metrics.step % self._WANDB_SCALAR_EVERY_STEPS == 0
            or metrics.diagnostics_measured
        ):
            flat: dict[str, int | float] = {
                "train/loss": metrics.loss,
                "train/accuracy": metrics.accuracy,
                "train/chance": metrics.chance,
            }
            for stream, values in metrics.diagnostics_by_stream.items():
                for modality_id, stats in values.items():
                    prefix = f"diagnostics/{stream}/modality_{modality_id}"
                    flat[f"{prefix}/variance"] = stats.variance
                    flat[f"{prefix}/effective_rank"] = stats.effective_rank
                    flat[f"{prefix}/off_diagonal_cosine"] = stats.off_diagonal_cosine
            now = time.perf_counter()
            if self._previous_wandb_step is not None and self._previous_wandb_time is not None:
                elapsed = now - self._previous_wandb_time
                completed = metrics.step - self._previous_wandb_step
                if elapsed > 0 and completed > 0:
                    flat["performance/interval_seconds"] = elapsed
                    flat["performance/completed_steps_per_second"] = completed / elapsed
            flat.update(self._runtime_telemetry())
            self._wandb_run.log(flat, step=metrics.step)
            # Exclude tracker transport latency from the next model/data interval.
            self._previous_wandb_step = metrics.step
            self._previous_wandb_time = time.perf_counter()


def _capture_torch_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )


def _restore_torch_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def run_short_matching(
    *,
    data_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    expected_manifest_sha256: str,
    split_path: str | os.PathLike[str],
    expected_split_sha256: str,
    case_grid_manifest_path: str | os.PathLike[str],
    expected_case_grid_manifest_sha256: str,
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    expected_git_sha: str,
    repo_root: str | os.PathLike[str] = ".",
    total_steps: int = 100,
    max_cases: int = 4,
    bags_per_case: int = 25,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    gradient_clip_norm: float = 10.0,
    collapse_warmup_steps: int = 25,
    candidate_pool_size: int = 512,
    max_plan_attempts: int = 8,
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Run a provenance-locked short hard-matching experiment."""

    if _FULL_GIT_SHA.fullmatch(expected_git_sha) is None:
        raise ValueError("expected_git_sha must be one full lowercase commit ID")
    for value, name in (
        (expected_manifest_sha256, "expected_manifest_sha256"),
        (expected_split_sha256, "expected_split_sha256"),
        (expected_case_grid_manifest_sha256, "expected_case_grid_manifest_sha256"),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    for value, name in (
        (total_steps, "total_steps"),
        (max_cases, "max_cases"),
        (bags_per_case, "bags_per_case"),
        (collapse_warmup_steps, "collapse_warmup_steps"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if learning_rate <= 0 or weight_decay < 0 or gradient_clip_norm <= 0:
        raise ValueError("optimizer values require lr>0, decay>=0, and grad clip>0")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ShortRunError("CUDA was requested but is unavailable")
    repo = Path(repo_root).expanduser().resolve(strict=True)
    launch_sha = verify_git_sha(expected_git_sha, repo)
    manifest_file = _resolve_file(manifest_path, "filtered manifest")
    split_file = _resolve_file(split_path, "subject split")
    grids_file = _resolve_file(case_grid_manifest_path, "case-grid manifest")
    config_file = _resolve_file(config_path, "experiment config")
    manifest = load_manifest(manifest_file, expected_sha256=expected_manifest_sha256)
    split = load_split(split_file, expected_sha256=expected_split_sha256)
    case_grids = load_case_grid_manifest(
        grids_file,
        expected_sha256=expected_case_grid_manifest_sha256,
    )
    config = load_experiment_config(config_file)
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    _require_canonical(manifest_file, manifest.to_dict(), "filtered manifest")
    _require_canonical(split_file, split.to_dict(), "subject split")
    cases = _ordered_train_cases(manifest, split, seed=config.seed, max_cases=max_cases)
    probe_cases = _held_out_probe_cases(
        manifest,
        split,
        seed=config.seed,
        optimization_cases=cases,
        case_count=_FIXED_PROBE_CASE_COUNT,
    )
    classification = run_classification(
        total_steps=total_steps,
        checkpoint_every_steps=config.checkpoint_every_steps,
    )
    wandb_module = _wandb_for_schedule(
        total_steps=total_steps,
        artifact_every_steps=config.artifact_every_steps,
    )
    tracking_mode = (
        "offline_wandb_and_canonical_jsonl"
        if wandb_module is not None
        else "canonical_jsonl_only_below_artifact_cadence"
    )

    requested_output = Path(output_dir).expanduser()
    parent = requested_output.parent.resolve(strict=True)
    destination = parent / requested_output.name
    destination.mkdir(mode=0o700, exist_ok=False)
    plans_dir = destination / "plans"
    plans_dir.mkdir(mode=0o700)
    probe_plans_dir = destination / "fixed-probe-plans"
    probe_plans_dir.mkdir(mode=0o700)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    system = build_matching_system(config).to(resolved_device).train()
    probe_build = _build_fixed_target_probe(
        data_root=data_root,
        manifest=manifest,
        case_grids=case_grids,
        cases=probe_cases,
        config=config,
        plans_dir=probe_plans_dir,
        candidate_pool_size=candidate_pool_size,
        max_plan_attempts=max_plan_attempts,
    )
    collapse_probe = probe_build.probe
    probe_artifact: dict[str, object] = {
        "schema": "simple-brats.short-run-fixed-target-probe",
        "schema_version": 1,
        "purpose": "representation-collapse-abort-only",
        "source_split": "train",
        "subject_disjoint_from_optimization_cases": True,
        "probe_sha256": collapse_probe.sha256,
        "patch_table_shape": list(collapse_probe.target_patches.shape),
        "modality_id_table_shape": list(collapse_probe.target_modality_ids.shape),
        "sample_count_by_modality": {
            str(modality_id): count
            for modality_id, count in collapse_probe.sample_count_by_modality.items()
        },
        "minimum_samples_per_modality": _MIN_FIXED_PROBE_SAMPLES_PER_MODALITY,
        "case_ids": [case.case_id for case in probe_cases],
        "subject_ids": [case.subject_id for case in probe_cases],
        "bags_per_case": probe_build.bags_per_case,
        "bag_count": len(probe_build.records),
        "plans_directory": probe_plans_dir.name,
        "bags": list(probe_build.records),
    }
    probe_artifact_sha256 = _write_new_canonical(
        destination / "fixed-target-probe.json",
        probe_artifact,
    )
    factory = DeterministicRealBatchFactory(
        data_root=data_root,
        manifest=manifest,
        case_grids=case_grids,
        cases=cases,
        config=config,
        plans_dir=plans_dir,
        bags_per_case=bags_per_case,
        candidate_pool_size=candidate_pool_size,
        max_plan_attempts=max_plan_attempts,
    )

    # Training-batch baselines remain descriptive.  Only the exact fixed probe
    # defines collapse references and later abort decisions.
    calibration_batch = factory(0).to(resolved_device)
    probe_patches = collapse_probe.target_patches.to(resolved_device)
    probe_modality_ids = collapse_probe.target_modality_ids.to(resolved_device)
    rng_state = _capture_torch_rng()
    try:
        with torch.no_grad():
            calibration_output = system(calibration_batch)
            probe_targets = system.target_teacher(probe_patches)
    finally:
        _restore_torch_rng(rng_state)
    references = stats_by_modality(
        probe_targets,
        probe_modality_ids,
    )
    training_teacher_baseline = stats_by_modality(
        calibration_output.targets,
        calibration_batch.target_modality_ids,
    )
    prediction_baseline = stats_by_modality(
        calibration_output.predictions,
        calibration_batch.query_modality_ids,
    )
    expected_modalities = set(range(len(config.task.modalities)))
    if (
        set(references) != expected_modalities
        or any(value.variance <= 0 for value in references.values())
        or {modality_id: value.count for modality_id, value in references.items()}
        != collapse_probe.sample_count_by_modality
    ):
        raise ShortRunError(
            "initial teacher calibration is missing a modality or has zero variance"
        )
    calibration: dict[str, object] = {
        "schema": "simple-brats.short-run-calibration",
        "schema_version": 2,
        "timing": "initialized_model_before_optimizer_construction_and_training",
        "training_batch": factory.last_record,
        "collapse_stream": TEACHER_TARGET_DIAGNOSTIC_STREAM,
        "fixed_probe": {
            "probe_sha256": collapse_probe.sha256,
            "artifact_file": "fixed-target-probe.json",
            "artifact_sha256": probe_artifact_sha256,
        },
        "teacher_reference_by_modality": _stats_record(references),
        "training_batch_teacher_baseline_by_modality": _stats_record(training_teacher_baseline),
        "training_batch_prediction_baseline_by_modality": _stats_record(prediction_baseline),
        "thresholds": _DEFAULT_THRESHOLDS.to_dict(),
        "collapse_warmup_steps": collapse_warmup_steps,
    }
    calibration_sha256 = _write_new_canonical(destination / "calibration.json", calibration)

    provenance: dict[str, object] = {
        "schema": "simple-brats.short-real-matching",
        "schema_version": 2,
        "launch_sha": launch_sha,
        "manifest_sha256": manifest.sha256,
        "split_sha256": split.sha256,
        "case_grid_manifest_sha256": case_grids.sha256,
        "case_grid_policy_sha256": case_grids.policy.sha256,
        "runtime_extraction_policy_sha256": case_grids.policy.for_patch_config(config.patch).sha256,
        "config_sha256": config.sha256,
        "config_file_sha256": sha256_file(config_file),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
        "calibration_sha256": calibration_sha256,
        "fixed_target_probe": {
            "probe_sha256": collapse_probe.sha256,
            "artifact_sha256": probe_artifact_sha256,
            "source_split": "train",
            "subject_disjoint_from_optimization_cases": True,
            "case_ids": [case.case_id for case in probe_cases],
            "subject_ids": [case.subject_id for case in probe_cases],
            "sample_count_by_modality": {
                str(modality_id): count
                for modality_id, count in collapse_probe.sample_count_by_modality.items()
            },
        },
        "objective": "hard_symmetric_conditional_info_nce",
        "run_classification": classification,
        "tracking_mode": tracking_mode,
        "selected_train_case_ids": [case.case_id for case in cases],
        "selected_train_subject_ids": sorted({case.subject_id for case in cases}),
        "schedule": {
            "total_steps": total_steps,
            "max_cases": max_cases,
            "bags_per_case": bags_per_case,
            "absolute_step_factory": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
        },
    }
    _write_new_canonical(destination / "run-provenance.json", provenance)

    wandb_run = None
    if wandb_module is not None:
        wandb_run = wandb_module.init(
            project="simple-brats",
            name=destination.name,
            dir=str(destination),
            mode="offline",
            config=provenance,
            reinit=True,
        )
        if wandb_run is None:
            raise ShortRunError("offline W&B initialization returned no run")
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(system, weight_decay=weight_decay),
        lr=learning_rate,
    )
    checkpoint_manager = CheckpointManager(
        destination / "checkpoints",
        policy=CheckpointPolicy(
            checkpoint_every_steps=config.checkpoint_every_steps,
            artifact_every_steps=config.artifact_every_steps,
        ),
        artifact_sink=(WandbArtifactSink(wandb_run) if wandb_run is not None else None),
    )
    logger = _MetricsLogger(destination / "metrics.jsonl", factory, wandb_run)
    try:
        result = run_matching_training(
            system,
            optimizer,
            factory,
            checkpoint_manager,
            provenance,
            total_steps=total_steps,
            collapse_probe=collapse_probe,
            collapse_reference=references,
            collapse_thresholds=_DEFAULT_THRESHOLDS,
            collapse_warmup_steps=collapse_warmup_steps,
            gradient_clip_norm=gradient_clip_norm,
            on_step=logger,
        )
    finally:
        logger.close()
        if wandb_run is not None:
            wandb_run.finish()
    report: dict[str, object] = {
        "schema": "simple-brats.short-run-result",
        "schema_version": 1,
        "status": "ok",
        "run_classification": classification,
        "start_step": result.start_step,
        "end_step": result.end_step,
        "total_steps": result.total_steps,
        "ema_update_count": result.ema_update_count,
        "latest_checkpoint": (
            str(result.latest_checkpoint.relative_to(destination))
            if result.latest_checkpoint is not None
            else None
        ),
        "runner_contract_sha256": result.runner_contract_sha256,
        "provenance": provenance,
    }
    _write_new_canonical(destination / "result.json", report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run short pinned real-data hard matching")
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
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--bags-per-case", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--collapse-warmup-steps", type=int, default=25)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--max-plan-attempts", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_short_matching(
        data_root=args.data_root,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        split_path=args.split,
        expected_split_sha256=args.expected_split_sha256,
        case_grid_manifest_path=args.case_grid_manifest,
        expected_case_grid_manifest_sha256=args.expected_case_grid_manifest_sha256,
        config_path=args.config,
        output_dir=args.output_dir,
        expected_git_sha=args.expected_git_sha,
        repo_root=args.repo_root,
        total_steps=args.total_steps,
        max_cases=args.max_cases,
        bags_per_case=args.bags_per_case,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        collapse_warmup_steps=args.collapse_warmup_steps,
        candidate_pool_size=args.candidate_pool_size,
        max_plan_attempts=args.max_plan_attempts,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
