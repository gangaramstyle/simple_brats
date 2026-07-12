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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
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
_DEFAULT_THRESHOLDS = CollapseThresholds(
    minimum_variance_ratio=0.10,
    minimum_effective_rank_ratio=0.25,
    maximum_off_diagonal_cosine=0.95,
)


class ShortRunError(RuntimeError):
    """The short run cannot satisfy its pinned experiment contract."""


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
        if optimized_runtime is not None:
            self._case_prefetcher = ScheduleKeyedPrefetcher(
                self._load_case_state,
                workers=optimized_runtime.prefetch_workers,
                depth=optimized_runtime.prefetch_depth,
                thread_name_prefix="simple-brats-case",
            )
            if optimized_runtime.batched_gpu_extraction:
                self._gpu_case_cache = ByteBoundedGpuCaseCache(
                    byte_budget=optimized_runtime.gpu_cache_bytes
                )
        self._cached_step: int | None = None
        self._cached_batch: Any | None = None
        self.last_record: dict[str, object] | None = None

    def _load_case_state(self, case_index: int) -> _CaseSamplingState:
        if isinstance(case_index, bool) or not isinstance(case_index, int) or not (
            0 <= case_index < len(self.cases)
        ):
            raise ShortRunError("prefetched case index is outside the exact case table")
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

    def _activate(self, case_index: int) -> _CaseSamplingState:
        with self._case_cache_lock:
            cached = self._case_cache.get(case_index)
            if cached is not None:
                self._case_cache_hit_count += 1
                self._case_cache.move_to_end(case_index)
                return cached
            self._case_cache_miss_count += 1
        state = (
            self._case_prefetcher.get(case_index)
            if self._case_prefetcher is not None
            else self._load_case_state(case_index)
        )
        with self._case_cache_lock:
            raced = self._case_cache.get(case_index)
            if raced is not None:
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

    def prime(self, absolute_step_index: int) -> tuple[int, ...]:
        """Prefetch exact upcoming scheduled cases without choosing any sample."""

        if self._case_prefetcher is None or self.optimized_runtime is None:
            return ()
        ordered: list[int] = []
        with self._case_cache_lock:
            resident = set(self._case_cache)
        seen: set[int] = set(resident)
        # Depth is counted in distinct scheduled case activations.  Inspecting
        # at most depth full blocks is bounded even when consecutive steps share
        # the same case.
        stop = absolute_step_index + self.optimized_runtime.prefetch_depth * max(
            self.bags_per_case, 1
        )
        for step in range(absolute_step_index, stop):
            case_index = self._assignment_for_absolute_step(step).case_index
            if case_index not in seen:
                ordered.append(case_index)
                seen.add(case_index)
                if len(ordered) == self.optimized_runtime.prefetch_depth:
                    break
        return self._case_prefetcher.prime(ordered)

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
        }

    def runtime_stats(self) -> dict[str, object]:
        return {
            "case_prefetch": (
                self._case_prefetcher.to_dict() if self._case_prefetcher else None
            ),
            "host_case_cache": {
                "resident_case_count": len(self._case_cache),
                "hit_count": self._case_cache_hit_count,
                "miss_count": self._case_cache_miss_count,
                "eviction_count": self._case_cache_eviction_count,
            },
            "gpu_case_cache": (
                self._gpu_case_cache.to_dict() if self._gpu_case_cache is not None else None
            ),
        }

    def discard_prefetch(self) -> tuple[int, ...]:
        """Discard unused lookahead when no overlapping keys will be re-primed."""

        return (
            self._case_prefetcher.discard_pending()
            if self._case_prefetcher is not None
            else ()
        )

    def close(self) -> None:
        if self._case_prefetcher is not None:
            self._case_prefetcher.close(cancel_pending=True)

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
        case = self.cases[assignment.case_index]
        state = self._activate(assignment.case_index)
        prepared = materialize_case_matching_plan_record(
            state.extractor,
            case,
            state.candidate_universe,
            epoch=assignment.epoch,
            bag_index=assignment.bag_index,
            experiment_seed=self.config.seed,
            geometry=self.geometry,
            target_count=self.config.task.positions_per_bag,
            candidate_pool_size=self.candidate_pool_size,
            max_attempts=self.max_plan_attempts,
        )
        stem = f"step-{absolute_step_index + 1:09d}"
        plan_path = self.plans_dir / f"{stem}.plan.json"
        audit_path = self.plans_dir / f"{stem}.prepared.json"
        if self.replay_existing and plan_path.exists():
            _require_canonical(plan_path, prepared.plan.to_dict(), "replayed patch plan")
        else:
            save_patch_plan(prepared.plan, plan_path, overwrite=False)
        if self.replay_existing and audit_path.exists():
            _require_canonical(audit_path, prepared.to_dict(), "replayed prepared-plan audit")
            audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        else:
            audit_sha256 = _write_new_canonical(audit_path, prepared.to_dict())
        if audit_sha256 != prepared.sha256:
            raise ShortRunError("prepared plan audit does not match its canonical SHA")
        if self._gpu_case_cache is None:
            batch = assemble_matching_batch(
                case,
                prepared.plan,
                state.extractor,
                data_manifest_sha256=self.manifest.sha256,
                plan_sha256=prepared.plan.sha256,
                extraction_spec_sha256=state.extractor.extraction_spec_sha256,
            )
        else:
            assert self.optimized_device is not None
            volumes = state.extractor.canonical_volumes_for_case(case)
            gpu_case = self._gpu_case_cache.get_or_upload(
                case=case,
                extraction_spec=state.extractor.extraction_spec,
                canonical_volumes=volumes,
                candidate_universe=state.candidate_universe,
                device=self.optimized_device,
            )
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
    samples_per_modality_per_bag = config.task.positions_per_bag // len(config.task.modalities)
    if samples_per_modality_per_bag <= 0:
        raise ShortRunError("probe bags do not contain every modality")
    bags_per_case = math.ceil(
        _MIN_FIXED_PROBE_SAMPLES_PER_MODALITY / (len(cases) * samples_per_modality_per_bag)
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

    def close(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def __call__(self, metrics: StepMetrics) -> None:
        if self._factory.last_record is None:
            raise ShortRunError("step metrics arrived without a batch provenance record")
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
            "batch": self._factory.last_record,
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
            flat: dict[str, float] = {
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
            self._wandb_run.log(flat, step=metrics.step)


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
