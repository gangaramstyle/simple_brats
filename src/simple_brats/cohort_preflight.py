"""Restartable cold-path validation of every scheduled training case.

This preflight deliberately performs no model forward pass.  It validates the
expensive data path that a long run would otherwise discover lazily: every
training visit is prepared from a cold four-modality cache, its strict
foreground candidate lattice is constructed, the visit's first real
subject-balanced bag is materialized, and the complete matching batch is
extracted and validated.  Validation and test images and segmentation files
are never supplied to an image-loading API.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.config import (
    MODALITIES,
    ExperimentConfig,
    ModelConfig,
    PatchConfig,
    TaskConfig,
    load_experiment_config,
)
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
    materialize_case_matching_plan_record,
    prepare_case_candidate_universe,
)
from simple_brats.data.real_batches import assemble_matching_batch
from simple_brats.data.splits import (
    SplitManifest,
    cases_for_splits,
    load_split,
    validate_split,
)
from simple_brats.long_run import SubjectBalancedSchedule
from simple_brats.provenance import verify_git_sha
from simple_brats.sampling import MaterializedPatchPlan, SlabGeometry
from simple_brats.training.matching import MatchingBatch

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESULT_NAME = re.compile(r"^case-([0-9]{4})\.json$")

DEFAULT_EXPECTED_TRAIN_CASES = 1_044
DEFAULT_EXPECTED_TRAIN_SUBJECTS = 643
DEFAULT_BAGS_PER_SUBJECT = 8
TARGET_COUNT = 32
SOURCE_COUNT = 96
CANDIDATE_POOL_SIZE = 512
MAX_PLAN_ATTEMPTS = 8
WALLTIME_REQUEUE_EXIT_CODE = 75

REGISTERED_CONFIG = ExperimentConfig(
    seed=0,
    checkpoint_every_steps=1_000,
    artifact_every_steps=5_000,
    patch=PatchConfig(
        footprint_mm=4.0,
        thin_mm=4.0,
        tensor_shape=(8, 8, 8),
    ),
    model=ModelConfig(
        width=256,
        depth=8,
        heads=4,
        mlp_ratio=4.0,
        predictor_depth=1,
        teacher_ema_momentum=0.996,
    ),
    task=TaskConfig(
        modalities=MODALITIES,
        prism_extent_mm=(32.0, 32.0, 32.0),
        target_patches_per_bag=TARGET_COUNT,
        context_patches_per_nontarget_modality=30,
        context_patches_target_modality=6,
        objective="match",
        allow_target_modality_elsewhere=True,
        allow_target_modality_at_target=False,
        pass_scan_statistics_to_teacher=False,
    ),
)
REGISTERED_CONFIG_8MM = ExperimentConfig(
    seed=0,
    checkpoint_every_steps=1_000,
    artifact_every_steps=5_000,
    patch=PatchConfig(
        footprint_mm=8.0,
        thin_mm=8.0,
        tensor_shape=(8, 8, 8),
    ),
    model=REGISTERED_CONFIG.model,
    task=TaskConfig(
        modalities=MODALITIES,
        prism_extent_mm=(64.0, 64.0, 64.0),
        target_patches_per_bag=TARGET_COUNT,
        context_patches_per_nontarget_modality=30,
        context_patches_target_modality=6,
        objective="match",
        allow_target_modality_elsewhere=True,
        allow_target_modality_at_target=False,
        pass_scan_statistics_to_teacher=False,
    ),
)
REGISTERED_CONFIG_SHA256 = "a261de64b08e19390a952a1d151066a10540acea55859d661cd0293848fd6bd3"
REGISTERED_CONFIG_8MM_SHA256 = "7ce7024c902e33878f019c1eac963d9c1e4da085261c9402b32123656d92a3bf"
_REGISTERED_CONFIG_BY_SHA = {
    REGISTERED_CONFIG_SHA256: REGISTERED_CONFIG,
    REGISTERED_CONFIG_8MM_SHA256: REGISTERED_CONFIG_8MM,
}
if REGISTERED_CONFIG.sha256 != REGISTERED_CONFIG_SHA256:  # pragma: no cover
    raise AssertionError("registered preflight config digest is internally inconsistent")
if REGISTERED_CONFIG_8MM.sha256 != REGISTERED_CONFIG_8MM_SHA256:  # pragma: no cover
    raise AssertionError("registered 8 mm preflight config digest is internally inconsistent")

_ACCESS_BOUNDARY = {
    "image_split": "train_only",
    "image_modalities": list(MODALITIES),
    "segmentation_image_access": False,
    "validation_image_access": False,
    "test_image_access": False,
    "manifest_and_split_metadata_for_all_partitions_validated": True,
}
_TENSOR_DTYPES = {
    "source_patches": "<f4",
    "source_modality_ids": "<i8",
    "source_position_ids": "<i8",
    "source_coordinates_mm": "<f4",
    "query_modality_ids": "<i8",
    "query_position_ids": "<i8",
    "query_coordinates_mm": "<f4",
    "query_bag_ids": "<i8",
    "query_pair_ids": "<i8",
    "target_patches": "<f4",
    "target_modality_ids": "<i8",
    "target_position_ids": "<i8",
    "target_coordinates_mm": "<f4",
    "target_bag_ids": "<i8",
    "target_pair_ids": "<i8",
    "anchor_mm": "<f4",
}
_TENSOR_SHAPES = {
    "source_patches": [1, 96, 8, 8, 8],
    "source_modality_ids": [1, 96],
    "source_position_ids": [1, 96],
    "source_coordinates_mm": [1, 96, 3],
    "query_modality_ids": [1, 32],
    "query_position_ids": [1, 32],
    "query_coordinates_mm": [1, 32, 3],
    "query_bag_ids": [1, 32],
    "query_pair_ids": [1, 32],
    "target_patches": [1, 32, 8, 8, 8],
    "target_modality_ids": [1, 32],
    "target_position_ids": [1, 32],
    "target_coordinates_mm": [1, 32, 3],
    "target_bag_ids": [1, 32],
    "target_pair_ids": [1, 32],
    "anchor_mm": [1, 3],
}
_TIMING_PHASES = (
    "candidate_universe",
    "plan_materialization",
    "batch_assembly",
    "total",
)


class CohortPreflightError(RuntimeError):
    """The full-cohort data path did not satisfy its immutable contract."""


@dataclass(frozen=True, slots=True)
class FirstOccurrence:
    """First bag-zero assignment of one visit in the long-run schedule."""

    absolute_step_index: int
    completed_step: int
    subject_epoch: int
    subject_position: int
    visit_rotation_index: int
    bag_index: int
    case_index: int

    def to_dict(self) -> dict[str, int]:
        return {
            "absolute_step_index": self.absolute_step_index,
            "completed_step": self.completed_step,
            "subject_epoch": self.subject_epoch,
            "subject_position": self.subject_position,
            "visit_rotation_index": self.visit_rotation_index,
            "bag_index": self.bag_index,
            "case_index": self.case_index,
        }


class WalltimeStop:
    """Signal-safe request checked only between atomically completed cases."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handle(self, signal_number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = signal_number

    def __call__(self) -> bool:
        return self.requested


def _case_key(case: CaseRecord) -> tuple[str, str, str, str, str]:
    return (case.source, case.release, case.subject_id, case.visit_id, case.case_id)


def derive_first_occurrences(
    schedule: SubjectBalancedSchedule,
) -> dict[tuple[str, str, str, str, str], FirstOccurrence]:
    """Derive every case's earliest real bag-zero assignment without a cursor."""

    if not isinstance(schedule, SubjectBalancedSchedule):
        raise TypeError("schedule must be a SubjectBalancedSchedule")
    expected_keys = {_case_key(case) for case in schedule.cases}
    if len(expected_keys) != schedule.case_count:
        raise CohortPreflightError("subject schedule contains duplicate case identities")
    occurrences: dict[tuple[str, str, str, str, str], FirstOccurrence] = {}
    for subject_epoch in range(schedule.maximum_visits_per_subject):
        epoch_start = subject_epoch * schedule.steps_per_subject_epoch
        for subject_position in range(schedule.subject_count):
            absolute_step_index = epoch_start + subject_position * schedule.bags_per_subject
            assignment = schedule.assignment_for_step(absolute_step_index)
            if assignment.bag_index != 0 or assignment.subject_epoch != subject_epoch:
                raise CohortPreflightError("schedule did not emit the expected bag-zero block")
            case = schedule.cases[assignment.case_index]
            key = _case_key(case)
            occurrences.setdefault(
                key,
                FirstOccurrence(
                    absolute_step_index=absolute_step_index,
                    completed_step=absolute_step_index + 1,
                    subject_epoch=assignment.subject_epoch,
                    subject_position=assignment.subject_position,
                    visit_rotation_index=assignment.visit_rotation_index,
                    bag_index=assignment.bag_index,
                    case_index=assignment.case_index,
                ),
            )
    if set(occurrences) != expected_keys:
        missing = sorted(expected_keys - set(occurrences))
        raise CohortPreflightError(
            f"subject schedule never exposes {len(missing)} training cases: {missing[:3]}"
        )
    return occurrences


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CohortPreflightError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CohortPreflightError(
            f"invalid {name} keys: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _decode_canonical_mapping(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CohortPreflightError(f"{description} must be a regular non-symlink file: {path}")
    raw = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CohortPreflightError(f"duplicate JSON key {key!r} in {description}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> object:
        raise CohortPreflightError(f"non-finite JSON number {token!r} in {description}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except CohortPreflightError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CohortPreflightError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise CohortPreflightError(f"{description} must contain one JSON object")
    if raw != canonical_json_bytes(value):
        raise CohortPreflightError(f"{description} is not in canonical byte form")
    return value


def _resolve_file(path: str | os.PathLike[str], description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise CohortPreflightError(f"{description} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise CohortPreflightError(f"{description} is unavailable: {path}") from error
    if not resolved.is_file():
        raise CohortPreflightError(f"{description} must be a regular file")
    return resolved


def _require_canonical_file(path: Path, value: Mapping[str, object], description: str) -> None:
    if path.read_bytes() != canonical_json_bytes(value):
        raise CohortPreflightError(f"{description} is not in canonical byte form")


def _git_output(repo: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _verify_pinned_detached_launch(
    expected_git_sha: str,
    repo_root: str | os.PathLike[str],
    config_file: Path,
) -> tuple[Path, str, str]:
    if _FULL_GIT_SHA.fullmatch(expected_git_sha) is None:
        raise ValueError("expected_git_sha must be one full lowercase commit ID")
    repo = Path(repo_root).expanduser().resolve(strict=True)
    launch_sha = verify_git_sha(expected_git_sha, repo)
    symbolic = _git_output(repo, ("symbolic-ref", "-q", "HEAD"))
    if symbolic.returncode == 0:
        raise CohortPreflightError("runtime launch checkout must be detached at LAUNCH_SHA")
    if symbolic.returncode != 1:
        raise CohortPreflightError(f"unable to verify detached git state: {symbolic.stderr}")
    status = _git_output(repo, ("status", "--porcelain", "--untracked-files=no"))
    if status.returncode != 0 or status.stdout:
        raise CohortPreflightError("runtime launch tree contains modified tracked files")
    try:
        relative_config = config_file.relative_to(repo).as_posix()
    except ValueError as error:
        raise CohortPreflightError(
            "experiment config must belong to the pinned repository"
        ) from error
    tracked = _git_output(repo, ("ls-files", "--error-unmatch", "--", relative_config))
    if tracked.returncode != 0:
        raise CohortPreflightError("experiment config is not tracked at LAUNCH_SHA")
    lock_file = _resolve_file(repo / "uv.lock", "uv lock")
    return repo, launch_sha, sha256_file(lock_file)


def _publish_or_require(path: Path, value: Mapping[str, object], description: str) -> str:
    payload = canonical_json_bytes(value)
    if os.path.lexists(path):
        _decode_canonical_mapping(path, description)
        if path.read_bytes() != payload:
            raise CohortPreflightError(f"existing {description} conflicts with this launch")
    else:
        try:
            atomic_create_bytes(path, payload)
        except FileExistsError:
            _decode_canonical_mapping(path, description)
            if path.read_bytes() != payload:
                raise CohortPreflightError(
                    f"racing {description} conflicts with this launch"
                ) from None
    if path.read_bytes() != payload:
        raise CohortPreflightError(f"{description} changed after atomic publication")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _exclusive_output_lock(destination: Path) -> Iterator[None]:
    lock_path = destination / ".cohort-preflight.lock"
    if lock_path.is_symlink():
        raise CohortPreflightError("preflight lock must not be a symlink")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CohortPreflightError(f"another process is already using {destination}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _initialize_output(output_dir: str | os.PathLike[str]) -> Path:
    requested = Path(output_dir).expanduser()
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise CohortPreflightError("output must be a non-symlink directory")
    else:
        destination.mkdir(mode=0o700)
    cases_dir = destination / "cases"
    if os.path.lexists(cases_dir):
        if cases_dir.is_symlink() or not cases_dir.is_dir():
            raise CohortPreflightError("cases output must be a non-symlink directory")
    else:
        cases_dir.mkdir(mode=0o700)
    return destination


def _remove_stale_atomic_temporaries(destination: Path) -> None:
    for directory in (destination, destination / "cases"):
        for path in directory.glob(".*.tmp-*"):
            if path.is_symlink() or not path.is_file():
                raise CohortPreflightError(f"unsafe stale atomic temporary: {path}")
            path.unlink()


def _tensor_audit(tensor: torch.Tensor) -> dict[str, object]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("batch tensor audit requires torch.Tensor values")
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise CohortPreflightError("matching batch contains non-finite tensor values")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.array(array, dtype=canonical_dtype, order="C", copy=True)
    if np.issubdtype(canonical.dtype, np.floating):
        canonical[canonical == 0] = 0.0
    metadata = {
        "schema": "simple-brats.preflight-batch-tensor",
        "schema_version": 1,
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return {**metadata, "sha256": digest.hexdigest()}


def _batch_audit(batch: MatchingBatch) -> dict[str, object]:
    if not isinstance(batch, MatchingBatch):
        raise TypeError("batch must be a MatchingBatch")
    tensors: dict[str, object] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if field.name == "source_padding_mask":
            if value is not None:
                raise CohortPreflightError("registered preflight batch must not require padding")
            continue
        if not isinstance(value, torch.Tensor):
            raise CohortPreflightError(f"matching batch field {field.name} is not a tensor")
        tensors[field.name] = _tensor_audit(value)
    digest_payload = {
        "schema": "simple-brats.preflight-matching-batch",
        "schema_version": 1,
        "source_padding_mask": None,
        "tensors": tensors,
    }
    return {
        **digest_payload,
        "sha256": hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest(),
    }


def _case_identity(case: CaseRecord) -> dict[str, str]:
    return {
        "source": case.source,
        "release": case.release,
        "case_id": case.case_id,
        "subject_id": case.subject_id,
        "visit_id": case.visit_id,
    }


def _process_case(
    *,
    data_root: str | os.PathLike[str],
    manifest: DatasetManifest,
    case_grids: CaseGridManifest,
    config: ExperimentConfig,
    case: CaseRecord,
    occurrence: FirstOccurrence,
    contract_sha256: str,
) -> dict[str, object]:
    """Run one cold train-only preparation, plan, and complete extraction."""

    total_start = time.perf_counter()
    geometry = SlabGeometry(
        in_plane_footprint_mm=config.patch.footprint_mm,
        thin_extent_mm=config.patch.thin_mm,
        model_shape=config.patch.tensor_shape,
    )
    extraction_spec = case_grids.extraction_spec_for_case(
        case,
        patch_config=config.patch,
    )
    extractor = CachedNiftiPatchExtractor(
        data_root=data_root,
        manifest=manifest,
        data_manifest_sha256=manifest.sha256,
        extraction_spec=extraction_spec,
        max_cached_volumes=4,
    )

    candidate_start = time.perf_counter()
    candidate_universe = prepare_case_candidate_universe(
        extractor,
        case,
        geometry=geometry,
    )
    candidate_seconds = time.perf_counter() - candidate_start
    if candidate_universe.candidate_count < TARGET_COUNT:
        raise CohortPreflightError(
            f"{case.case_id} has only {candidate_universe.candidate_count} strict safe centers; "
            f"at least {TARGET_COUNT} are required"
        )

    plan_start = time.perf_counter()
    prepared = materialize_case_matching_plan_record(
        extractor,
        case,
        candidate_universe,
        epoch=occurrence.subject_epoch,
        bag_index=0,
        experiment_seed=config.seed,
        geometry=geometry,
        prism_extent_mm=config.task.prism_extent_mm,
        target_count=TARGET_COUNT,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        max_attempts=MAX_PLAN_ATTEMPTS,
    )
    plan_seconds = time.perf_counter() - plan_start

    batch_start = time.perf_counter()
    batch = assemble_matching_batch(
        case,
        prepared.plan,
        extractor,
        data_manifest_sha256=manifest.sha256,
        plan_sha256=prepared.plan.sha256,
        extraction_spec_sha256=extractor.extraction_spec_sha256,
    )
    batch_record = _batch_audit(batch)
    batch_seconds = time.perf_counter() - batch_start
    timings = {
        "candidate_universe": candidate_seconds,
        "plan_materialization": plan_seconds,
        "batch_assembly": batch_seconds,
        "total": time.perf_counter() - total_start,
    }
    result = {
        "schema": "simple-brats.cohort-cold-path-case",
        "schema_version": 1,
        "status": "passed",
        "contract_sha256": contract_sha256,
        "case": _case_identity(case),
        "schedule_first_occurrence": occurrence.to_dict(),
        "candidate_universe": {
            "candidate_count": candidate_universe.candidate_count,
            "candidate_centers_sha256": candidate_universe.candidate_centers_sha256,
            "geometry_sha256": candidate_universe.geometry_sha256,
            "extraction_spec_sha256": candidate_universe.extraction_spec_sha256,
            "volume_digests": [item.to_dict() for item in candidate_universe.volume_digests],
        },
        "prepared_plan": prepared.to_dict(),
        "materialized_plan": prepared.plan.to_dict(),
        "batch": batch_record,
        "timings_seconds": timings,
        "access_boundary": dict(_ACCESS_BOUNDARY),
    }
    _validate_case_result(
        result,
        case=case,
        occurrence=occurrence,
        contract_sha256=contract_sha256,
        data_manifest_sha256=manifest.sha256,
    )
    return result


def _validate_volume_digests(value: object, case: CaseRecord) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(MODALITIES):
        raise CohortPreflightError("case result must contain four MRI volume digests")
    files = {item.modality: item for item in case.files}
    records: list[dict[str, object]] = []
    for expected_modality, raw in zip(MODALITIES, value, strict=True):
        if not isinstance(raw, dict):
            raise CohortPreflightError("volume digest entries must be JSON objects")
        _exact_keys(
            raw,
            {
                "modality",
                "raw_file_sha256",
                "canonical_voxel_sha256",
                "normalized_voxel_sha256",
            },
            "volume digest",
        )
        if raw["modality"] != expected_modality:
            raise CohortPreflightError("volume digests are not in canonical MRI modality order")
        if files[expected_modality].sha256 != raw["raw_file_sha256"]:
            raise CohortPreflightError("volume digest does not match the train case manifest")
        for name in (
            "raw_file_sha256",
            "canonical_voxel_sha256",
            "normalized_voxel_sha256",
        ):
            _require_sha256(raw[name], name)
        records.append(raw)
    return records


def _validate_batch_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CohortPreflightError("batch audit must be a JSON object")
    _exact_keys(
        value,
        {"schema", "schema_version", "source_padding_mask", "tensors", "sha256"},
        "batch audit",
    )
    if (
        value["schema"] != "simple-brats.preflight-matching-batch"
        or value["schema_version"] != 1
        or value["source_padding_mask"] is not None
    ):
        raise CohortPreflightError("batch audit schema or padding contract changed")
    tensors = value["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != set(_TENSOR_DTYPES):
        raise CohortPreflightError("batch audit tensor table is incomplete")
    for name, raw in tensors.items():
        if not isinstance(raw, dict):
            raise CohortPreflightError("batch tensor audits must be JSON objects")
        _exact_keys(
            raw,
            {"schema", "schema_version", "shape", "dtype", "order", "sha256"},
            "batch tensor audit",
        )
        if (
            raw["schema"] != "simple-brats.preflight-batch-tensor"
            or raw["schema_version"] != 1
            or raw["shape"] != _TENSOR_SHAPES[name]
            or raw["dtype"] != _TENSOR_DTYPES[name]
            or raw["order"] != "C"
        ):
            raise CohortPreflightError(f"batch tensor contract changed for {name}")
        _require_sha256(raw["sha256"], f"batch.{name}.sha256")
    payload = {
        "schema": value["schema"],
        "schema_version": value["schema_version"],
        "source_padding_mask": value["source_padding_mask"],
        "tensors": tensors,
    }
    expected_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if value["sha256"] != expected_sha:
        raise CohortPreflightError("batch audit aggregate digest is inconsistent")
    return value


def _validate_case_result(
    value: Mapping[str, object],
    *,
    case: CaseRecord,
    occurrence: FirstOccurrence,
    contract_sha256: str,
    data_manifest_sha256: str,
) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "schema_version",
            "status",
            "contract_sha256",
            "case",
            "schedule_first_occurrence",
            "candidate_universe",
            "prepared_plan",
            "materialized_plan",
            "batch",
            "timings_seconds",
            "access_boundary",
        },
        "per-case preflight result",
    )
    if (
        value["schema"] != "simple-brats.cohort-cold-path-case"
        or value["schema_version"] != 1
        or value["status"] != "passed"
        or value["contract_sha256"] != contract_sha256
        or value["case"] != _case_identity(case)
        or value["schedule_first_occurrence"] != occurrence.to_dict()
        or value["access_boundary"] != _ACCESS_BOUNDARY
    ):
        raise CohortPreflightError("per-case result does not match this immutable assignment")

    universe = value["candidate_universe"]
    if not isinstance(universe, dict):
        raise CohortPreflightError("candidate universe audit must be a JSON object")
    _exact_keys(
        universe,
        {
            "candidate_count",
            "candidate_centers_sha256",
            "geometry_sha256",
            "extraction_spec_sha256",
            "volume_digests",
        },
        "candidate universe audit",
    )
    candidate_count = universe["candidate_count"]
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < TARGET_COUNT
    ):
        raise CohortPreflightError("candidate universe has fewer than 32 strict safe centers")
    for name in (
        "candidate_centers_sha256",
        "geometry_sha256",
        "extraction_spec_sha256",
    ):
        _require_sha256(universe[name], name)
    volume_digests = _validate_volume_digests(universe["volume_digests"], case)

    raw_plan = value["materialized_plan"]
    if not isinstance(raw_plan, dict):
        raise CohortPreflightError("materialized plan must be a JSON object")
    try:
        plan = MaterializedPatchPlan.from_dict(raw_plan)
    except Exception as error:
        raise CohortPreflightError(f"invalid materialized plan: {error}") from error
    if (
        plan.case.to_dict() != _case_identity(case)
        or plan.epoch != occurrence.subject_epoch
        or plan.bag_index != 0
        or plan.data_manifest_sha256 != data_manifest_sha256
        or plan.extraction_spec_sha256 != universe["extraction_spec_sha256"]
        or len(plan.targets) != TARGET_COUNT
        or len(plan.sources) != SOURCE_COUNT
    ):
        raise CohortPreflightError("materialized plan is not the scheduled 32-position bag zero")

    prepared = value["prepared_plan"]
    if not isinstance(prepared, dict):
        raise CohortPreflightError("prepared-plan audit must be a JSON object")
    _exact_keys(
        prepared,
        {
            "schema",
            "schema_version",
            "case",
            "data_manifest_sha256",
            "extraction_spec_sha256",
            "plan_sha256",
            "candidate_count",
            "candidate_centers_sha256",
            "volume_digests",
        },
        "prepared-plan audit",
    )
    if (
        prepared["schema"] != "simple-brats.prepared-case-plan"
        or prepared["schema_version"] != 1
        or prepared["case"] != _case_identity(case)
        or prepared["data_manifest_sha256"] != plan.data_manifest_sha256
        or prepared["extraction_spec_sha256"] != plan.extraction_spec_sha256
        or prepared["plan_sha256"] != plan.sha256
        or prepared["candidate_count"] != candidate_count
        or prepared["candidate_centers_sha256"] != universe["candidate_centers_sha256"]
        or prepared["volume_digests"] != volume_digests
    ):
        raise CohortPreflightError("prepared-plan audit is inconsistent with its plan/universe")

    _validate_batch_record(value["batch"])
    timings = value["timings_seconds"]
    if not isinstance(timings, dict) or set(timings) != set(_TIMING_PHASES):
        raise CohortPreflightError("case timing audit is incomplete")
    for name, timing in timings.items():
        if (
            isinstance(timing, bool)
            or not isinstance(timing, (int, float))
            or not math.isfinite(float(timing))
            or timing < 0
        ):
            raise CohortPreflightError(f"case timing {name!r} must be finite and non-negative")
    if float(timings["total"]) < math.fsum(
        float(timings[name]) for name in _TIMING_PHASES if name != "total"
    ):
        raise CohortPreflightError("total case timing is smaller than its measured phases")


def _case_result_path(cases_dir: Path, case_index: int) -> Path:
    return cases_dir / f"case-{case_index:04d}.json"


def _load_case_result(
    path: Path,
    *,
    case: CaseRecord,
    occurrence: FirstOccurrence,
    contract_sha256: str,
    data_manifest_sha256: str,
) -> dict[str, object]:
    value = _decode_canonical_mapping(path, "per-case preflight result")
    _validate_case_result(
        value,
        case=case,
        occurrence=occurrence,
        contract_sha256=contract_sha256,
        data_manifest_sha256=data_manifest_sha256,
    )
    return value


def _nearest_rank_summary(values: Sequence[int | float]) -> dict[str, object]:
    if not values:
        raise ValueError("summary requires at least one value")
    ordered = sorted(values)
    quantiles: dict[str, int | float] = {}
    for label, percentile in (
        ("p00", 0),
        ("p01", 1),
        ("p05", 5),
        ("p25", 25),
        ("p50", 50),
        ("p75", 75),
        ("p95", 95),
        ("p99", 99),
        ("p100", 100),
    ):
        index = 0 if percentile == 0 else math.ceil(percentile * len(ordered) / 100) - 1
        quantiles[label] = ordered[index]
    total = math.fsum(float(item) for item in ordered)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": total / len(ordered),
        "sum": total,
        "quantile_method": "nearest_rank_v1",
        "quantiles": quantiles,
    }


def _digest_records(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _build_aggregate(
    *,
    contract: Mapping[str, object],
    contract_sha256: str,
    schedule: SubjectBalancedSchedule,
    case_results: Sequence[tuple[CaseRecord, FirstOccurrence, Path, Mapping[str, object]]],
) -> dict[str, object]:
    if len(case_results) != schedule.case_count:
        raise CohortPreflightError("cannot publish aggregate before every training case passes")
    ordered = sorted(case_results, key=lambda item: item[1].case_index)
    if [item[1].case_index for item in ordered] != list(range(schedule.case_count)):
        raise CohortPreflightError("aggregate case-index coverage is not exact")
    subjects = sorted({case.subject_id for case, _, _, _ in ordered})
    if len(subjects) != schedule.subject_count:
        raise CohortPreflightError("aggregate subject coverage is not exact")

    cases: list[dict[str, object]] = []
    candidate_digest_table: list[dict[str, object]] = []
    plan_digest_table: list[dict[str, object]] = []
    batch_digest_table: list[dict[str, object]] = []
    volume_digest_table: list[dict[str, object]] = []
    candidate_counts: list[int] = []
    timing_values: dict[str, list[float]] = {name: [] for name in _TIMING_PHASES}
    for case, occurrence, path, result in ordered:
        payload = canonical_json_bytes(result)
        result_sha = hashlib.sha256(payload).hexdigest()
        universe = result["candidate_universe"]
        plan = result["materialized_plan"]
        batch = result["batch"]
        timings = result["timings_seconds"]
        assert isinstance(universe, Mapping)
        assert isinstance(plan, Mapping)
        assert isinstance(batch, Mapping)
        assert isinstance(timings, Mapping)
        candidate_count = universe["candidate_count"]
        assert isinstance(candidate_count, int)
        candidate_counts.append(candidate_count)
        for name in _TIMING_PHASES:
            timing_values[name].append(float(timings[name]))  # type: ignore[arg-type]
        cases.append(
            {
                **_case_identity(case),
                "case_index": occurrence.case_index,
                "first_absolute_step_index": occurrence.absolute_step_index,
                "first_subject_epoch": occurrence.subject_epoch,
                "result_file": path.name,
                "result_sha256": result_sha,
                "candidate_count": candidate_count,
                "candidate_centers_sha256": universe["candidate_centers_sha256"],
                "plan_sha256": plan["payload_sha256"],
                "batch_sha256": batch["sha256"],
            }
        )
        candidate_digest_table.append(
            {
                "case_index": occurrence.case_index,
                "count": candidate_count,
                "sha256": universe["candidate_centers_sha256"],
            }
        )
        plan_digest_table.append(
            {"case_index": occurrence.case_index, "sha256": plan["payload_sha256"]}
        )
        batch_digest_table.append({"case_index": occurrence.case_index, "sha256": batch["sha256"]})
        volume_digest_table.append(
            {
                "case_index": occurrence.case_index,
                "volume_digests": universe["volume_digests"],
            }
        )

    result_digest_table = [
        {"case_index": item["case_index"], "sha256": item["result_sha256"]} for item in cases
    ]
    aggregate = {
        "schema": "simple-brats.cohort-cold-path-preflight",
        "schema_version": 1,
        "status": "passed",
        "contract_sha256": contract_sha256,
        "launch_sha": contract["launch_sha"],
        "schedule_sha256": schedule.sha256,
        "passed_case_count": len(cases),
        "passed_subject_count": len(subjects),
        "passed_subject_ids": subjects,
        "passed_subject_ids_sha256": _digest_records(subjects),
        "candidate_count_summary": _nearest_rank_summary(candidate_counts),
        "timing_seconds_summary": {
            name: _nearest_rank_summary(values) for name, values in timing_values.items()
        },
        "digest_sets": {
            "case_results_sha256": _digest_records(result_digest_table),
            "candidate_universes_sha256": _digest_records(candidate_digest_table),
            "materialized_plans_sha256": _digest_records(plan_digest_table),
            "matching_batches_sha256": _digest_records(batch_digest_table),
            "prepared_volumes_sha256": _digest_records(volume_digest_table),
        },
        "cases": cases,
        "access_boundary": dict(_ACCESS_BOUNDARY),
    }
    return aggregate


def _build_contract(
    *,
    launch_sha: str,
    lock_sha256: str,
    manifest: DatasetManifest,
    split: SplitManifest,
    case_grids: CaseGridManifest,
    config: ExperimentConfig,
    config_file_sha256: str,
    schedule: SubjectBalancedSchedule,
) -> dict[str, object]:
    return {
        "schema": "simple-brats.cohort-cold-path-contract",
        "schema_version": 1,
        "launch_sha": launch_sha,
        "uv_lock_sha256": lock_sha256,
        "data_manifest_sha256": manifest.sha256,
        "split_manifest_sha256": split.sha256,
        "case_grid_manifest_sha256": case_grids.sha256,
        "config_sha256": config.sha256,
        "config_file_sha256": config_file_sha256,
        "config": config.to_dict(),
        "schedule_algorithm": schedule.algorithm,
        "schedule_sha256": schedule.sha256,
        "bags_per_subject": schedule.bags_per_subject,
        "expected_train_cases": schedule.case_count,
        "expected_train_subjects": schedule.subject_count,
        "target_count": TARGET_COUNT,
        "source_count": config.task.source_patches_per_bag,
        "prism_extent_mm": list(config.task.prism_extent_mm),
        "context_patches_per_nontarget_modality": (
            config.task.context_patches_per_nontarget_modality
        ),
        "context_patches_target_modality": config.task.context_patches_target_modality,
        "registered_single_d_arm": config.registered_single_d_arm,
        "minimum_safe_candidate_centers": TARGET_COUNT,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "max_plan_attempts": MAX_PLAN_ATTEMPTS,
        "plan_assignment": "first_subject_balanced_occurrence_bag_zero",
        "case_cache_policy": "new_four-volume_extractor_per_case_cold_path",
        "patch_source_shape": list(config.patch.source_shape),
        "patch_physical_extent_mm": list(config.patch.physical_extent_mm),
        "model_visible_shape": list(config.patch.tensor_shape),
        "access_boundary": dict(_ACCESS_BOUNDARY),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def run_cohort_preflight(
    *,
    data_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    expected_manifest_sha256: str,
    split_path: str | os.PathLike[str],
    expected_split_sha256: str,
    case_grid_manifest_path: str | os.PathLike[str],
    expected_case_grid_manifest_sha256: str,
    config_path: str | os.PathLike[str],
    expected_config_sha256: str,
    output_dir: str | os.PathLike[str],
    expected_git_sha: str,
    repo_root: str | os.PathLike[str] = ".",
    expected_train_cases: int = DEFAULT_EXPECTED_TRAIN_CASES,
    expected_train_subjects: int = DEFAULT_EXPECTED_TRAIN_SUBJECTS,
    bags_per_subject: int = DEFAULT_BAGS_PER_SUBJECT,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Validate every train visit, resuming from immutable per-case results."""

    for value, name in (
        (expected_manifest_sha256, "expected_manifest_sha256"),
        (expected_split_sha256, "expected_split_sha256"),
        (expected_case_grid_manifest_sha256, "expected_case_grid_manifest_sha256"),
        (expected_config_sha256, "expected_config_sha256"),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if expected_config_sha256 not in _REGISTERED_CONFIG_BY_SHA:
        raise CohortPreflightError("preflight config pin is not an exact registered scale arm")
    for value, name in (
        (expected_train_cases, "expected_train_cases"),
        (expected_train_subjects, "expected_train_subjects"),
        (bags_per_subject, "bags_per_subject"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if bags_per_subject != DEFAULT_BAGS_PER_SUBJECT:
        raise CohortPreflightError("preflight requires the registered eight-bag subject schedule")

    manifest_file = _resolve_file(manifest_path, "filtered manifest")
    split_file = _resolve_file(split_path, "subject split")
    grids_file = _resolve_file(case_grid_manifest_path, "case-grid manifest")
    config_file = _resolve_file(config_path, "experiment config")
    _repo, launch_sha, lock_sha256 = _verify_pinned_detached_launch(
        expected_git_sha,
        repo_root,
        config_file,
    )
    manifest = load_manifest(manifest_file, expected_sha256=expected_manifest_sha256)
    split = load_split(split_file, expected_sha256=expected_split_sha256)
    case_grids = load_case_grid_manifest(
        grids_file,
        expected_sha256=expected_case_grid_manifest_sha256,
    )
    config = load_experiment_config(config_file)
    if (
        config != _REGISTERED_CONFIG_BY_SHA[expected_config_sha256]
        or config.sha256 != expected_config_sha256
        or config.registered_single_d_arm is None
    ):
        raise CohortPreflightError("preflight is locked to an exact registered scale arm")
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    _require_canonical_file(manifest_file, manifest.to_dict(), "filtered manifest")
    _require_canonical_file(split_file, split.to_dict(), "subject split")

    train_cases = cases_for_splits(manifest, split, ("train",))
    schedule = SubjectBalancedSchedule(
        train_cases,
        seed=config.seed,
        bags_per_subject=bags_per_subject,
    )
    if schedule.case_count != expected_train_cases:
        raise CohortPreflightError(
            f"locked train case count is {expected_train_cases}, observed {schedule.case_count}"
        )
    if schedule.subject_count != expected_train_subjects:
        raise CohortPreflightError(
            "locked train subject count is "
            f"{expected_train_subjects}, observed {schedule.subject_count}"
        )
    occurrences = derive_first_occurrences(schedule)
    contract = _build_contract(
        launch_sha=launch_sha,
        lock_sha256=lock_sha256,
        manifest=manifest,
        split=split,
        case_grids=case_grids,
        config=config,
        config_file_sha256=sha256_file(config_file),
        schedule=schedule,
    )
    destination = _initialize_output(output_dir)
    cases_dir = destination / "cases"

    with _exclusive_output_lock(destination):
        _remove_stale_atomic_temporaries(destination)
        contract_sha256 = _publish_or_require(
            destination / "run-contract.json",
            contract,
            "preflight run contract",
        )
        expected_case_paths = {
            _case_result_path(cases_dir, index).name for index in range(schedule.case_count)
        }
        unexpected = sorted(
            path.name
            for path in cases_dir.iterdir()
            if path.name not in expected_case_paths
            or _RESULT_NAME.fullmatch(path.name) is None
            or path.is_symlink()
            or not path.is_file()
        )
        if unexpected:
            raise CohortPreflightError(f"unexpected entries in cases output: {unexpected}")

        ordered_work = sorted(
            ((case, occurrences[_case_key(case)]) for case in schedule.cases),
            key=lambda item: item[1].absolute_step_index,
        )
        completed: dict[int, tuple[CaseRecord, FirstOccurrence, Path, Mapping[str, object]]] = {}
        for case, occurrence in ordered_work:
            path = _case_result_path(cases_dir, occurrence.case_index)
            if os.path.lexists(path):
                result = _load_case_result(
                    path,
                    case=case,
                    occurrence=occurrence,
                    contract_sha256=contract_sha256,
                    data_manifest_sha256=manifest.sha256,
                )
                resumed = True
            else:
                if should_stop is not None and should_stop():
                    break
                result = _process_case(
                    data_root=data_root,
                    manifest=manifest,
                    case_grids=case_grids,
                    config=config,
                    case=case,
                    occurrence=occurrence,
                    contract_sha256=contract_sha256,
                )
                atomic_create_bytes(path, canonical_json_bytes(result))
                result = _load_case_result(
                    path,
                    case=case,
                    occurrence=occurrence,
                    contract_sha256=contract_sha256,
                    data_manifest_sha256=manifest.sha256,
                )
                resumed = False
            completed[occurrence.case_index] = (case, occurrence, path, result)
            if progress is not None:
                universe = result["candidate_universe"]
                timings = result["timings_seconds"]
                assert isinstance(universe, Mapping)
                assert isinstance(timings, Mapping)
                progress(
                    {
                        "case_index": occurrence.case_index,
                        "case_id": case.case_id,
                        "completed_cases": len(completed),
                        "total_cases": schedule.case_count,
                        "resumed": resumed,
                        "candidate_count": universe["candidate_count"],
                        "total_seconds": timings["total"],
                    }
                )
            if should_stop is not None and should_stop():
                break

        if len(completed) != schedule.case_count:
            return {
                "schema": "simple-brats.cohort-cold-path-invocation",
                "schema_version": 1,
                "status": "partial_walltime_stop",
                "contract_sha256": contract_sha256,
                "completed_case_count": len(completed),
                "remaining_case_count": schedule.case_count - len(completed),
                "output_dir": os.fspath(destination),
            }

        aggregate = _build_aggregate(
            contract=contract,
            contract_sha256=contract_sha256,
            schedule=schedule,
            case_results=list(completed.values()),
        )
        _publish_or_require(destination / "result.json", aggregate, "preflight aggregate")
        return aggregate


def _progress_line(value: Mapping[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--expected-config-sha256",
        default=REGISTERED_CONFIG_SHA256,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    walltime_stop = WalltimeStop()
    signal.signal(signal.SIGUSR1, walltime_stop.handle)
    result = run_cohort_preflight(
        data_root=args.data_root,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        split_path=args.split,
        expected_split_sha256=args.expected_split_sha256,
        case_grid_manifest_path=args.case_grid_manifest,
        expected_case_grid_manifest_sha256=args.expected_case_grid_manifest_sha256,
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        output_dir=args.output_dir,
        expected_git_sha=args.expected_git_sha,
        repo_root=args.repo_root,
        progress=_progress_line,
        should_stop=walltime_stop,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return WALLTIME_REQUEUE_EXIT_CODE if result["status"] == "partial_walltime_stop" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_POOL_SIZE",
    "CohortPreflightError",
    "DEFAULT_BAGS_PER_SUBJECT",
    "DEFAULT_EXPECTED_TRAIN_CASES",
    "DEFAULT_EXPECTED_TRAIN_SUBJECTS",
    "FirstOccurrence",
    "MAX_PLAN_ATTEMPTS",
    "REGISTERED_CONFIG_SHA256",
    "REGISTERED_CONFIG_8MM_SHA256",
    "TARGET_COUNT",
    "WALLTIME_REQUEUE_EXIT_CODE",
    "WalltimeStop",
    "derive_first_occurrences",
    "run_cohort_preflight",
]
