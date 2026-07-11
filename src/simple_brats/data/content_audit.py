"""Resumable, label-free MRI content-identity audit.

This command is diagnostic by design.  Exact duplicates are serialized into
the final report but never make the audit fail.  Failures are reserved for
broken provenance, corrupt resume state, unavailable inputs, or extraction
errors.

Each completed case is written as one canonical JSON shard before the next
case begins.  Re-running the exact command validates and reuses those shards,
so a Slurm timeout does not discard completed canonicalization work.  The
final report is published once, with no overwrite mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from simple_brats.config import MODALITIES, ExperimentConfig, load_experiment_config

from .case_grids import (
    CaseGridManifest,
    CaseGridRecord,
    load_case_grid_manifest,
)
from .extraction import CanonicalVolume, ExtractionSpec
from .manifest import CaseRecord, DatasetManifest, canonical_json_bytes, load_manifest
from .pipeline import CachedNiftiPatchExtractor
from .splits import SplitManifest, load_split, partition_cases

CONTENT_AUDIT_SCHEMA = "simple-brats.content-audit"
CONTENT_AUDIT_SCHEMA_VERSION = 3
CONTENT_AUDIT_STATE_SCHEMA = "simple-brats.content-audit-state"
CONTENT_AUDIT_CASE_SCHEMA = "simple-brats.content-audit-case"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPRESENTATIONS = (
    ("raw_file", "raw_file_sha256"),
    ("canonical_voxel", "canonical_voxel_sha256"),
    ("normalized_voxel", "normalized_voxel_sha256"),
)


class ContentAuditError(ValueError):
    """Raised when content-audit provenance or resume state is invalid."""


class CanonicalVolumeExtractor(Protocol):
    """Minimal label-free interface used by the content audit."""

    data_manifest_sha256: str
    extraction_spec: ExtractionSpec
    extraction_spec_sha256: str

    def canonical_volumes_for_case(
        self,
        case: CaseRecord,
    ) -> dict[str, CanonicalVolume]: ...

    def clear_cache(self) -> None: ...


class CanonicalVolumeExtractorFactory(Protocol):
    """Construct one manifest-bound extractor for one case-specific grid."""

    def __call__(
        self,
        *,
        case: CaseRecord,
        extraction_spec: ExtractionSpec,
    ) -> CanonicalVolumeExtractor: ...


@dataclass(frozen=True, slots=True)
class ContentAuditProgress:
    """Result of one complete or deliberately bounded audit invocation."""

    completed_cases: int
    total_cases: int
    newly_completed_cases: int
    output_path: str | None
    output_sha256: str | None

    @property
    def complete(self) -> bool:
        return self.completed_cases == self.total_cases

    def to_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "completed_cases": self.completed_cases,
            "newly_completed_cases": self.newly_completed_cases,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "total_cases": self.total_cases,
        }


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContentAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_git_sha(value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ContentAuditError("launch_sha must be a full lowercase 40-character Git SHA")
    return value


def _canonical_object(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ContentAuditError(f"{description} must be a regular file, not a symlink: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContentAuditError(f"unable to read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContentAuditError(f"{description} must contain one JSON object: {path}")
    try:
        canonical = canonical_json_bytes(value)
    except ValueError as error:
        raise ContentAuditError(f"{description} is not canonical JSON: {path}") from error
    if raw != canonical:
        raise ContentAuditError(
            f"{description} bytes differ from their canonical JSON serialization: {path}"
        )
    return value


def _write_canonical_new(path: Path, value: Mapping[str, object]) -> str:
    """Atomically publish canonical JSON without replacing an existing path."""

    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing audit artifact: {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if os.path.lexists(temporary):
        raise FileExistsError(f"audit temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing audit artifact: {path}"
            ) from error
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _state_header(
    *,
    launch_sha: str,
    manifest: DatasetManifest,
    split: SplitManifest,
    case_grid_manifest: CaseGridManifest,
    experiment_config: ExperimentConfig,
) -> dict[str, object]:
    runtime_policy = case_grid_manifest.policy.for_patch_config(experiment_config.patch)
    return {
        "schema": CONTENT_AUDIT_STATE_SCHEMA,
        "schema_version": CONTENT_AUDIT_SCHEMA_VERSION,
        "launch_sha": launch_sha,
        "manifest_sha256": manifest.sha256,
        "split_sha256": split.sha256,
        "case_grid_manifest_sha256": case_grid_manifest.sha256,
        "case_grid_policy_sha256": case_grid_manifest.policy.sha256,
        "experiment_config_sha256": experiment_config.sha256,
        "runtime_extraction_policy_sha256": runtime_policy.sha256,
        "patch_config": {
            "footprint_mm": experiment_config.patch.footprint_mm,
            "thin_mm": experiment_config.patch.thin_mm,
            "tensor_shape": list(experiment_config.patch.tensor_shape),
        },
        "case_count": len(manifest.cases),
        "modalities": list(MODALITIES),
    }


def _case_identity(case: CaseRecord) -> dict[str, str]:
    return {
        "source": case.source,
        "release": case.release,
        "case_id": case.case_id,
        "subject_id": case.subject_id,
        "visit_id": case.visit_id,
    }


def _case_shard_path(case_dir: Path, index: int, case: CaseRecord) -> Path:
    identity_digest = hashlib.sha256(canonical_json_bytes(_case_identity(case))).hexdigest()
    return case_dir / f"{index:06d}-{identity_digest[:20]}.json"


def _case_record(
    case: CaseRecord,
    split_name: str,
    case_grid_record: CaseGridRecord,
    extraction_spec: ExtractionSpec,
    volumes: Mapping[str, CanonicalVolume],
) -> dict[str, object]:
    if set(volumes) != set(MODALITIES):
        raise ContentAuditError(
            f"extractor returned the wrong modalities for {case.case_id}: "
            f"expected={list(MODALITIES)}, got={sorted(volumes)}"
        )
    files = {record.modality: record for record in case.files}
    if any(modality not in files for modality in MODALITIES):
        raise ContentAuditError(f"case {case.case_id} lacks a required MRI modality")

    modalities: list[dict[str, object]] = []
    for modality in MODALITIES:
        volume = volumes[modality]
        if not isinstance(volume, CanonicalVolume):
            raise ContentAuditError(
                f"extractor returned a non-CanonicalVolume for {case.case_id}/{modality}"
            )
        if volume.extraction_spec_sha256 != extraction_spec.sha256:
            raise ContentAuditError(
                f"extractor returned a volume prepared under the wrong case spec for "
                f"{case.case_id}/{modality}"
            )
        stats = volume.normalization_stats
        modalities.append(
            {
                "modality": modality,
                "path": files[modality].path,
                "raw_file_sha256": files[modality].sha256,
                "canonical_voxel_sha256": volume.voxel_content_sha256,
                "normalized_voxel_sha256": volume.normalized_sha256,
                "normalization": {
                    "foreground_voxels": stats.foreground_voxels,
                    "mean": stats.mean,
                    "std": stats.std,
                },
            }
        )
    return {
        **_case_identity(case),
        "split": split_name,
        "case_grid_record_sha256": case_grid_record.sha256,
        "extraction_spec_sha256": extraction_spec.sha256,
        "modalities": modalities,
    }


def _validate_case_record(
    value: object,
    *,
    case: CaseRecord,
    split_name: str,
    case_grid_record: CaseGridRecord,
    extraction_spec: ExtractionSpec,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContentAuditError(f"resume record for {case.case_id} is not a JSON object")
    expected_keys = {
        "source",
        "release",
        "case_id",
        "subject_id",
        "visit_id",
        "split",
        "case_grid_record_sha256",
        "extraction_spec_sha256",
        "modalities",
    }
    if set(value) != expected_keys:
        raise ContentAuditError(f"resume record keys are invalid for {case.case_id}")
    expected_identity = _case_identity(case)
    for field, expected in expected_identity.items():
        if value[field] != expected:
            raise ContentAuditError(f"resume record {field} mismatch for {case.case_id}")
    if value["split"] != split_name:
        raise ContentAuditError(f"resume split mismatch for {case.case_id}")
    if value["case_grid_record_sha256"] != case_grid_record.sha256:
        raise ContentAuditError(f"resume case-grid record mismatch for {case.case_id}")
    if value["extraction_spec_sha256"] != extraction_spec.sha256:
        raise ContentAuditError(f"resume extraction-spec mismatch for {case.case_id}")
    modalities = value["modalities"]
    if not isinstance(modalities, list) or len(modalities) != len(MODALITIES):
        raise ContentAuditError(f"resume modalities are invalid for {case.case_id}")
    expected_modality_keys = {
        "modality",
        "path",
        "raw_file_sha256",
        "canonical_voxel_sha256",
        "normalized_voxel_sha256",
        "normalization",
    }
    files = {record.modality: record for record in case.files}
    for expected_modality, item in zip(MODALITIES, modalities, strict=True):
        if not isinstance(item, dict) or set(item) != expected_modality_keys:
            raise ContentAuditError(
                f"resume modality record is invalid for {case.case_id}/{expected_modality}"
            )
        if item["modality"] != expected_modality:
            raise ContentAuditError(f"resume modality order changed for {case.case_id}")
        expected_file = files[expected_modality]
        if item["path"] != expected_file.path or item["raw_file_sha256"] != expected_file.sha256:
            raise ContentAuditError(
                f"resume raw-file identity mismatch for {case.case_id}/{expected_modality}"
            )
        _require_sha256(item["canonical_voxel_sha256"], "canonical_voxel_sha256")
        _require_sha256(item["normalized_voxel_sha256"], "normalized_voxel_sha256")
        normalization = item["normalization"]
        if not isinstance(normalization, dict) or set(normalization) != {
            "foreground_voxels",
            "mean",
            "std",
        }:
            raise ContentAuditError(
                f"resume normalization record is invalid for {case.case_id}/{expected_modality}"
            )
        foreground = normalization["foreground_voxels"]
        mean = normalization["mean"]
        std = normalization["std"]
        if (
            isinstance(foreground, bool)
            or not isinstance(foreground, int)
            or foreground < 2
            or isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not float("-inf") < float(mean) < float("inf")
            or isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not 0.0 < float(std) < float("inf")
        ):
            raise ContentAuditError(
                f"resume normalization values are invalid for {case.case_id}/{expected_modality}"
            )
    return value


def _case_shard(
    *,
    header: Mapping[str, object],
    case_index: int,
    case_grid_record: CaseGridRecord,
    extraction_spec: ExtractionSpec,
    record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": CONTENT_AUDIT_CASE_SCHEMA,
        "schema_version": CONTENT_AUDIT_SCHEMA_VERSION,
        "launch_sha": header["launch_sha"],
        "manifest_sha256": header["manifest_sha256"],
        "split_sha256": header["split_sha256"],
        "case_grid_manifest_sha256": header["case_grid_manifest_sha256"],
        "case_grid_policy_sha256": header["case_grid_policy_sha256"],
        "experiment_config_sha256": header["experiment_config_sha256"],
        "runtime_extraction_policy_sha256": header[
            "runtime_extraction_policy_sha256"
        ],
        "case_grid_record_sha256": case_grid_record.sha256,
        "extraction_spec_sha256": extraction_spec.sha256,
        "case_index": case_index,
        "record": dict(record),
    }


def _validate_case_shard(
    shard: Mapping[str, object],
    *,
    header: Mapping[str, object],
    case_index: int,
    case: CaseRecord,
    split_name: str,
    case_grid_record: CaseGridRecord,
    extraction_spec: ExtractionSpec,
) -> dict[str, object]:
    expected_keys = {
        "schema",
        "schema_version",
        "launch_sha",
        "manifest_sha256",
        "split_sha256",
        "case_grid_manifest_sha256",
        "case_grid_policy_sha256",
        "experiment_config_sha256",
        "runtime_extraction_policy_sha256",
        "case_grid_record_sha256",
        "extraction_spec_sha256",
        "case_index",
        "record",
    }
    if set(shard) != expected_keys:
        raise ContentAuditError(f"resume shard keys are invalid for {case.case_id}")
    if (
        shard["schema"] != CONTENT_AUDIT_CASE_SCHEMA
        or shard["schema_version"] != CONTENT_AUDIT_SCHEMA_VERSION
    ):
        raise ContentAuditError(f"resume shard schema is invalid for {case.case_id}")
    for field in (
        "launch_sha",
        "manifest_sha256",
        "split_sha256",
        "case_grid_manifest_sha256",
        "case_grid_policy_sha256",
        "experiment_config_sha256",
        "runtime_extraction_policy_sha256",
    ):
        if shard[field] != header[field]:
            raise ContentAuditError(f"resume shard {field} mismatch for {case.case_id}")
    if shard["case_grid_record_sha256"] != case_grid_record.sha256:
        raise ContentAuditError(f"resume shard case-grid record mismatch for {case.case_id}")
    if shard["extraction_spec_sha256"] != extraction_spec.sha256:
        raise ContentAuditError(f"resume shard extraction-spec mismatch for {case.case_id}")
    if shard["case_index"] != case_index:
        raise ContentAuditError(f"resume case index mismatch for {case.case_id}")
    return _validate_case_record(
        shard["record"],
        case=case,
        split_name=split_name,
        case_grid_record=case_grid_record,
        extraction_spec=extraction_spec,
    )


def _duplicate_components(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for case in records:
        identity = {field: str(case[field]) for field in _case_identity_fields()}
        split_name = str(case["split"])
        modalities = case["modalities"]
        if not isinstance(modalities, list):
            raise AssertionError("validated case record lost its modality list")
        for modality_record in modalities:
            if not isinstance(modality_record, dict):
                raise AssertionError("validated modality record changed type")
            member = {
                **identity,
                "split": split_name,
                "modality": str(modality_record["modality"]),
                "path": str(modality_record["path"]),
            }
            for representation, digest_field in _REPRESENTATIONS:
                groups[(representation, str(modality_record[digest_field]))].append(member)

    cross_subject: list[dict[str, object]] = []
    cross_split: list[dict[str, object]] = []
    for (representation, digest), members in sorted(groups.items()):
        members.sort(
            key=lambda item: (
                item["subject_id"],
                item["visit_id"],
                item["source"],
                item["release"],
                item["case_id"],
                item["modality"],
                item["path"],
            )
        )
        subjects = sorted({item["subject_id"] for item in members})
        if len(subjects) < 2:
            continue
        splits = sorted({item["split"] for item in members})
        component = {
            "representation": representation,
            "sha256": digest,
            "member_count": len(members),
            "subjects": subjects,
            "splits": splits,
            "modalities": sorted({item["modality"] for item in members}),
            "members": members,
        }
        cross_subject.append(component)
        if len(splits) >= 2:
            cross_split.append(component)
    return {
        "cross_subject_components": cross_subject,
        "cross_split_components": cross_split,
    }


def _case_identity_fields() -> tuple[str, ...]:
    return ("source", "release", "case_id", "subject_id", "visit_id")


def _final_report(
    *,
    header: Mapping[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    duplicates = _duplicate_components(records)
    cross_subject = duplicates["cross_subject_components"]
    cross_split = duplicates["cross_split_components"]
    if not isinstance(cross_subject, list) or not isinstance(cross_split, list):
        raise AssertionError("duplicate report changed type")
    return {
        "schema": CONTENT_AUDIT_SCHEMA,
        "schema_version": CONTENT_AUDIT_SCHEMA_VERSION,
        "launch_sha": header["launch_sha"],
        "manifest_sha256": header["manifest_sha256"],
        "split_sha256": header["split_sha256"],
        "case_grid_manifest_sha256": header["case_grid_manifest_sha256"],
        "case_grid_policy_sha256": header["case_grid_policy_sha256"],
        "experiment_config_sha256": header["experiment_config_sha256"],
        "runtime_extraction_policy_sha256": header[
            "runtime_extraction_policy_sha256"
        ],
        "patch_config": header["patch_config"],
        "counts": {
            "cases": len(records),
            "subjects": len({str(record["subject_id"]) for record in records}),
            "mri_volume_records": len(records) * len(MODALITIES),
            "distinct_extraction_specs": len(
                {str(record["extraction_spec_sha256"]) for record in records}
            ),
            "cross_subject_duplicate_components": len(cross_subject),
            "cross_split_duplicate_components": len(cross_split),
        },
        "cases": records,
        "duplicates": duplicates,
    }


def run_content_audit(
    *,
    manifest: DatasetManifest,
    split: SplitManifest,
    case_grid_manifest: CaseGridManifest,
    experiment_config: ExperimentConfig,
    extractor_factory: CanonicalVolumeExtractorFactory,
    launch_sha: str,
    state_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    max_new_cases: int | None = None,
) -> ContentAuditProgress:
    """Resume or complete one exact, manifest-bound content audit.

    ``max_new_cases`` is a testing/operational checkpoint control.  When it is
    supplied and missing shards remain after that many new cases, no partial
    final report is written.  A later invocation with identical pins resumes.
    """

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if not isinstance(split, SplitManifest):
        raise TypeError("split must be a SplitManifest")
    if not isinstance(case_grid_manifest, CaseGridManifest):
        raise TypeError("case_grid_manifest must be a CaseGridManifest")
    if not isinstance(experiment_config, ExperimentConfig):
        raise TypeError("experiment_config must be an ExperimentConfig")
    if not callable(extractor_factory):
        raise TypeError("extractor_factory must be callable")
    launch_sha = _require_git_sha(launch_sha)
    if max_new_cases is not None and (
        isinstance(max_new_cases, bool)
        or not isinstance(max_new_cases, int)
        or max_new_cases <= 0
    ):
        raise ValueError("max_new_cases must be a positive integer when supplied")

    # This intentionally does not call validate_split: byte-duplicate overlap
    # is the diagnostic output of this audit, not a reason to abort it.
    partitions = partition_cases(manifest, split)
    empty_splits = sorted(name for name, cases in partitions.items() if not cases)
    if empty_splits:
        raise ContentAuditError(f"declared splits contain no cases: {empty_splits}")
    split_by_subject = {item.subject_id: item.split for item in split.assignments}
    case_grid_manifest.validate_manifest(manifest)

    state_path = Path(state_dir).expanduser()
    if state_path.is_symlink():
        raise ContentAuditError("content-audit state_dir must not be a symlink")
    state_path.mkdir(parents=True, exist_ok=True)
    if not state_path.is_dir():
        raise ContentAuditError("content-audit state_dir must be a directory")
    state_path = state_path.resolve(strict=True)
    case_dir = state_path / "cases"
    if case_dir.is_symlink():
        raise ContentAuditError("content-audit case state must not be a symlink")
    case_dir.mkdir(exist_ok=True)

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve(strict=True) / output.name
    if output.is_symlink() or os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing content-audit output: {output}")

    header = _state_header(
        launch_sha=launch_sha,
        manifest=manifest,
        split=split,
        case_grid_manifest=case_grid_manifest,
        experiment_config=experiment_config,
    )
    state_manifest_path = state_path / "state.json"
    if os.path.lexists(state_manifest_path):
        observed_header = _canonical_object(state_manifest_path, "content-audit state")
        if observed_header != header:
            raise ContentAuditError(
                "content-audit resume state is bound to different provenance or case ordering"
            )
    else:
        _write_canonical_new(state_manifest_path, header)

    expected_shards = {
        _case_shard_path(case_dir, index, case).name
        for index, case in enumerate(manifest.cases)
    }
    unexpected = sorted(
        path.name
        for path in case_dir.iterdir()
        if not path.name.startswith(".") and path.name not in expected_shards
    )
    if unexpected:
        raise ContentAuditError(f"unexpected files in content-audit case state: {unexpected}")

    records_by_index: dict[int, dict[str, object]] = {}
    missing: list[
        tuple[int, CaseRecord, Path, CaseGridRecord, ExtractionSpec]
    ] = []
    for index, case in enumerate(manifest.cases):
        shard_path = _case_shard_path(case_dir, index, case)
        split_name = split_by_subject[case.subject_id]
        case_grid_record = case_grid_manifest.record_for_case(case)
        extraction_spec = case_grid_manifest.extraction_spec_for_case(
            case,
            patch_config=experiment_config.patch,
        )
        if os.path.lexists(shard_path):
            shard = _canonical_object(shard_path, "content-audit case shard")
            records_by_index[index] = _validate_case_shard(
                shard,
                header=header,
                case_index=index,
                case=case,
                split_name=split_name,
                case_grid_record=case_grid_record,
                extraction_spec=extraction_spec,
            )
        else:
            missing.append(
                (index, case, shard_path, case_grid_record, extraction_spec)
            )

    selected_missing = missing if max_new_cases is None else missing[:max_new_cases]
    newly_completed = 0
    for index, case, shard_path, case_grid_record, extraction_spec in selected_missing:
        extractor = extractor_factory(case=case, extraction_spec=extraction_spec)
        if extractor.data_manifest_sha256 != manifest.sha256:
            raise ContentAuditError("extractor is bound to a different manifest")
        if extractor.extraction_spec_sha256 != extraction_spec.sha256:
            raise ContentAuditError("extractor is bound to a different extraction spec")
        if extractor.extraction_spec != extraction_spec:
            raise ContentAuditError(
                "extractor extraction-spec value differs from the runtime config spec"
            )
        try:
            volumes = extractor.canonical_volumes_for_case(case)
            record = _case_record(
                case,
                split_by_subject[case.subject_id],
                case_grid_record,
                extraction_spec,
                volumes,
            )
            shard = _case_shard(
                header=header,
                case_index=index,
                case_grid_record=case_grid_record,
                extraction_spec=extraction_spec,
                record=record,
            )
            _write_canonical_new(shard_path, shard)
            records_by_index[index] = record
            newly_completed += 1
        finally:
            extractor.clear_cache()

    completed = len(records_by_index)
    if completed != len(manifest.cases):
        return ContentAuditProgress(
            completed_cases=completed,
            total_cases=len(manifest.cases),
            newly_completed_cases=newly_completed,
            output_path=None,
            output_sha256=None,
        )

    ordered_records = [records_by_index[index] for index in range(len(manifest.cases))]
    report = _final_report(header=header, records=ordered_records)
    output_sha256 = _write_canonical_new(output, report)
    return ContentAuditProgress(
        completed_cases=completed,
        total_cases=len(manifest.cases),
        newly_completed_cases=newly_completed,
        output_path=os.fspath(output),
        output_sha256=output_sha256,
    )


def _verify_git_sha(expected: str, repo_root: Path) -> str:
    expected = _require_git_sha(expected)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContentAuditError(f"unable to inspect Git checkout at {repo_root}") from error
    actual = result.stdout.strip().lower()
    if actual != expected:
        raise ContentAuditError(f"Git provenance mismatch: expected {expected}, got {actual}")
    return actual


def _require_canonical_input(path: Path, value: Mapping[str, object], description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContentAuditError(f"{description} must be a regular file, not a symlink")
    if path.read_bytes() != canonical_json_bytes(value):
        raise ContentAuditError(f"{description} is valid but is not canonical JSON on disk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simple_brats.data.content_audit",
        description="Resume a pinned label-free MRI canonical-content audit.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-cases", type=int)
    parser.add_argument("--max-cached-volumes", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    launch_sha = _verify_git_sha(args.expected_git_sha, repo_root)

    manifest_path = Path(args.manifest).expanduser()
    split_path = Path(args.split).expanduser()
    case_grid_manifest_path = Path(args.case_grid_manifest).expanduser()
    config_path = Path(args.config).expanduser()
    if config_path.is_symlink() or not config_path.is_file():
        raise ContentAuditError("config must be a regular file, not a symlink")
    manifest = load_manifest(
        manifest_path,
        expected_sha256=args.expected_manifest_sha256,
    )
    split = load_split(split_path, expected_sha256=args.expected_split_sha256)
    case_grid_manifest = load_case_grid_manifest(
        case_grid_manifest_path,
        expected_sha256=args.expected_case_grid_manifest_sha256,
    )
    experiment_config = load_experiment_config(config_path)
    _require_canonical_input(manifest_path, manifest.to_dict(), "manifest")
    _require_canonical_input(split_path, split.to_dict(), "split")
    _require_canonical_input(
        case_grid_manifest_path,
        case_grid_manifest.to_dict(),
        "case-grid manifest",
    )

    def extractor_factory(
        *,
        case: CaseRecord,
        extraction_spec: ExtractionSpec,
    ) -> CachedNiftiPatchExtractor:
        # The case argument makes the one-extractor-per-case lifecycle explicit;
        # manifest binding still prevents access to files outside this catalog.
        case_grid_manifest.record_for_case(case)
        return CachedNiftiPatchExtractor(
            data_root=args.data_root,
            manifest=manifest,
            data_manifest_sha256=manifest.sha256,
            extraction_spec=extraction_spec,
            max_cached_volumes=args.max_cached_volumes,
        )

    progress = run_content_audit(
        manifest=manifest,
        split=split,
        case_grid_manifest=case_grid_manifest,
        experiment_config=experiment_config,
        extractor_factory=extractor_factory,
        launch_sha=launch_sha,
        state_dir=args.state_dir,
        output_path=args.output,
        max_new_cases=args.max_new_cases,
    )
    print(json.dumps(progress.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTENT_AUDIT_SCHEMA",
    "CONTENT_AUDIT_SCHEMA_VERSION",
    "ContentAuditError",
    "ContentAuditProgress",
    "run_content_audit",
]
