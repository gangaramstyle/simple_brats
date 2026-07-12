"""Deterministic, segmentation-labeled patch sets for frozen-token evaluation.

Segmentation is used only here, after SSL training, to label a materialized
downstream set.  The only assumed semantics are ``seg > 0``.  Boundary and
near-tumor samples are excluded by an explicit ternary rule rather than being
silently assigned to either binary class.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn.functional as F

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.config import PatchConfig
from simple_brats.data.case_grids import CaseGridManifest
from simple_brats.data.extraction import ExtractionSpec, LoadedNifti, load_nifti_ras
from simple_brats.data.manifest import (
    CaseRecord,
    DatasetManifest,
    canonical_json_bytes,
    sha256_file,
)
from simple_brats.data.pipeline import (
    CachedNiftiPatchExtractor,
    prepare_case_candidate_universe,
)
from simple_brats.data.splits import SplitManifest, partition_cases, validate_split

EVALUATION_PATCH_SCHEMA = "simple-brats.evaluation-patches"
EVALUATION_PATCH_SCHEMA_VERSION = 1


class PatchEvaluationError(ValueError):
    """Evaluation patch construction violated a data or leakage contract."""


class NoEligibleBinaryPatchesError(PatchEvaluationError):
    """A case has no balanced samples under the predeclared ternary rule."""


@dataclass(frozen=True, slots=True)
class VerifiedSegmentationLabelAudit:
    file_sha256: str
    numeric_label_values: tuple[int, ...]
    launch_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_sha256", _sha256(self.file_sha256, "file_sha256"))
        values = tuple(self.numeric_label_values)
        if (
            not values
            or tuple(sorted(set(values))) != values
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
            or values[0] != 0
            or values[-1] <= 0
        ):
            raise PatchEvaluationError(
                "audited numeric labels must be sorted unique nonnegative integers "
                "containing zero and foreground"
            )
        if (
            not isinstance(self.launch_sha, str)
            or len(self.launch_sha) != 40
            or any(character not in "0123456789abcdef" for character in self.launch_sha)
        ):
            raise PatchEvaluationError("label-audit launch_sha must be a full lowercase Git SHA")
        object.__setattr__(self, "numeric_label_values", values)


def verify_segmentation_label_audit(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    manifest: DatasetManifest,
    split: SplitManifest,
) -> VerifiedSegmentationLabelAudit:
    """Verify the actual label-audit artifact before any label is consumed."""

    expected = _sha256(expected_sha256, "expected segmentation label audit SHA")
    audit_path = Path(path).expanduser()
    if audit_path.is_symlink():
        raise PatchEvaluationError("segmentation label audit must not be a symlink")
    try:
        audit_path = audit_path.resolve(strict=True)
    except OSError as error:
        raise PatchEvaluationError("segmentation label audit is unavailable") from error
    if not audit_path.is_file():
        raise PatchEvaluationError("segmentation label audit must be a regular file")
    actual = sha256_file(audit_path)
    if actual != expected:
        raise PatchEvaluationError(
            f"segmentation label audit SHA mismatch: expected {expected}, got {actual}"
        )
    try:
        value = json.loads(audit_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PatchEvaluationError("segmentation label audit is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "schema_version",
        "provenance",
        "results",
        "label_semantics",
    }:
        raise PatchEvaluationError("segmentation label audit has an unsupported top-level schema")
    provenance = value["provenance"]
    results = value["results"]
    semantics = value["label_semantics"]
    if not all(isinstance(item, Mapping) for item in (provenance, results, semantics)):
        raise PatchEvaluationError("label-audit provenance/results/semantics must be objects")
    if provenance.get("manifest_sha256") != manifest.sha256:
        raise PatchEvaluationError("label audit is bound to a different data manifest")
    if provenance.get("split_sha256") != split.sha256:
        raise PatchEvaluationError("label audit is bound to a different subject split")
    if semantics.get("semantic_names_assigned") is not False:
        raise PatchEvaluationError("numeric segmentation labels must remain semantically unnamed")
    raw_values = results.get("numeric_label_values")
    if not isinstance(raw_values, list):
        raise PatchEvaluationError("label audit has no aggregate numeric_label_values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or not float(item).is_integer()
        for item in raw_values
    ):
        raise PatchEvaluationError("audited numeric label values must be finite integers")
    return VerifiedSegmentationLabelAudit(
        file_sha256=actual,
        numeric_label_values=tuple(sorted({int(item) for item in raw_values})),
        launch_sha=provenance.get("launch_sha"),  # type: ignore[arg-type]
    )


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PatchEvaluationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PatchEvaluationError(f"{name} must be a non-empty canonical string")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise PatchEvaluationError(
            f"invalid {name} keys: missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


@dataclass(frozen=True, slots=True)
class BinaryPatchLabelRule:
    """Boundary-safe binary task derived solely from a ``seg > 0`` mask."""

    positive_minimum_fraction: float = 0.25
    negative_halo_mm: float = 4.0
    segmentation_foreground: str = "voxel_value_gt_zero"
    positive_definition: str = "crop_fraction_greater_than_or_equal_to_threshold"
    negative_definition: str = "no_foreground_in_crop_or_axis_aligned_halo"
    ambiguous_policy: str = "exclude"

    def __post_init__(self) -> None:
        if (
            isinstance(self.positive_minimum_fraction, bool)
            or not isinstance(self.positive_minimum_fraction, (int, float))
            or not math.isfinite(float(self.positive_minimum_fraction))
            or not 0 < float(self.positive_minimum_fraction) <= 1
        ):
            raise PatchEvaluationError("positive_minimum_fraction must lie in (0, 1]")
        if (
            isinstance(self.negative_halo_mm, bool)
            or not isinstance(self.negative_halo_mm, (int, float))
            or not math.isfinite(float(self.negative_halo_mm))
            or float(self.negative_halo_mm) < 0
            or not float(self.negative_halo_mm).is_integer()
        ):
            raise PatchEvaluationError(
                "negative_halo_mm must be a non-negative integer on the 1mm grid"
            )
        fixed = {
            "segmentation_foreground": "voxel_value_gt_zero",
            "positive_definition": "crop_fraction_greater_than_or_equal_to_threshold",
            "negative_definition": "no_foreground_in_crop_or_axis_aligned_halo",
            "ambiguous_policy": "exclude",
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise PatchEvaluationError(f"label rule requires {name}={expected!r}")
        object.__setattr__(self, "positive_minimum_fraction", float(self.positive_minimum_fraction))
        object.__setattr__(self, "negative_halo_mm", float(self.negative_halo_mm))

    def minimum_positive_voxels(self, crop_voxels: int) -> int:
        if isinstance(crop_voxels, bool) or not isinstance(crop_voxels, int) or crop_voxels <= 0:
            raise PatchEvaluationError("crop_voxels must be a positive integer")
        return math.ceil(self.positive_minimum_fraction * crop_voxels)

    def to_dict(self) -> dict[str, object]:
        return {
            "segmentation_foreground": self.segmentation_foreground,
            "positive_definition": self.positive_definition,
            "positive_minimum_fraction": self.positive_minimum_fraction,
            "negative_definition": self.negative_definition,
            "negative_halo_mm": self.negative_halo_mm,
            "ambiguous_policy": self.ambiguous_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BinaryPatchLabelRule:
        _exact_keys(value, set(cls().to_dict()), "binary patch label rule")
        return cls(**value)  # type: ignore[arg-type]


def _record_identity_payload(
    *,
    source: str,
    release: str,
    case_id: str,
    subject_id: str,
    partition: str,
    center_mm: tuple[float, float, float],
    seg_positive_voxels: int,
    crop_voxels: int,
    halo_clear: bool,
    label: int,
) -> dict[str, object]:
    return {
        "source": source,
        "release": release,
        "case_id": case_id,
        "subject_id": subject_id,
        "partition": partition,
        "center_mm": list(center_mm),
        "seg_positive_voxels": seg_positive_voxels,
        "crop_voxels": crop_voxels,
        "halo_clear": halo_clear,
        "label": label,
    }


@dataclass(frozen=True, slots=True)
class EvaluationPatchRecord:
    sample_id: str
    source: str
    release: str
    case_id: str
    subject_id: str
    partition: str
    center_mm: tuple[float, float, float]
    seg_positive_voxels: int
    crop_voxels: int
    halo_clear: bool
    label: int

    def __post_init__(self) -> None:
        for name in ("source", "release", "case_id", "subject_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.partition not in {"probe_train", "validation"}:
            raise PatchEvaluationError("record partition must be probe_train or validation")
        try:
            center = tuple(float(value) for value in self.center_mm)
        except (TypeError, ValueError) as error:
            raise PatchEvaluationError("center_mm must contain three finite values") from error
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise PatchEvaluationError("center_mm must contain three finite values")
        for name in ("seg_positive_voxels", "crop_voxels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PatchEvaluationError(f"{name} must be a non-negative integer")
        if self.crop_voxels <= 0 or self.seg_positive_voxels > self.crop_voxels:
            raise PatchEvaluationError("seg_positive_voxels must lie within the patch crop")
        if not isinstance(self.halo_clear, bool):
            raise PatchEvaluationError("halo_clear must be boolean")
        if isinstance(self.label, bool) or self.label not in {0, 1}:
            raise PatchEvaluationError("label must be binary 0/1")
        if self.label == 0 and not self.halo_clear:
            raise PatchEvaluationError("a negative evaluation patch must have a clear halo")
        object.__setattr__(self, "center_mm", center)
        payload = _record_identity_payload(
            source=self.source,
            release=self.release,
            case_id=self.case_id,
            subject_id=self.subject_id,
            partition=self.partition,
            center_mm=center,  # type: ignore[arg-type]
            seg_positive_voxels=self.seg_positive_voxels,
            crop_voxels=self.crop_voxels,
            halo_clear=self.halo_clear,
            label=self.label,
        )
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if self.sample_id != expected:
            raise PatchEvaluationError("sample_id does not address the exact patch record")

    @classmethod
    def create(cls, **values: object) -> EvaluationPatchRecord:
        payload = _record_identity_payload(**values)  # type: ignore[arg-type]
        return cls(
            sample_id=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            **values,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            **_record_identity_payload(
                source=self.source,
                release=self.release,
                case_id=self.case_id,
                subject_id=self.subject_id,
                partition=self.partition,
                center_mm=self.center_mm,
                seg_positive_voxels=self.seg_positive_voxels,
                crop_voxels=self.crop_voxels,
                halo_clear=self.halo_clear,
                label=self.label,
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EvaluationPatchRecord:
        expected = set(
            EvaluationPatchRecord.create(
                source="s",
                release="r",
                case_id="c",
                subject_id="u",
                partition="probe_train",
                center_mm=(0.0, 0.0, 0.0),
                seg_positive_voxels=0,
                crop_voxels=1,
                halo_clear=True,
                label=0,
            ).to_dict()
        )
        _exact_keys(value, expected, "evaluation patch record")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SegmentationAuditRecord:
    case_id: str
    file_sha256: str
    observed_positive_values: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "file_sha256", _sha256(self.file_sha256, "file_sha256"))
        values = tuple(self.observed_positive_values)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise PatchEvaluationError("observed segmentation values must be positive integers")
        if tuple(sorted(set(values))) != values:
            raise PatchEvaluationError("observed segmentation values must be sorted and unique")
        object.__setattr__(self, "observed_positive_values", values)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "file_sha256": self.file_sha256,
            "observed_positive_values": list(self.observed_positive_values),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SegmentationAuditRecord:
        _exact_keys(
            value,
            {"case_id", "file_sha256", "observed_positive_values"},
            "segmentation audit record",
        )
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EvaluationPatchManifest:
    data_manifest_sha256: str
    subject_split_sha256: str
    case_grid_manifest_sha256: str
    segmentation_label_audit_sha256: str
    patch_config: PatchConfig
    seed: int
    label_rule: BinaryPatchLabelRule
    probe_train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    ineligible_probe_train_subjects: tuple[str, ...]
    locked_test_subject_count: int
    segmentation_audit: tuple[SegmentationAuditRecord, ...]
    records: tuple[EvaluationPatchRecord, ...]
    schema: str = EVALUATION_PATCH_SCHEMA
    schema_version: int = EVALUATION_PATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "data_manifest_sha256",
            "subject_split_sha256",
            "case_grid_manifest_sha256",
            "segmentation_label_audit_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if not isinstance(self.patch_config, PatchConfig) or (
            self.patch_config.footprint_mm,
            self.patch_config.thin_mm,
            self.patch_config.tensor_shape,
        ) != (4.0, 4.0, (16, 16, 16)):
            raise PatchEvaluationError(
                "primary frozen-token evaluation is locked to 4mm cubes / 16x16x16"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PatchEvaluationError("seed must be a non-negative integer")
        if not isinstance(self.label_rule, BinaryPatchLabelRule):
            raise TypeError("label_rule must be a BinaryPatchLabelRule")
        probe_subjects = tuple(self.probe_train_subjects)
        validation_subjects = tuple(self.validation_subjects)
        ineligible_subjects = tuple(self.ineligible_probe_train_subjects)
        for name, subjects in (
            ("probe_train_subjects", probe_subjects),
            ("validation_subjects", validation_subjects),
        ):
            if (
                not subjects
                or len(set(subjects)) != len(subjects)
                or any(not isinstance(subject, str) or not subject for subject in subjects)
            ):
                raise PatchEvaluationError(f"{name} must contain unique non-empty subjects")
        if set(probe_subjects) & set(validation_subjects):
            raise PatchEvaluationError("probe-train and validation subjects must be disjoint")
        if (
            len(set(ineligible_subjects)) != len(ineligible_subjects)
            or set(ineligible_subjects) & (set(probe_subjects) | set(validation_subjects))
            or any(not isinstance(subject, str) or not subject for subject in ineligible_subjects)
        ):
            raise PatchEvaluationError(
                "ineligible probe subjects must be unique and absent from evaluated subjects"
            )
        if (
            isinstance(self.locked_test_subject_count, bool)
            or not isinstance(self.locked_test_subject_count, int)
            or self.locked_test_subject_count <= 0
        ):
            raise PatchEvaluationError("locked_test_subject_count must be positive")
        records = tuple(self.records)
        if not records or not all(isinstance(record, EvaluationPatchRecord) for record in records):
            raise PatchEvaluationError("records must contain evaluation patch records")
        if len({record.sample_id for record in records}) != len(records):
            raise PatchEvaluationError("evaluation sample IDs must be unique")
        allowed = {(subject, "probe_train") for subject in probe_subjects} | {
            (subject, "validation") for subject in validation_subjects
        }
        if {(record.subject_id, record.partition) for record in records} - allowed:
            raise PatchEvaluationError("a patch record is outside its declared subject partition")
        counts: dict[tuple[str, int], int] = defaultdict(int)
        for record in records:
            minimum = self.label_rule.minimum_positive_voxels(record.crop_voxels)
            if record.label == 1 and record.seg_positive_voxels < minimum:
                raise PatchEvaluationError("positive patch falls below the occupancy threshold")
            if record.label == 0 and (record.seg_positive_voxels != 0 or not record.halo_clear):
                raise PatchEvaluationError("negative patch is not tumor-free with a clear halo")
            counts[(record.subject_id, record.label)] += 1
        for subject in (*probe_subjects, *validation_subjects):
            if counts[(subject, 0)] == 0 or counts[(subject, 0)] != counts[(subject, 1)]:
                raise PatchEvaluationError(
                    "every evaluation subject must contribute equal nonzero class counts"
                )
        audits = tuple(self.segmentation_audit)
        selected_cases = {record.case_id for record in records}
        if (
            len({audit.case_id for audit in audits}) != len(audits)
            or {audit.case_id for audit in audits} != selected_cases
        ):
            raise PatchEvaluationError("segmentation audit must cover every selected case once")
        if self.schema != EVALUATION_PATCH_SCHEMA or (
            self.schema_version != EVALUATION_PATCH_SCHEMA_VERSION
        ):
            raise PatchEvaluationError("unsupported evaluation-patch schema")
        object.__setattr__(self, "probe_train_subjects", probe_subjects)
        object.__setattr__(self, "validation_subjects", validation_subjects)
        object.__setattr__(self, "ineligible_probe_train_subjects", ineligible_subjects)
        object.__setattr__(
            self,
            "segmentation_audit",
            tuple(sorted(audits, key=lambda item: item.case_id)),
        )
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda x: x.sample_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "data_manifest_sha256": self.data_manifest_sha256,
            "subject_split_sha256": self.subject_split_sha256,
            "case_grid_manifest_sha256": self.case_grid_manifest_sha256,
            "segmentation_label_audit_sha256": self.segmentation_label_audit_sha256,
            "patch_config": {
                "footprint_mm": self.patch_config.footprint_mm,
                "thin_mm": self.patch_config.thin_mm,
                "tensor_shape": list(self.patch_config.tensor_shape),
            },
            "seed": self.seed,
            "label_rule": self.label_rule.to_dict(),
            "probe_train_subjects": list(self.probe_train_subjects),
            "validation_subjects": list(self.validation_subjects),
            "ineligible_probe_train_subjects": list(self.ineligible_probe_train_subjects),
            "locked_test_subject_count": self.locked_test_subject_count,
            "locked_test_image_or_label_access": False,
            "segmentation_audit": [record.to_dict() for record in self.segmentation_audit],
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EvaluationPatchManifest:
        expected = {
            "schema",
            "schema_version",
            "data_manifest_sha256",
            "subject_split_sha256",
            "case_grid_manifest_sha256",
            "segmentation_label_audit_sha256",
            "patch_config",
            "seed",
            "label_rule",
            "probe_train_subjects",
            "validation_subjects",
            "ineligible_probe_train_subjects",
            "locked_test_subject_count",
            "locked_test_image_or_label_access",
            "segmentation_audit",
            "records",
        }
        _exact_keys(value, expected, "evaluation patch manifest")
        if value["locked_test_image_or_label_access"] is not False:
            raise PatchEvaluationError("locked test images and labels must remain untouched")
        if not isinstance(value["patch_config"], Mapping) or not isinstance(
            value["label_rule"], Mapping
        ):
            raise PatchEvaluationError("patch_config and label_rule must be objects")
        raw_audit = value["segmentation_audit"]
        raw_records = value["records"]
        if not isinstance(raw_audit, list) or not isinstance(raw_records, list):
            raise PatchEvaluationError("segmentation_audit and records must be arrays")
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            data_manifest_sha256=value["data_manifest_sha256"],  # type: ignore[arg-type]
            subject_split_sha256=value["subject_split_sha256"],  # type: ignore[arg-type]
            case_grid_manifest_sha256=value["case_grid_manifest_sha256"],  # type: ignore[arg-type]
            segmentation_label_audit_sha256=value["segmentation_label_audit_sha256"],  # type: ignore[arg-type]
            patch_config=PatchConfig(
                footprint_mm=value["patch_config"]["footprint_mm"],  # type: ignore[arg-type,index]
                thin_mm=value["patch_config"]["thin_mm"],  # type: ignore[arg-type,index]
                tensor_shape=tuple(value["patch_config"]["tensor_shape"]),  # type: ignore[arg-type,index]
            ),
            seed=value["seed"],  # type: ignore[arg-type]
            label_rule=BinaryPatchLabelRule.from_dict(value["label_rule"]),
            probe_train_subjects=tuple(value["probe_train_subjects"]),  # type: ignore[arg-type]
            validation_subjects=tuple(value["validation_subjects"]),  # type: ignore[arg-type]
            ineligible_probe_train_subjects=tuple(
                value["ineligible_probe_train_subjects"]  # type: ignore[arg-type]
            ),
            locked_test_subject_count=value["locked_test_subject_count"],  # type: ignore[arg-type]
            segmentation_audit=tuple(SegmentationAuditRecord.from_dict(item) for item in raw_audit),
            records=tuple(EvaluationPatchRecord.from_dict(item) for item in raw_records),
        )


def _resolve_manifest_path(data_root: Path, recorded_path: str) -> Path:
    pure = PurePosixPath(recorded_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise PatchEvaluationError("segmentation path must be canonical and data-root-relative")
    candidate = data_root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise PatchEvaluationError("segmentation path must not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PatchEvaluationError("segmentation file is unavailable") from error
    if not resolved.is_file() or data_root not in resolved.parents:
        raise PatchEvaluationError("segmentation path escapes the data root")
    return resolved


def _nearest_resample_binary(image: LoadedNifti, spec: ExtractionSpec) -> np.ndarray:
    source = torch.from_numpy(np.array(image.data > 0, dtype=np.float32, order="C"))
    source = source.permute(2, 1, 0)[None, None]
    transform = torch.as_tensor(
        np.linalg.inv(image.affine) @ np.asarray(spec.canonical_affine), dtype=torch.float64
    )
    output = np.zeros(spec.canonical_shape, dtype=np.bool_)
    sx, sy, sz = image.data.shape
    tx, ty, tz = spec.canonical_shape

    def normalized(coordinate: torch.Tensor, size: int) -> torch.Tensor:
        return torch.zeros_like(coordinate) if size == 1 else 2 * coordinate / (size - 1) - 1

    with torch.no_grad():
        for z_start in range(0, tz, 8):
            z_stop = min(z_start + 8, tz)
            zz, yy, xx = torch.meshgrid(
                torch.arange(z_start, z_stop, dtype=torch.float64),
                torch.arange(ty, dtype=torch.float64),
                torch.arange(tx, dtype=torch.float64),
                indexing="ij",
            )
            source_x = (
                transform[0, 0] * xx + transform[0, 1] * yy + transform[0, 2] * zz + transform[0, 3]
            )
            source_y = (
                transform[1, 0] * xx + transform[1, 1] * yy + transform[1, 2] * zz + transform[1, 3]
            )
            source_z = (
                transform[2, 0] * xx + transform[2, 1] * yy + transform[2, 2] * zz + transform[2, 3]
            )
            valid = (
                (source_x >= -1e-6)
                & (source_x <= sx - 1 + 1e-6)
                & (source_y >= -1e-6)
                & (source_y <= sy - 1 + 1e-6)
                & (source_z >= -1e-6)
                & (source_z <= sz - 1 + 1e-6)
            )
            grid = torch.stack(
                (
                    normalized(source_x.clamp(0, sx - 1), sx),
                    normalized(source_y.clamp(0, sy - 1), sy),
                    normalized(source_z.clamp(0, sz - 1), sz),
                ),
                dim=-1,
            ).float()
            sampled = (
                F.grid_sample(
                    source, grid[None], mode="nearest", padding_mode="zeros", align_corners=True
                )[0, 0].bool()
                & valid
            )
            output[:, :, z_start:z_stop] = sampled.permute(2, 1, 0).numpy()
    output.setflags(write=False)
    return output


def _load_segmentation(
    *,
    data_root: Path,
    case: CaseRecord,
    case_grids: CaseGridManifest,
    spec: ExtractionSpec,
) -> tuple[np.ndarray, SegmentationAuditRecord]:
    matches = [record for record in case.files if record.modality == "seg"]
    if len(matches) != 1:
        raise PatchEvaluationError(f"case {case.case_id} must contain exactly one segmentation")
    record = matches[0]
    path = _resolve_manifest_path(data_root, record.path)
    before = sha256_file(path)
    if before != record.sha256:
        raise PatchEvaluationError(f"segmentation SHA mismatch for {case.case_id}")
    image = load_nifti_ras(path, allow_unknown_spatial_unit=True)
    grid_record = case_grids.record_for_case(case)
    if image.data.shape != grid_record.native_grid.shape or not np.allclose(
        image.affine,
        np.asarray(grid_record.native_grid.affine),
        atol=case_grids.policy.within_case_affine_atol,
        rtol=case_grids.policy.within_case_affine_rtol,
    ):
        raise PatchEvaluationError(
            f"segmentation grid for {case.case_id} is not registered to the MRI reference"
        )
    rounded = np.rint(image.data)
    if bool((image.data < 0).any()) or not np.allclose(image.data, rounded, atol=1e-6, rtol=0):
        raise PatchEvaluationError("segmentation values must be non-negative integers")
    values = tuple(int(value) for value in np.unique(rounded) if value > 0)
    if not values:
        raise NoEligibleBinaryPatchesError(
            f"segmentation for {case.case_id} contains no seg>0 voxels"
        )
    mask = _nearest_resample_binary(image, spec)
    after = sha256_file(path)
    if after != before:
        raise PatchEvaluationError("segmentation changed while it was being loaded")
    return mask, SegmentationAuditRecord(
        case_id=case.case_id,
        file_sha256=record.sha256,
        observed_positive_values=values,
    )


def _window_sums(mask: np.ndarray, window: tuple[int, int, int]) -> np.ndarray:
    result = np.asarray(mask, dtype=np.int32)
    for axis, width in enumerate(window):
        cumulative = np.cumsum(result, axis=axis, dtype=np.int32)
        zero_shape = list(cumulative.shape)
        zero_shape[axis] = 1
        padded = np.concatenate((np.zeros(zero_shape, dtype=np.int32), cumulative), axis=axis)
        upper = [slice(None)] * 3
        lower = [slice(None)] * 3
        upper[axis] = slice(width, None)
        lower[axis] = slice(None, -width)
        result = padded[tuple(upper)] - padded[tuple(lower)]
    return result


def label_candidate_centers(
    *,
    spec: ExtractionSpec,
    centers_mm: np.ndarray,
    segmentation_mask: np.ndarray,
    rule: BinaryPatchLabelRule,
) -> tuple[np.ndarray, np.ndarray]:
    """Return crop positive counts and halo-clear flags for candidate centers."""

    centers = np.asarray(centers_mm, dtype=np.float64)
    mask = np.asarray(segmentation_mask)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise PatchEvaluationError("candidate centers must be a finite Nx3 array")
    if mask.shape != spec.canonical_shape or mask.dtype != np.bool_:
        raise PatchEvaluationError("segmentation mask must match the canonical grid")
    affine = np.asarray(spec.canonical_affine)
    voxel = (centers - affine[:3, 3]) / np.diag(affine[:3, :3])
    starts_float = voxel - (np.asarray(spec.patch_source_shape) - 1.0) / 2.0
    starts = np.rint(starts_float).astype(np.int64)
    maximum = np.asarray(spec.canonical_shape) - np.asarray(spec.patch_source_shape)
    if not np.allclose(starts, starts_float, atol=1e-6, rtol=0) or bool(
        ((starts < 0) | (starts > maximum)).any()
    ):
        raise PatchEvaluationError("candidate center is off-lattice or outside the patch grid")
    crop_grid = _window_sums(mask, spec.patch_source_shape)
    crop_counts = crop_grid[starts[:, 0], starts[:, 1], starts[:, 2]].astype(np.int32)

    halo = int(rule.negative_halo_mm)
    tensor = torch.from_numpy(np.array(mask, dtype=np.float32, order="C"))
    tensor = tensor.permute(2, 1, 0)[None, None]
    with torch.no_grad():
        dilated = (
            F.max_pool3d(
                tensor,
                kernel_size=2 * halo + 1,
                stride=1,
                padding=halo,
            )[0, 0]
            .permute(2, 1, 0)
            .bool()
            .numpy()
        )
    halo_grid = _window_sums(dilated, spec.patch_source_shape)
    halo_clear = halo_grid[starts[:, 0], starts[:, 1], starts[:, 2]] == 0
    return crop_counts, halo_clear


def _stable_seed(seed: int, *parts: str) -> int:
    payload = canonical_json_bytes({"seed": seed, "parts": list(parts)})
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_balanced_patch_records(
    *,
    case: CaseRecord,
    partition: str,
    centers_mm: np.ndarray,
    crop_counts: np.ndarray,
    halo_clear: np.ndarray,
    rule: BinaryPatchLabelRule,
    seed: int,
    maximum_per_class: int,
    minimum_per_class: int,
    crop_voxels: int,
) -> tuple[EvaluationPatchRecord, ...]:
    """Select a deterministic balanced subset while excluding ambiguous centers."""

    centers = np.asarray(centers_mm, dtype=np.float64)
    counts = np.asarray(crop_counts)
    clear = np.asarray(halo_clear)
    if counts.shape != (len(centers),) or clear.shape != (len(centers),):
        raise PatchEvaluationError("candidate labels do not align with candidate centers")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (maximum_per_class, minimum_per_class, crop_voxels)
        )
        or minimum_per_class > maximum_per_class
    ):
        raise PatchEvaluationError("invalid per-class sampling bounds")
    positive = np.flatnonzero(counts >= rule.minimum_positive_voxels(crop_voxels))
    negative = np.flatnonzero((counts == 0) & clear)
    selected_count = min(maximum_per_class, len(positive), len(negative))
    if selected_count < minimum_per_class:
        raise NoEligibleBinaryPatchesError(
            f"case {case.case_id} has only {len(positive)} robust positive and "
            f"{len(negative)} halo-negative candidates"
        )
    generator = np.random.Generator(
        np.random.PCG64(_stable_seed(seed, "balanced-evaluation-patches", case.case_id))
    )
    selected_positive = generator.choice(positive, size=selected_count, replace=False)
    selected_negative = generator.choice(negative, size=selected_count, replace=False)
    records: list[EvaluationPatchRecord] = []
    for label, indices in ((0, selected_negative), (1, selected_positive)):
        for index in indices.tolist():
            records.append(
                EvaluationPatchRecord.create(
                    source=case.source,
                    release=case.release,
                    case_id=case.case_id,
                    subject_id=case.subject_id,
                    partition=partition,
                    center_mm=tuple(float(value) for value in centers[index]),
                    seg_positive_voxels=int(counts[index]),
                    crop_voxels=crop_voxels,
                    halo_clear=bool(clear[index]),
                    label=label,
                )
            )
    return tuple(sorted(records, key=lambda record: record.sample_id))


def _ordered_subjects(subjects: set[str], *, seed: int, purpose: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            subjects,
            key=lambda subject: (
                hashlib.sha256(
                    canonical_json_bytes({"seed": seed, "purpose": purpose, "subject_id": subject})
                ).hexdigest(),
                subject,
            ),
        )
    )


def _one_case_per_subject(
    cases: Sequence[CaseRecord], subjects: Sequence[str]
) -> tuple[CaseRecord, ...]:
    by_subject: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        by_subject[case.subject_id].append(case)
    result: list[CaseRecord] = []
    for subject in subjects:
        candidates = sorted(by_subject[subject], key=lambda case: case.key)
        if not candidates:
            raise PatchEvaluationError(f"selected subject {subject!r} has no cases")
        result.append(candidates[0])
    return tuple(result)


def build_evaluation_patch_manifest(
    *,
    data_root: str | os.PathLike[str],
    manifest: DatasetManifest,
    split: SplitManifest,
    case_grids: CaseGridManifest,
    segmentation_label_audit_path: str | os.PathLike[str],
    expected_segmentation_label_audit_sha256: str,
    patch_config: PatchConfig,
    probe_train_subject_count: int,
    maximum_patches_per_class_per_subject: int = 32,
    minimum_patches_per_class_per_subject: int = 4,
    seed: int = 0,
    label_rule: BinaryPatchLabelRule | None = None,
) -> EvaluationPatchManifest:
    """Materialize balanced 4mm patch locations without touching locked test images."""

    if not isinstance(manifest, DatasetManifest) or not isinstance(split, SplitManifest):
        raise TypeError("manifest and split must be DatasetManifest and SplitManifest")
    if not isinstance(case_grids, CaseGridManifest):
        raise TypeError("case_grids must be a CaseGridManifest")
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    partitions = partition_cases(manifest, split)
    if set(partitions) != {"train", "validation", "test"}:
        raise PatchEvaluationError("evaluation requires exactly train/validation/test splits")
    train_subjects = {case.subject_id for case in partitions["train"]}
    validation_subject_set = {case.subject_id for case in partitions["validation"]}
    test_subjects = {case.subject_id for case in partitions["test"]}
    if (
        isinstance(probe_train_subject_count, bool)
        or not isinstance(probe_train_subject_count, int)
        or probe_train_subject_count <= 0
        or probe_train_subject_count >= len(train_subjects)
    ):
        raise PatchEvaluationError(
            "probe_train_subject_count must select a nonempty strict subset of SSL-train subjects"
        )
    if (
        not isinstance(patch_config, PatchConfig)
        or not patch_config.is_cubic
        or (patch_config.footprint_mm != 4.0)
    ):
        raise PatchEvaluationError("primary evaluation requires the registered 4mm cube config")
    root = Path(data_root).expanduser()
    if root.is_symlink():
        raise PatchEvaluationError("data_root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise PatchEvaluationError("data_root is unavailable") from error
    if not root.is_dir():
        raise PatchEvaluationError("data_root must be a directory")
    ordered_probe_candidates = _ordered_subjects(
        train_subjects, seed=seed, purpose="probe-train-subject-order"
    )
    selected_validation = _ordered_subjects(
        validation_subject_set, seed=seed, purpose="validation-subject-order"
    )
    verified_label_audit = verify_segmentation_label_audit(
        segmentation_label_audit_path,
        expected_sha256=expected_segmentation_label_audit_sha256,
        manifest=manifest,
        split=split,
    )
    selected_rule = label_rule or BinaryPatchLabelRule()
    records: list[EvaluationPatchRecord] = []
    audits: list[SegmentationAuditRecord] = []
    selected_probe: list[str] = []
    ineligible_probe: list[str] = []
    crop_voxels = int(np.prod(patch_config.source_shape))

    cases_by_subject: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in (*partitions["train"], *partitions["validation"]):
        cases_by_subject[case.subject_id].append(case)

    def materialize_subject(subject: str, partition: str) -> bool:
        for case in sorted(cases_by_subject[subject], key=lambda item: item.key):
            spec = case_grids.extraction_spec_for_case(case, patch_config=patch_config)
            extractor = CachedNiftiPatchExtractor(
                data_root=root,
                manifest=manifest,
                data_manifest_sha256=manifest.sha256,
                extraction_spec=spec,
            )
            universe = prepare_case_candidate_universe(extractor, case)
            try:
                segmentation, audit = _load_segmentation(
                    data_root=root,
                    case=case,
                    case_grids=case_grids,
                    spec=spec,
                )
                counts, halo_clear = label_candidate_centers(
                    spec=spec,
                    centers_mm=universe.candidate_centers.values,
                    segmentation_mask=segmentation,
                    rule=selected_rule,
                )
                selected = select_balanced_patch_records(
                    case=case,
                    partition=partition,
                    centers_mm=universe.candidate_centers.values,
                    crop_counts=counts,
                    halo_clear=halo_clear,
                    rule=selected_rule,
                    seed=seed,
                    maximum_per_class=maximum_patches_per_class_per_subject,
                    minimum_per_class=minimum_patches_per_class_per_subject,
                    crop_voxels=crop_voxels,
                )
                if set(audit.observed_positive_values) - set(
                    verified_label_audit.numeric_label_values
                ):
                    raise PatchEvaluationError(
                        f"case {case.case_id} contains values absent from the label audit"
                    )
            except NoEligibleBinaryPatchesError:
                continue
            records.extend(selected)
            audits.append(audit)
            return True
        return False

    for subject in ordered_probe_candidates:
        if len(selected_probe) == probe_train_subject_count:
            break
        if materialize_subject(subject, "probe_train"):
            selected_probe.append(subject)
        else:
            ineligible_probe.append(subject)
    if len(selected_probe) != probe_train_subject_count:
        raise PatchEvaluationError("too few eligible SSL-train subjects for the probe pool")
    for subject in selected_validation:
        if not materialize_subject(subject, "validation"):
            raise PatchEvaluationError(
                f"validation subject {subject} has no eligible visit under the locked rule"
            )
    return EvaluationPatchManifest(
        data_manifest_sha256=manifest.sha256,
        subject_split_sha256=split.sha256,
        case_grid_manifest_sha256=case_grids.sha256,
        segmentation_label_audit_sha256=verified_label_audit.file_sha256,
        patch_config=patch_config,
        seed=seed,
        label_rule=selected_rule,
        probe_train_subjects=tuple(selected_probe),
        validation_subjects=selected_validation,
        ineligible_probe_train_subjects=tuple(ineligible_probe),
        locked_test_subject_count=len(test_subjects),
        segmentation_audit=tuple(audits),
        records=tuple(records),
    )


def save_evaluation_patch_manifest(
    manifest: EvaluationPatchManifest, path: str | os.PathLike[str]
) -> None:
    if not isinstance(manifest, EvaluationPatchManifest):
        raise TypeError("manifest must be an EvaluationPatchManifest")
    atomic_create_bytes(path, canonical_json_bytes(manifest.to_dict()))


def load_evaluation_patch_manifest(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> EvaluationPatchManifest:
    import json

    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PatchEvaluationError(f"could not load evaluation patch manifest: {error}") from error
    if not isinstance(value, Mapping):
        raise PatchEvaluationError("evaluation patch manifest must be an object")
    result = EvaluationPatchManifest.from_dict(value)
    if expected_sha256 is not None and result.sha256 != expected_sha256:
        raise PatchEvaluationError(
            f"evaluation patch SHA mismatch: expected {expected_sha256}, got {result.sha256}"
        )
    return result


__all__ = [
    "EVALUATION_PATCH_SCHEMA",
    "EVALUATION_PATCH_SCHEMA_VERSION",
    "BinaryPatchLabelRule",
    "EvaluationPatchManifest",
    "EvaluationPatchRecord",
    "PatchEvaluationError",
    "SegmentationAuditRecord",
    "VerifiedSegmentationLabelAudit",
    "build_evaluation_patch_manifest",
    "label_candidate_centers",
    "load_evaluation_patch_manifest",
    "save_evaluation_patch_manifest",
    "select_balanced_patch_records",
    "verify_segmentation_label_audit",
]
