"""Per-case physical grids for heterogeneous, co-registered MRI releases.

The global extraction policy deliberately contains no patient array shape or
scanner-world origin.  Each case retains its exact RAS native grid and derives
its own axis-aligned 1 mm prepared grid from native voxel-cell bounds.  The
four MRI modalities must agree exactly within a case, while unrelated cases
may differ in shape, spacing, origin, or orientation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np

from simple_brats.config import MODALITIES

from .extraction import ExtractionSpec
from .manifest import (
    CaseRecord,
    DatasetManifest,
    canonical_json_bytes,
    sha256_file,
)

CASE_GRID_SCHEMA = "simple_brats.case_grid_manifest"
CASE_GRID_SCHEMA_VERSION = 1
EXTRACTION_POLICY_SCHEMA = "simple_brats.extraction_policy"
EXTRACTION_POLICY_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUND_ROUNDING_TOLERANCE = 1e-7


class CaseGridError(ValueError):
    """A case cannot satisfy the pinned physical-grid contract."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CaseGridError(f"{field} must be a non-empty string without surrounding whitespace")
    if any(ord(character) < 32 for character in value):
        raise CaseGridError(f"{field} must not contain control characters")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _required_text(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise CaseGridError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _exact_keys(value: Mapping[str, object], expected: set[str], description: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CaseGridError(
            f"invalid {description} keys: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _decode_json(payload: str | bytes | bytearray) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CaseGridError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> object:
        raise CaseGridError(f"non-finite JSON number {token!r} is forbidden")

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except CaseGridError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise CaseGridError(f"invalid case-grid JSON: {error}") from error


def _shape3(value: object, field: str) -> tuple[int, int, int]:
    try:
        shape = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise CaseGridError(f"{field} must contain three positive integers") from error
    if len(shape) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape
    ):
        raise CaseGridError(f"{field} must contain three positive integers")
    return shape  # type: ignore[return-value]


def _float3(value: object, field: str, *, positive: bool = False) -> tuple[float, float, float]:
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise CaseGridError(f"{field} must contain three finite numbers") from error
    if len(raw) != 3 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw
    ):
        raise CaseGridError(f"{field} must contain three finite numbers")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise CaseGridError(f"{field} must contain three finite numbers")
    if positive and any(item <= 0 for item in result):
        raise CaseGridError(f"{field} must contain three positive numbers")
    return tuple(0.0 if item == 0.0 else item for item in result)  # type: ignore[return-value]


def _affine4(value: object, field: str) -> tuple[tuple[float, ...], ...]:
    try:
        affine = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CaseGridError(f"{field} must be a finite numeric 4x4 matrix") from error
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise CaseGridError(f"{field} must be a finite numeric 4x4 matrix")
    if not np.array_equal(affine[3], np.asarray((0.0, 0.0, 0.0, 1.0))):
        raise CaseGridError(f"{field} must end in homogeneous row [0, 0, 0, 1]")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-10:
        raise CaseGridError(f"{field} must be nonsingular")
    return tuple(
        tuple(0.0 if float(item) == 0.0 else float(item) for item in row) for row in affine
    )


@dataclass(frozen=True, slots=True)
class SpatialGrid:
    """One RAS/mm array grid whose affine maps voxel centers to world mm."""

    shape: tuple[int, int, int]
    affine: tuple[tuple[float, ...], ...]
    orientation: str = "RAS+"
    spatial_unit: str = "mm"

    def __post_init__(self) -> None:
        shape = _shape3(self.shape, "grid.shape")
        affine = _affine4(self.affine, "grid.affine")
        if self.orientation != "RAS+":
            raise CaseGridError("grid.orientation must be 'RAS+'")
        if self.spatial_unit != "mm":
            raise CaseGridError("grid.spatial_unit must be 'mm'")
        if tuple(nib.aff2axcodes(np.asarray(affine))) != ("R", "A", "S"):
            raise CaseGridError("grid affine must use RAS anatomical axes")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "affine", affine)

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        linear = np.asarray(self.affine, dtype=np.float64)[:3, :3]
        return tuple(float(value) for value in np.linalg.norm(linear, axis=0))  # type: ignore[return-value]

    @property
    def voxel_cell_bounds_mm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        corners = np.asarray(
            [
                (*corner, 1.0)
                for corner in itertools.product(*[(-0.5, float(size) - 0.5) for size in self.shape])
            ],
            dtype=np.float64,
        )
        world = (np.asarray(self.affine, dtype=np.float64) @ corners.T).T[:, :3]
        lower = tuple(float(value) for value in world.min(axis=0))
        upper = tuple(float(value) for value in world.max(axis=0))
        return lower, upper  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "affine": [list(row) for row in self.affine],
            "orientation": self.orientation,
            "spatial_unit": self.spatial_unit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SpatialGrid:
        _exact_keys(value, {"shape", "affine", "orientation", "spatial_unit"}, "spatial grid")
        return cls(
            shape=value["shape"],  # type: ignore[arg-type]
            affine=value["affine"],  # type: ignore[arg-type]
            orientation=value["orientation"],  # type: ignore[arg-type]
            spatial_unit=value["spatial_unit"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ExtractionPolicy:
    """Global transform policy with no patient-specific shape or origin."""

    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    within_case_registration: str = "exact-shape-and-affine-after-RAS"
    source_spatial_unit_policy: str = "mm-or-unknown-after-case-mm-consensus"
    prepared_bounds: str = "axis-aligned-native-voxel-cell-bounds"
    volume_interpolation: str = "trilinear"
    volume_align_corners: bool = True
    volume_padding: str = "zeros-with-invalid-support"
    foreground_rule: str = "four-modality-valid-and-nonzero-intersection"
    normalization: str = "per-modality-foreground-zscore-clip[-5,5]"
    patch_source_shape: tuple[int, int, int] = (4, 4, 1)
    patch_physical_extent_mm: tuple[float, float, float] = (4.0, 4.0, 1.0)
    model_visible_shape: tuple[int, int, int] = (16, 16, 1)
    patch_interpolation: str = "trilinear-align-corners-false"
    world_origin_policy: str = "preserve-case-physical-bounds"
    schema: str = EXTRACTION_POLICY_SCHEMA
    schema_version: int = EXTRACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        spacing = _float3(self.target_spacing_mm, "target_spacing_mm", positive=True)
        if spacing != (1.0, 1.0, 1.0):
            raise CaseGridError("v0 target_spacing_mm must be exactly [1, 1, 1]")
        source_shape = _shape3(self.patch_source_shape, "patch_source_shape")
        model_shape = _shape3(self.model_visible_shape, "model_visible_shape")
        extent = _float3(
            self.patch_physical_extent_mm,
            "patch_physical_extent_mm",
            positive=True,
        )
        fixed = {
            "within_case_registration": "exact-shape-and-affine-after-RAS",
            "source_spatial_unit_policy": "mm-or-unknown-after-case-mm-consensus",
            "prepared_bounds": "axis-aligned-native-voxel-cell-bounds",
            "volume_interpolation": "trilinear",
            "volume_padding": "zeros-with-invalid-support",
            "foreground_rule": "four-modality-valid-and-nonzero-intersection",
            "normalization": "per-modality-foreground-zscore-clip[-5,5]",
            "patch_interpolation": "trilinear-align-corners-false",
            "world_origin_policy": "preserve-case-physical-bounds",
            "schema": EXTRACTION_POLICY_SCHEMA,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise CaseGridError(f"v0 requires {name}={expected!r}")
        if self.volume_align_corners is not True:
            raise CaseGridError("v0 volume_align_corners must be true")
        if source_shape != (4, 4, 1) or extent != (4.0, 4.0, 1.0):
            raise CaseGridError("v0 patch source must be 4x4x1 voxels / mm")
        if model_shape != (16, 16, 1):
            raise CaseGridError("v0 model_visible_shape must be 16x16x1")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != EXTRACTION_POLICY_SCHEMA_VERSION
        ):
            raise CaseGridError("unsupported extraction-policy schema version")
        object.__setattr__(self, "target_spacing_mm", spacing)
        object.__setattr__(self, "patch_source_shape", source_shape)
        object.__setattr__(self, "patch_physical_extent_mm", extent)
        object.__setattr__(self, "model_visible_shape", model_shape)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "target_spacing_mm": list(self.target_spacing_mm),
            "within_case_registration": self.within_case_registration,
            "source_spatial_unit_policy": self.source_spatial_unit_policy,
            "prepared_bounds": self.prepared_bounds,
            "volume_interpolation": self.volume_interpolation,
            "volume_align_corners": self.volume_align_corners,
            "volume_padding": self.volume_padding,
            "foreground_rule": self.foreground_rule,
            "normalization": self.normalization,
            "patch_source_shape": list(self.patch_source_shape),
            "patch_physical_extent_mm": list(self.patch_physical_extent_mm),
            "model_visible_shape": list(self.model_visible_shape),
            "patch_interpolation": self.patch_interpolation,
            "world_origin_policy": self.world_origin_policy,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def extraction_spec(self, prepared_grid: SpatialGrid) -> ExtractionSpec:
        if not isinstance(prepared_grid, SpatialGrid):
            raise TypeError("prepared_grid must be a SpatialGrid")
        if not np.allclose(
            prepared_grid.spacing_mm,
            self.target_spacing_mm,
            atol=1e-7,
            rtol=0,
        ):
            raise CaseGridError("prepared grid spacing differs from the extraction policy")
        return ExtractionSpec(
            canonical_shape=prepared_grid.shape,
            canonical_affine=prepared_grid.affine,  # type: ignore[arg-type]
            source_spatial_unit_policy=self.source_spatial_unit_policy,
            world_origin_policy=self.world_origin_policy,
            patch_source_shape=self.patch_source_shape,
            patch_physical_extent_mm=self.patch_physical_extent_mm,
            model_visible_shape=self.model_visible_shape,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExtractionPolicy:
        expected = set(cls().to_dict())
        _exact_keys(value, expected, "extraction policy")
        return cls(**value)  # type: ignore[arg-type]


def derive_prepared_grid(native_grid: SpatialGrid, policy: ExtractionPolicy) -> SpatialGrid:
    """Cover native voxel cells with a deterministic case-local target grid."""

    if not isinstance(native_grid, SpatialGrid) or not isinstance(policy, ExtractionPolicy):
        raise TypeError("native_grid and policy must be SpatialGrid and ExtractionPolicy")
    lower, upper = native_grid.voxel_cell_bounds_mm
    spacing = np.asarray(policy.target_spacing_mm, dtype=np.float64)
    extent = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    ratios = extent / spacing
    nearest = np.rint(ratios)
    tolerance = _BOUND_ROUNDING_TOLERANCE * np.maximum(1.0, np.abs(nearest))
    ratios = np.where(
        np.abs(ratios - nearest) <= tolerance,
        nearest,
        ratios,
    )
    shape_array = np.ceil(ratios).astype(np.int64)
    if np.any(shape_array <= 0):
        raise CaseGridError("native physical bounds cannot produce a non-empty prepared grid")
    origin = np.asarray(lower, dtype=np.float64) + 0.5 * spacing
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = origin
    return SpatialGrid(
        shape=tuple(int(value) for value in shape_array),  # type: ignore[arg-type]
        affine=tuple(tuple(float(value) for value in row) for row in affine),
    )


@dataclass(frozen=True, slots=True)
class CaseGridRecord:
    """One full manifest case bound to native and prepared physical grids."""

    data_manifest_sha256: str
    case: CaseRecord
    declared_spatial_units: tuple[str, str, str, str]
    extraction_policy_sha256: str
    native_grid: SpatialGrid
    prepared_grid: SpatialGrid
    extraction_spec_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_manifest_sha256",
            _sha256(self.data_manifest_sha256, "data_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "extraction_policy_sha256",
            _sha256(self.extraction_policy_sha256, "extraction_policy_sha256"),
        )
        object.__setattr__(
            self,
            "extraction_spec_sha256",
            _sha256(self.extraction_spec_sha256, "extraction_spec_sha256"),
        )
        if not isinstance(self.case, CaseRecord):
            raise TypeError("case must be a CaseRecord")
        try:
            units = tuple(self.declared_spatial_units)
        except TypeError as error:
            raise CaseGridError("declared_spatial_units must contain four strings") from error
        if len(units) != len(MODALITIES) or any(
            not isinstance(unit, str) or unit not in {"mm", "unknown"} for unit in units
        ):
            raise CaseGridError(
                "declared_spatial_units must contain one 'mm' or 'unknown' value per modality"
            )
        if "mm" not in units:
            raise CaseGridError(
                "at least one MRI modality must explicitly declare mm spatial units"
            )
        if not isinstance(self.native_grid, SpatialGrid) or not isinstance(
            self.prepared_grid, SpatialGrid
        ):
            raise TypeError("native_grid and prepared_grid must be SpatialGrid records")
        object.__setattr__(self, "declared_spatial_units", units)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.case.key

    def to_dict(self) -> dict[str, object]:
        return {
            "data_manifest_sha256": self.data_manifest_sha256,
            "case": self.case.to_dict(),
            "declared_spatial_units": list(self.declared_spatial_units),
            "extraction_policy_sha256": self.extraction_policy_sha256,
            "native_grid": self.native_grid.to_dict(),
            "prepared_grid": self.prepared_grid.to_dict(),
            "extraction_spec_sha256": self.extraction_spec_sha256,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CaseGridRecord:
        _exact_keys(
            value,
            {
                "data_manifest_sha256",
                "case",
                "declared_spatial_units",
                "extraction_policy_sha256",
                "native_grid",
                "prepared_grid",
                "extraction_spec_sha256",
            },
            "case-grid record",
        )
        if not isinstance(value["case"], Mapping):
            raise CaseGridError("case-grid case must be an object")
        if not isinstance(value["native_grid"], Mapping) or not isinstance(
            value["prepared_grid"], Mapping
        ):
            raise CaseGridError("case-grid native/prepared grids must be objects")
        return cls(
            data_manifest_sha256=value["data_manifest_sha256"],  # type: ignore[arg-type]
            case=CaseRecord.from_dict(value["case"]),
            declared_spatial_units=value["declared_spatial_units"],  # type: ignore[arg-type]
            extraction_policy_sha256=value["extraction_policy_sha256"],  # type: ignore[arg-type]
            native_grid=SpatialGrid.from_dict(value["native_grid"]),
            prepared_grid=SpatialGrid.from_dict(value["prepared_grid"]),
            extraction_spec_sha256=value["extraction_spec_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CaseGridManifest:
    """Canonical catalog of per-case grids under one extraction policy."""

    data_manifest_sha256: str
    policy: ExtractionPolicy
    records: tuple[CaseGridRecord, ...]
    schema: str = CASE_GRID_SCHEMA
    schema_version: int = CASE_GRID_SCHEMA_VERSION

    def __post_init__(self) -> None:
        manifest_sha = _sha256(self.data_manifest_sha256, "data_manifest_sha256")
        if not isinstance(self.policy, ExtractionPolicy):
            raise TypeError("policy must be an ExtractionPolicy")
        records = tuple(self.records)
        if not records or not all(isinstance(record, CaseGridRecord) for record in records):
            raise CaseGridError("records must contain at least one CaseGridRecord")
        if self.schema != CASE_GRID_SCHEMA or (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CASE_GRID_SCHEMA_VERSION
        ):
            raise CaseGridError("unsupported case-grid manifest schema")
        keys = [record.key for record in records]
        if len(keys) != len(set(keys)):
            raise CaseGridError("case-grid records must have unique case keys")
        for record in records:
            if record.data_manifest_sha256 != manifest_sha:
                raise CaseGridError("case-grid record is bound to a different data manifest")
            if record.extraction_policy_sha256 != self.policy.sha256:
                raise CaseGridError("case-grid record is bound to a different extraction policy")
            expected_prepared = derive_prepared_grid(record.native_grid, self.policy)
            if record.prepared_grid != expected_prepared:
                raise CaseGridError("case-grid prepared grid is not deterministically derived")
            expected_spec_sha = self.policy.extraction_spec(record.prepared_grid).sha256
            if record.extraction_spec_sha256 != expected_spec_sha:
                raise CaseGridError("case-grid extraction spec SHA is inconsistent")
        object.__setattr__(self, "data_manifest_sha256", manifest_sha)
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda record: record.key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "data_manifest_sha256": self.data_manifest_sha256,
            "policy": self.policy.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def validate_manifest(self, manifest: DatasetManifest) -> None:
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        if manifest.sha256 != self.data_manifest_sha256:
            raise CaseGridError("case-grid catalog is bound to a different data manifest")
        records_by_key = {record.key: record.case for record in self.records}
        manifest_by_key = {case.key: case for case in manifest.cases}
        if records_by_key != manifest_by_key:
            raise CaseGridError("case-grid catalog does not exactly cover manifest cases")

    def record_for_case(self, case: CaseRecord) -> CaseGridRecord:
        if not isinstance(case, CaseRecord):
            raise TypeError("case must be a CaseRecord")
        matches = tuple(record for record in self.records if record.key == case.key)
        if len(matches) != 1 or matches[0].case != case:
            raise CaseGridError("case does not exactly match one case-grid record")
        return matches[0]

    def extraction_spec_for_case(self, case: CaseRecord) -> ExtractionSpec:
        record = self.record_for_case(case)
        return self.policy.extraction_spec(record.prepared_grid)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CaseGridManifest:
        _exact_keys(
            value,
            {"schema", "schema_version", "data_manifest_sha256", "policy", "records"},
            "case-grid manifest",
        )
        if not isinstance(value["policy"], Mapping):
            raise CaseGridError("case-grid policy must be an object")
        raw_records = value["records"]
        if not isinstance(raw_records, list) or not all(
            isinstance(record, Mapping) for record in raw_records
        ):
            raise CaseGridError("case-grid records must be an array of objects")
        return cls(
            schema=value["schema"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            data_manifest_sha256=value["data_manifest_sha256"],  # type: ignore[arg-type]
            policy=ExtractionPolicy.from_dict(value["policy"]),
            records=tuple(CaseGridRecord.from_dict(record) for record in raw_records),
        )


def _resolve_root(data_root: str | os.PathLike[str]) -> Path:
    root = Path(data_root).expanduser()
    if root.is_symlink():
        raise CaseGridError("data_root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise CaseGridError(f"data_root is unavailable: {data_root}") from error
    if not root.is_dir():
        raise CaseGridError("data_root must be a directory")
    return root


def _resolve_manifest_path(root: Path, recorded_path: str) -> Path:
    relative = PurePosixPath(recorded_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != recorded_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CaseGridError(f"manifest path is not canonical and relative: {recorded_path!r}")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise CaseGridError(f"manifest path traverses a symlink: {recorded_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise CaseGridError(f"manifest path escapes or is missing: {recorded_path}") from error
    if not resolved.is_file():
        raise CaseGridError(f"manifest path is not a regular file: {recorded_path}")
    return resolved


def _audit_case_native_grid(
    case: CaseRecord,
    root: Path,
) -> tuple[SpatialGrid, tuple[str, str, str, str]]:
    files = {record.modality: record for record in case.files}
    missing = sorted(set(MODALITIES) - set(files))
    if missing:
        raise CaseGridError(f"case {case.case_id} is missing MRI modalities {missing}")
    reference: SpatialGrid | None = None
    reference_modality: str | None = None
    declared_units: list[str] = []
    for modality in MODALITIES:
        record = files[modality]
        path = _resolve_manifest_path(root, record.path)
        actual_sha = sha256_file(path)
        if actual_sha != record.sha256:
            raise CaseGridError(
                f"manifest file bytes changed for {case.case_id}/{modality}: "
                f"expected {record.sha256}, got {actual_sha}"
            )
        try:
            image = nib.load(os.fspath(path), mmap="r")
            if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
                raise CaseGridError(f"{case.case_id}/{modality} is not one non-empty 3D MRI")
            spatial_unit, _ = image.header.get_xyzt_units()
            if spatial_unit not in {"mm", "unknown"}:
                raise CaseGridError(
                    f"{case.case_id}/{modality} has unsupported spatial unit {spatial_unit!r}"
                )
            declared_units.append(spatial_unit)
            canonical = nib.as_closest_canonical(image, enforce_diag=False)
            grid = SpatialGrid(
                shape=tuple(int(size) for size in canonical.shape),
                affine=tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(canonical.affine, dtype=np.float64)
                ),
            )
        except CaseGridError:
            raise
        except Exception as error:
            raise CaseGridError(f"could not audit {case.case_id}/{modality}: {error}") from error
        if reference is None:
            reference = grid
            reference_modality = modality
        elif grid != reference:
            raise CaseGridError(
                f"case {case.case_id} modalities are not exactly registered after RAS: "
                f"{modality} grid differs from {reference_modality}"
            )
    if reference is None:
        raise AssertionError("case-grid audit did not inspect any MRI")
    if "mm" not in declared_units:
        raise CaseGridError(
            f"case {case.case_id} has no MRI modality that explicitly declares mm units"
        )
    return reference, tuple(declared_units)  # type: ignore[return-value]


def audit_case_grids(
    manifest: DatasetManifest,
    data_root: str | os.PathLike[str],
    *,
    policy: ExtractionPolicy | None = None,
) -> CaseGridManifest:
    """Hash and audit every case header, allowing arbitrary cross-case grids."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    selected_policy = ExtractionPolicy() if policy is None else policy
    if not isinstance(selected_policy, ExtractionPolicy):
        raise TypeError("policy must be an ExtractionPolicy")
    root = _resolve_root(data_root)
    records: list[CaseGridRecord] = []
    for case in manifest.cases:
        native, declared_units = _audit_case_native_grid(case, root)
        prepared = derive_prepared_grid(native, selected_policy)
        spec = selected_policy.extraction_spec(prepared)
        records.append(
            CaseGridRecord(
                data_manifest_sha256=manifest.sha256,
                case=case,
                declared_spatial_units=declared_units,
                extraction_policy_sha256=selected_policy.sha256,
                native_grid=native,
                prepared_grid=prepared,
                extraction_spec_sha256=spec.sha256,
            )
        )
    catalog = CaseGridManifest(
        data_manifest_sha256=manifest.sha256,
        policy=selected_policy,
        records=tuple(records),
    )
    catalog.validate_manifest(manifest)
    return catalog


def save_case_grid_manifest(
    manifest: CaseGridManifest,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(manifest, CaseGridManifest):
        raise TypeError("manifest must be a CaseGridManifest")
    destination = Path(path)
    mode = "wb" if overwrite else "xb"
    with destination.open(mode) as handle:
        handle.write(canonical_json_bytes(manifest.to_dict()))


def load_case_grid_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> CaseGridManifest:
    raw = Path(path).read_bytes()
    value = _decode_json(raw)
    if not isinstance(value, Mapping):
        raise CaseGridError("case-grid manifest must be a JSON object")
    manifest = CaseGridManifest.from_dict(value)
    if raw != canonical_json_bytes(manifest.to_dict()):
        raise CaseGridError("case-grid manifest JSON is not in canonical byte form")
    if expected_sha256 is not None and manifest.sha256 != _sha256(
        expected_sha256, "expected_sha256"
    ):
        raise CaseGridError(
            f"case-grid SHA mismatch: expected {expected_sha256}, got {manifest.sha256}"
        )
    return manifest


__all__ = [
    "CASE_GRID_SCHEMA",
    "CASE_GRID_SCHEMA_VERSION",
    "EXTRACTION_POLICY_SCHEMA",
    "EXTRACTION_POLICY_SCHEMA_VERSION",
    "CaseGridError",
    "CaseGridManifest",
    "CaseGridRecord",
    "ExtractionPolicy",
    "SpatialGrid",
    "audit_case_grids",
    "derive_prepared_grid",
    "load_case_grid_manifest",
    "save_case_grid_manifest",
]
