"""Manifest-bound, cached real-data preparation for matching experiments.

The lower-level extraction module intentionally accepts a filesystem path.  A
training process needs a stricter boundary: only paths and digests recorded in
the pinned manifest may be loaded, every path must remain below the dataset
root without traversing a symlink, and raw bytes must be verified before they
can influence a canonical volume.  This module provides that boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from simple_brats.config import MODALITIES

from .extraction import (
    CanonicalVolume,
    ExtractionError,
    ExtractionSpec,
    extract_patch,
    intersect_modality_foreground_support_masks,
    prepare_canonical_volume,
    valid_patch_centers_mm,
)
from .manifest import (
    CaseRecord,
    DatasetManifest,
    FileRecord,
    canonical_json_bytes,
    sha256_file,
)
from .plan_factory import CanonicalCandidateCenters, materialize_matching_plan

if TYPE_CHECKING:
    from simple_brats.sampling import MaterializedPatchPlan, SlabGeometry

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OPTIONAL_NON_IMAGE_MODALITIES = frozenset({"seg"})


class DataPipelineError(ValueError):
    """Raised when real data cannot satisfy its pinned provenance contract."""


@dataclass(frozen=True, slots=True)
class CanonicalVolumeDigest:
    """Auditable raw and prepared identities for one MRI modality."""

    modality: str
    raw_file_sha256: str
    canonical_voxel_sha256: str
    normalized_voxel_sha256: str

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise DataPipelineError(f"unrecognized canonical-volume modality {self.modality!r}")
        for digest_field in (
            "raw_file_sha256",
            "canonical_voxel_sha256",
            "normalized_voxel_sha256",
        ):
            _pinned_sha256(getattr(self, digest_field), digest_field)

    def to_dict(self) -> dict[str, str]:
        return {
            "modality": self.modality,
            "raw_file_sha256": self.raw_file_sha256,
            "canonical_voxel_sha256": self.canonical_voxel_sha256,
            "normalized_voxel_sha256": self.normalized_voxel_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PreparedCaseCandidateUniverse:
    """Case-invariant safe centers bound to every input that can change them.

    This is an in-memory preparation record, not a serialized experiment
    artifact.  Per-bag plans continue to use :class:`PreparedCasePlan` and its
    existing versioned JSON schema.
    """

    case: CaseRecord
    data_manifest_sha256: str
    extraction_spec_sha256: str
    geometry_sha256: str
    candidate_centers: CanonicalCandidateCenters = field(repr=False)
    candidate_count: int
    candidate_centers_sha256: str
    volume_digests: tuple[CanonicalVolumeDigest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case, CaseRecord):
            raise TypeError("case must be a CaseRecord")
        for name in (
            "data_manifest_sha256",
            "extraction_spec_sha256",
            "geometry_sha256",
            "candidate_centers_sha256",
        ):
            _pinned_sha256(getattr(self, name), name)
        if not isinstance(self.candidate_centers, CanonicalCandidateCenters):
            raise TypeError("candidate_centers must be CanonicalCandidateCenters")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count <= 0
            or self.candidate_count != len(self.candidate_centers)
        ):
            raise DataPipelineError(
                "candidate_count must equal the positive immutable candidate-center count"
            )
        actual_centers_sha256 = _candidate_centers_sha256(self.candidate_centers.values)
        if actual_centers_sha256 != self.candidate_centers_sha256:
            raise DataPipelineError("candidate centers do not match candidate_centers_sha256")

        digests = tuple(self.volume_digests)
        if (
            not all(isinstance(item, CanonicalVolumeDigest) for item in digests)
            or tuple(item.modality for item in digests) != MODALITIES
        ):
            raise DataPipelineError(
                f"volume_digests must use canonical modality order {MODALITIES}"
            )
        files_by_modality = {record.modality: record for record in self.case.files}
        if any(
            files_by_modality.get(item.modality) is None
            or files_by_modality[item.modality].sha256 != item.raw_file_sha256
            for item in digests
        ):
            raise DataPipelineError("volume digests do not match the bound case's raw MRI files")
        object.__setattr__(self, "volume_digests", digests)


@dataclass(frozen=True, slots=True)
class PreparedCasePlan:
    """A plan plus the complete label-free candidate-generation audit record.

    The materialized plan remains the replay source of truth.  This companion
    digest proves which eligible candidate universe and prepared volumes were
    used to choose it, without serializing a potentially very large center
    array into every plan file.
    """

    plan: MaterializedPatchPlan
    candidate_count: int
    candidate_centers_sha256: str
    volume_digests: tuple[CanonicalVolumeDigest, ...]

    def __post_init__(self) -> None:
        from simple_brats.sampling import MaterializedPatchPlan

        if not isinstance(self.plan, MaterializedPatchPlan):
            raise TypeError("plan must be a MaterializedPatchPlan")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count <= 0
        ):
            raise DataPipelineError("candidate_count must be a positive integer")
        _pinned_sha256(self.candidate_centers_sha256, "candidate_centers_sha256")
        digests = tuple(self.volume_digests)
        if not all(isinstance(item, CanonicalVolumeDigest) for item in digests):
            raise TypeError("volume_digests must contain CanonicalVolumeDigest records")
        if tuple(item.modality for item in digests) != MODALITIES:
            raise DataPipelineError(
                f"volume_digests must use canonical modality order {MODALITIES}"
            )
        object.__setattr__(self, "volume_digests", digests)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "simple-brats.prepared-case-plan",
            "schema_version": 1,
            "case": self.plan.case.to_dict(),
            "data_manifest_sha256": self.plan.data_manifest_sha256,
            "extraction_spec_sha256": self.plan.extraction_spec_sha256,
            "plan_sha256": self.plan.sha256,
            "candidate_count": self.candidate_count,
            "candidate_centers_sha256": self.candidate_centers_sha256,
            "volume_digests": [item.to_dict() for item in self.volume_digests],
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _canonical_relative_path(recorded_path: object) -> PurePosixPath:
    if not isinstance(recorded_path, str) or not recorded_path:
        raise DataPipelineError("manifest image path must be a non-empty string")
    if "\\" in recorded_path:
        raise DataPipelineError(
            f"manifest image path must use canonical POSIX separators: {recorded_path!r}"
        )
    path = PurePosixPath(recorded_path)
    if (
        path.is_absolute()
        or path.as_posix() != recorded_path
        or path == PurePosixPath(".")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DataPipelineError(
            f"manifest image path must be canonical and data-root-relative: {recorded_path!r}"
        )
    return path


def _resolve_data_root(data_root: str | os.PathLike[str]) -> Path:
    candidate = Path(data_root).expanduser()
    if candidate.is_symlink():
        raise DataPipelineError("data_root must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise DataPipelineError(f"data_root is unavailable: {data_root}") from error
    if not resolved.is_dir():
        raise DataPipelineError("data_root must be a directory")
    return resolved


def _resolve_manifest_file(data_root: Path, recorded_path: str) -> Path:
    relative = _canonical_relative_path(recorded_path)
    candidate = data_root
    for component in relative.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise DataPipelineError(f"manifest image path traverses a symlink: {recorded_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(data_root)
    except (OSError, ValueError) as error:
        raise DataPipelineError(
            f"manifest image path escapes the data root or is missing: {recorded_path}"
        ) from error
    if not resolved.is_file():
        raise DataPipelineError(f"manifest image path is not a regular file: {recorded_path}")
    return resolved


def _pinned_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DataPipelineError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _candidate_centers_sha256(candidate_centers: np.ndarray) -> str:
    centers = np.asarray(candidate_centers)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or not np.isfinite(centers).all():
        raise DataPipelineError("candidate centers must be a finite Nx3 array")
    canonical = np.array(centers, dtype=np.dtype("<f8"), order="C", copy=True)
    canonical[canonical == 0] = 0.0
    metadata = {
        "schema": "simple-brats.candidate-centers",
        "schema_version": 1,
        "shape": list(canonical.shape),
        "dtype": "little-endian-float64",
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


class CachedNiftiPatchExtractor:
    """Load only manifest-approved MRI files and cache canonical volumes.

    The cache is intentionally small and process-local.  Its key is the exact
    canonical manifest-relative path; a cache entry is inserted only after the
    raw file hashes to the manifest digest both before and after preparation.
    The second check closes the ordinary file-mutation window while nibabel is
    reading the image.
    """

    def __init__(
        self,
        *,
        data_root: str | os.PathLike[str],
        manifest: DatasetManifest,
        data_manifest_sha256: str,
        extraction_spec: ExtractionSpec,
        max_cached_volumes: int = 4,
    ) -> None:
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        if not isinstance(extraction_spec, ExtractionSpec):
            raise TypeError("extraction_spec must be an ExtractionSpec")
        manifest_pin = _pinned_sha256(data_manifest_sha256, "data_manifest_sha256")
        if manifest.sha256 != manifest_pin:
            raise DataPipelineError("manifest does not match the pinned data_manifest_sha256")
        if (
            isinstance(max_cached_volumes, bool)
            or not isinstance(max_cached_volumes, int)
            or max_cached_volumes <= 0
        ):
            raise ValueError("max_cached_volumes must be a positive integer")

        self._data_root = _resolve_data_root(data_root)
        self._manifest = manifest
        self.data_manifest_sha256 = manifest_pin
        self.extraction_spec = extraction_spec
        self.extraction_spec_sha256 = extraction_spec.sha256
        self._max_cached_volumes = max_cached_volumes
        self._cache: OrderedDict[str, CanonicalVolume] = OrderedDict()
        self._cache_lock = threading.RLock()

        records_by_path: dict[str, FileRecord] = {}
        cases_by_key: dict[tuple[str, str, str], CaseRecord] = {}
        allowed_case_modalities = set(MODALITIES) | _OPTIONAL_NON_IMAGE_MODALITIES
        for case in manifest.cases:
            files_by_modality = {record.modality: record for record in case.files}
            image_modalities = set(files_by_modality) & set(MODALITIES)
            if image_modalities != set(MODALITIES):
                missing = sorted(set(MODALITIES) - image_modalities)
                raise DataPipelineError(
                    f"case {case.case_id} must contain exactly the four v0 MRI modalities; "
                    f"missing={missing}"
                )
            unexpected = sorted(set(files_by_modality) - allowed_case_modalities)
            if unexpected:
                raise DataPipelineError(
                    f"case {case.case_id} contains unreviewed modalities: {unexpected}"
                )
            cases_by_key[case.key] = case
            for record in case.files:
                _canonical_relative_path(record.path)
                previous = records_by_path.get(record.path)
                if previous is not None:
                    raise DataPipelineError(
                        f"manifest path is reused by multiple records: {record.path}"
                    )
                records_by_path[record.path] = record

        self._records_by_path = records_by_path
        self._cases_by_key = cases_by_key

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def cache_size(self) -> int:
        with self._cache_lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def _validate_geometry(self, geometry: object) -> SlabGeometry:
        # Import lazily so exporting this module from simple_brats.data cannot
        # create a package-initialization cycle through sampling.records.
        from simple_brats.sampling import SlabGeometry

        if not isinstance(geometry, SlabGeometry):
            raise TypeError("geometry must be a SlabGeometry")
        expected = SlabGeometry(
            in_plane_axes=(0, 1),
            thin_axis=2,
            in_plane_footprint_mm=self.extraction_spec.patch_physical_extent_mm[0],
            thin_extent_mm=self.extraction_spec.patch_physical_extent_mm[2],
            model_shape=self.extraction_spec.model_visible_shape,
        )
        if geometry != expected:
            raise DataPipelineError(
                "patch geometry does not match the pinned extraction specification"
            )
        return geometry

    def _registered_record(
        self,
        *,
        path: object,
        file_sha256: object,
        modality: object,
    ) -> FileRecord:
        if not isinstance(modality, str) or modality not in MODALITIES:
            raise DataPipelineError(
                f"modality must be one of the four v0 MRI modalities {MODALITIES}"
            )
        if not isinstance(path, str):
            raise DataPipelineError("path must be the exact manifest-relative string")
        _canonical_relative_path(path)
        expected_digest = _pinned_sha256(file_sha256, "file_sha256")
        record = self._records_by_path.get(path)
        if record is None:
            raise DataPipelineError(f"path is not registered in the pinned manifest: {path}")
        if record.modality != modality:
            raise DataPipelineError(
                f"manifest path {path!r} is registered for {record.modality!r}, not {modality!r}"
            )
        if record.sha256 != expected_digest:
            raise DataPipelineError(
                f"file_sha256 does not match the pinned manifest record for {path}"
            )
        return record

    def _load_registered_volume(self, record: FileRecord) -> CanonicalVolume:
        with self._cache_lock:
            cached = self._cache.get(record.path)
            if cached is not None:
                self._cache.move_to_end(record.path)
                return cached

            filesystem_path = _resolve_manifest_file(self._data_root, record.path)
            try:
                digest_before = sha256_file(filesystem_path)
            except OSError as error:
                raise DataPipelineError(
                    f"unable to hash manifest image before loading: {record.path}"
                ) from error
            if digest_before != record.sha256:
                raise DataPipelineError(
                    f"raw file SHA mismatch before loading {record.path}: "
                    f"expected {record.sha256}, got {digest_before}"
                )

            try:
                volume = prepare_canonical_volume(filesystem_path, self.extraction_spec)
            except (ExtractionError, OSError) as error:
                raise DataPipelineError(
                    f"failed to prepare manifest image {record.path}: {error}"
                ) from error

            try:
                digest_after = sha256_file(filesystem_path)
            except OSError as error:
                raise DataPipelineError(
                    f"unable to hash manifest image after loading: {record.path}"
                ) from error
            if digest_after != record.sha256:
                raise DataPipelineError(
                    f"raw file changed while loading {record.path}: "
                    f"expected {record.sha256}, got {digest_after}"
                )

            self._cache[record.path] = volume
            self._cache.move_to_end(record.path)
            while len(self._cache) > self._max_cached_volumes:
                self._cache.popitem(last=False)
            return volume

    def _registered_case(self, case: CaseRecord) -> CaseRecord:
        if not isinstance(case, CaseRecord):
            raise TypeError("case must be a CaseRecord")
        registered_case = self._cases_by_key.get(case.key)
        if registered_case is None or registered_case != case:
            raise DataPipelineError("case does not exactly match a record in the pinned manifest")
        return registered_case

    def canonical_volumes_for_case(
        self,
        case: CaseRecord,
    ) -> dict[str, CanonicalVolume]:
        """Return exactly four verified canonical MRI volumes for one manifest case."""

        registered_case = self._registered_case(case)
        files_by_modality = {record.modality: record for record in registered_case.files}
        if set(MODALITIES) - set(files_by_modality):
            raise AssertionError("validated manifest case lost a required MRI modality")
        return {
            modality: self._load_registered_volume(files_by_modality[modality])
            for modality in MODALITIES
        }

    def __call__(
        self,
        *,
        path: str,
        file_sha256: str,
        modality: str,
        center_mm: tuple[float, float, float],
        geometry: SlabGeometry,
    ) -> Tensor:
        """Extract one manifest-bound normalized patch as a CPU float32 tensor."""

        self._validate_geometry(geometry)
        record = self._registered_record(
            path=path,
            file_sha256=file_sha256,
            modality=modality,
        )
        volume = self._load_registered_volume(record)
        patch = extract_patch(volume, center_mm, spec=self.extraction_spec)
        tensor = torch.from_numpy(patch.data.copy()).to(dtype=torch.float32).contiguous()
        if tuple(tensor.shape) != geometry.model_shape or not bool(torch.isfinite(tensor).all()):
            raise DataPipelineError("extraction produced an invalid model-visible patch tensor")
        return tensor


def _matching_geometry(
    extractor: CachedNiftiPatchExtractor,
    geometry: SlabGeometry | None = None,
) -> SlabGeometry:
    from simple_brats.sampling import SlabGeometry

    if not isinstance(extractor, CachedNiftiPatchExtractor):
        raise TypeError("extractor must be a CachedNiftiPatchExtractor")
    selected_geometry = (
        SlabGeometry(
            in_plane_axes=(0, 1),
            thin_axis=2,
            in_plane_footprint_mm=extractor.extraction_spec.patch_physical_extent_mm[0],
            thin_extent_mm=extractor.extraction_spec.patch_physical_extent_mm[2],
            model_shape=extractor.extraction_spec.model_visible_shape,
        )
        if geometry is None
        else geometry
    )
    return extractor._validate_geometry(selected_geometry)


def _geometry_sha256(geometry: SlabGeometry) -> str:
    from simple_brats.sampling import GeometryRecord

    return GeometryRecord.from_geometry(geometry).sha256


def _volume_digests(
    case: CaseRecord,
    volumes: dict[str, CanonicalVolume],
) -> tuple[CanonicalVolumeDigest, ...]:
    files_by_modality = {record.modality: record for record in case.files}
    return tuple(
        CanonicalVolumeDigest(
            modality=modality,
            raw_file_sha256=files_by_modality[modality].sha256,
            canonical_voxel_sha256=volumes[modality].voxel_content_sha256,
            normalized_voxel_sha256=volumes[modality].normalized_sha256,
        )
        for modality in MODALITIES
    )


def prepare_case_candidate_universe(
    extractor: CachedNiftiPatchExtractor,
    case: CaseRecord,
    *,
    geometry: SlabGeometry | None = None,
) -> PreparedCaseCandidateUniverse:
    """Prepare one immutable label-free candidate universe for a case.

    Candidate centers come only from the intersection of the four MRI
    foreground/support masks.  No segmentation record is loaded or passed to
    this boundary.  All case-invariant full-volume work is performed here so
    many independently seeded bag plans can safely reuse the result.
    """

    if not isinstance(case, CaseRecord):
        raise TypeError("case must be a CaseRecord")
    selected_geometry = _matching_geometry(extractor, geometry)
    volumes = extractor.canonical_volumes_for_case(case)
    shared_mask = intersect_modality_foreground_support_masks(
        volumes,
        spec=extractor.extraction_spec,
    )
    candidate_centers = CanonicalCandidateCenters(
        valid_patch_centers_mm(extractor.extraction_spec, shared_mask)
    )
    candidate_digest = _candidate_centers_sha256(candidate_centers.values)
    return PreparedCaseCandidateUniverse(
        case=case,
        data_manifest_sha256=extractor.data_manifest_sha256,
        extraction_spec_sha256=extractor.extraction_spec_sha256,
        geometry_sha256=_geometry_sha256(selected_geometry),
        candidate_centers=candidate_centers,
        candidate_count=len(candidate_centers),
        candidate_centers_sha256=candidate_digest,
        volume_digests=_volume_digests(case, volumes),
    )


def materialize_case_matching_plan_record(
    extractor: CachedNiftiPatchExtractor,
    case: CaseRecord,
    candidate_universe: PreparedCaseCandidateUniverse,
    *,
    epoch: int,
    bag_index: int,
    experiment_seed: int,
    geometry: SlabGeometry | None = None,
    prism_extent_mm: float | Sequence[float] | None = None,
    target_count: int = 32,
    candidate_pool_size: int = 512,
    max_attempts: int = 8,
) -> PreparedCasePlan:
    """Materialize one fresh bag from a strictly bound candidate universe."""

    if not isinstance(candidate_universe, PreparedCaseCandidateUniverse):
        raise TypeError("candidate_universe must be a PreparedCaseCandidateUniverse")
    registered_case = extractor._registered_case(case)
    selected_geometry = _matching_geometry(extractor, geometry)
    if candidate_universe.case != registered_case:
        raise DataPipelineError("candidate universe does not match the exact manifest case")
    if candidate_universe.data_manifest_sha256 != extractor.data_manifest_sha256:
        raise DataPipelineError("candidate universe does not match the pinned data manifest")
    if candidate_universe.extraction_spec_sha256 != extractor.extraction_spec_sha256:
        raise DataPipelineError("candidate universe does not match the extraction specification")
    if candidate_universe.geometry_sha256 != _geometry_sha256(selected_geometry):
        raise DataPipelineError("candidate universe does not match the patch geometry")

    plan = materialize_matching_plan(
        case=case,
        data_manifest_sha256=extractor.data_manifest_sha256,
        candidate_centers_mm=candidate_universe.candidate_centers,
        geometry=selected_geometry,
        extraction_spec_sha256=extractor.extraction_spec_sha256,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=experiment_seed,
        prism_extent_mm=prism_extent_mm,
        target_count=target_count,
        candidate_pool_size=candidate_pool_size,
        max_attempts=max_attempts,
    )
    return PreparedCasePlan(
        plan=plan,  # type: ignore[arg-type]
        candidate_count=candidate_universe.candidate_count,
        candidate_centers_sha256=candidate_universe.candidate_centers_sha256,
        volume_digests=candidate_universe.volume_digests,
    )


def prepare_case_matching_plan_record(
    extractor: CachedNiftiPatchExtractor,
    case: CaseRecord,
    *,
    epoch: int,
    bag_index: int,
    experiment_seed: int,
    geometry: SlabGeometry | None = None,
    prism_extent_mm: float | Sequence[float] | None = None,
    target_count: int = 32,
    candidate_pool_size: int = 512,
    max_attempts: int = 8,
) -> PreparedCasePlan:
    """Compatibility path that prepares a universe and materializes one plan."""

    selected_geometry = _matching_geometry(extractor, geometry)
    candidate_universe = prepare_case_candidate_universe(
        extractor,
        case,
        geometry=selected_geometry,
    )
    return materialize_case_matching_plan_record(
        extractor,
        case,
        candidate_universe,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=experiment_seed,
        geometry=selected_geometry,
        prism_extent_mm=prism_extent_mm,
        target_count=target_count,
        candidate_pool_size=candidate_pool_size,
        max_attempts=max_attempts,
    )


def prepare_case_matching_plan(
    extractor: CachedNiftiPatchExtractor,
    case: CaseRecord,
    *,
    epoch: int,
    bag_index: int,
    experiment_seed: int,
    geometry: SlabGeometry | None = None,
    prism_extent_mm: float | Sequence[float] | None = None,
    target_count: int = 32,
    candidate_pool_size: int = 512,
    max_attempts: int = 8,
) -> MaterializedPatchPlan:
    """Compatibility helper returning the replayable plan from its audit record."""

    return prepare_case_matching_plan_record(
        extractor,
        case,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=experiment_seed,
        geometry=geometry,
        prism_extent_mm=prism_extent_mm,
        target_count=target_count,
        candidate_pool_size=candidate_pool_size,
        max_attempts=max_attempts,
    ).plan


__all__ = [
    "CanonicalVolumeDigest",
    "CachedNiftiPatchExtractor",
    "DataPipelineError",
    "PreparedCaseCandidateUniverse",
    "PreparedCasePlan",
    "materialize_case_matching_plan_record",
    "prepare_case_candidate_universe",
    "prepare_case_matching_plan",
    "prepare_case_matching_plan_record",
]
