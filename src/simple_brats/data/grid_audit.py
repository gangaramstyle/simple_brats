"""Header-only audit that locks one common canonical grid for v0 extraction."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

from simple_brats.config import MODALITIES

from .extraction import ExtractionError, ExtractionSpec
from .manifest import DatasetManifest, sha256_file


class GridAuditError(ValueError):
    """Manifest images do not share one admissible canonical physical grid."""


def _safe_manifest_path(data_root: Path, recorded_path: str) -> Path:
    candidate = data_root / recorded_path
    if candidate.is_symlink():
        raise GridAuditError(f"manifest image path must not be a symlink: {recorded_path}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(data_root)
    except (OSError, ValueError) as error:
        raise GridAuditError(f"manifest image escapes or is missing: {recorded_path}") from error
    if not resolved.is_file():
        raise GridAuditError(f"manifest image is not a regular file: {recorded_path}")
    return resolved


def infer_common_extraction_spec(
    manifest: DatasetManifest,
    data_root: str | os.PathLike[str],
    *,
    modalities: Iterable[str] = MODALITIES,
) -> ExtractionSpec:
    """Read image headers and lock one case-local RAS+ 1 mm grid.

    All modalities within a case must share one exact affine after lossless
    RAS reorientation. Across patients, scanner-world origin translations are
    treated as an irrelevant coordinate gauge and rebased to zero; shape and
    the complete affine linear transform must still be identical. This keeps
    physical extents and registration exact without pretending that patient
    scanner origins form a shared anatomical coordinate system.
    """

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    requested_modalities = tuple(modalities)
    if requested_modalities != MODALITIES:
        raise GridAuditError(f"v0 grid audit requires canonical modalities {MODALITIES}")
    root = Path(data_root).expanduser()
    if root.is_symlink():
        raise GridAuditError("data_root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise GridAuditError(f"data_root is unavailable: {data_root}") from error
    if not root.is_dir():
        raise GridAuditError("data_root must be a directory")

    reference_shape: tuple[int, int, int] | None = None
    reference_linear: np.ndarray | None = None
    reference_description: str | None = None
    audited = 0
    for case in manifest.cases:
        files = {record.modality: record for record in case.files}
        missing = [modality for modality in MODALITIES if modality not in files]
        if missing:
            raise GridAuditError(f"case {case.case_id} is missing image modalities {missing}")
        case_affine: np.ndarray | None = None
        case_reference: str | None = None
        for modality in MODALITIES:
            record = files[modality]
            path = _safe_manifest_path(root, record.path)
            actual_sha256 = sha256_file(path)
            if actual_sha256 != record.sha256:
                raise GridAuditError(
                    f"manifest image bytes changed for {case.case_id}/{modality}: "
                    f"expected {record.sha256}, got {actual_sha256}"
                )
            try:
                image = nib.load(os.fspath(path), mmap="r")
                if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
                    raise GridAuditError(f"{case.case_id}/{modality} is not one non-empty 3D image")
                spatial_unit, _ = image.header.get_xyzt_units()
                if spatial_unit != "mm":
                    raise GridAuditError(
                        f"{case.case_id}/{modality} spatial units must be explicitly "
                        f"millimetres, got {spatial_unit!r}"
                    )
                canonical = nib.as_closest_canonical(image, enforce_diag=False)
                shape = tuple(int(size) for size in canonical.shape)
                affine = np.asarray(canonical.affine, dtype=np.float64)
            except GridAuditError:
                raise
            except Exception as error:
                raise GridAuditError(
                    f"could not audit header for {case.case_id}/{modality}: {error}"
                ) from error
            if affine.shape != (4, 4) or not np.isfinite(affine).all():
                raise GridAuditError(f"{case.case_id}/{modality} has an invalid affine")
            description = f"{case.case_id}/{modality}"
            if case_affine is None:
                case_affine = affine
                case_reference = description
            elif not np.array_equal(affine, case_affine):
                raise GridAuditError(
                    "modalities within one case do not share an exact RAS grid: "
                    f"{description} has affine={affine.tolist()} but "
                    f"{case_reference} has affine={case_affine.tolist()}"
                )
            linear = affine[:3, :3]
            if reference_shape is None:
                reference_shape = shape  # type: ignore[assignment]
                reference_linear = linear
                reference_description = description
            elif shape != reference_shape or not np.array_equal(linear, reference_linear):
                raise GridAuditError(
                    "manifest does not share one exact case-local grid shape and linear transform: "
                    f"{description} has shape={shape}, linear={linear.tolist()} but "
                    f"{reference_description} has shape={reference_shape}, "
                    f"linear={reference_linear.tolist()}"
                )
            audited += 1

    if audited == 0 or reference_shape is None or reference_linear is None:
        raise GridAuditError("manifest contains no auditable v0 images")
    canonical_affine = np.eye(4, dtype=np.float64)
    canonical_affine[:3, :3] = reference_linear
    try:
        return ExtractionSpec(
            canonical_shape=reference_shape,
            canonical_affine=tuple(
                tuple(float(value) for value in row) for row in canonical_affine
            ),
        )
    except ExtractionError as error:
        raise GridAuditError(f"common image grid is not admissible for v0: {error}") from error


__all__ = ["GridAuditError", "infer_common_extraction_spec"]
