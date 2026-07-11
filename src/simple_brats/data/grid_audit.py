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
    """Read image headers and require one exact global RAS+ 1 mm grid.

    This legacy helper is suitable only for a genuinely uniform release. A
    heterogeneous release must use a case-grid manifest; this function never
    erases origins or silently chooses one patient's grid for another.
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
    reference_affine: np.ndarray | None = None
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
            if reference_shape is None:
                reference_shape = shape  # type: ignore[assignment]
                reference_affine = affine
                reference_description = description
            elif shape != reference_shape or not np.array_equal(affine, reference_affine):
                raise GridAuditError(
                    "manifest does not share one exact global grid; use a case-grid manifest: "
                    f"{description} has shape={shape}, affine={affine.tolist()} but "
                    f"{reference_description} has shape={reference_shape}, "
                    f"affine={reference_affine.tolist()}"
                )
            audited += 1

    if audited == 0 or reference_shape is None or reference_affine is None:
        raise GridAuditError("manifest contains no auditable v0 images")
    try:
        return ExtractionSpec(
            canonical_shape=reference_shape,
            canonical_affine=tuple(
                tuple(float(value) for value in row) for row in reference_affine
            ),
        )
    except ExtractionError as error:
        raise GridAuditError(f"common image grid is not admissible for v0: {error}") from error


__all__ = ["GridAuditError", "infer_common_extraction_spec"]
