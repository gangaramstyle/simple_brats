from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simple_brats.data.case_grids import (
    CaseGridError,
    ExtractionPolicy,
    SpatialGrid,
    audit_case_grids,
    derive_prepared_grid,
    load_case_grid_manifest,
    save_case_grid_manifest,
)
from simple_brats.data.extraction import prepare_canonical_volume
from simple_brats.data.manifest import (
    CaseRecord,
    DatasetManifest,
    FileRecord,
    sha256_file,
)

MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def _save_image(
    path: Path,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    units: str = "mm",
) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units(units)
    nib.save(image, path)


def _case(
    root: Path,
    case_id: str,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    include_seg: bool = False,
) -> CaseRecord:
    directory = root / case_id
    directory.mkdir()
    files = []
    for modality in MODALITIES:
        path = directory / f"{case_id}-{modality}.nii.gz"
        _save_image(path, shape, affine)
        files.append(FileRecord(modality, path.relative_to(root).as_posix(), sha256_file(path)))
    if include_seg:
        path = directory / f"{case_id}-seg.nii.gz"
        # Labels are recorded in the data manifest but never inspected by the
        # case-grid audit; intentionally leave their units unspecified.
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine), path)
        files.append(FileRecord("seg", path.relative_to(root).as_posix(), sha256_file(path)))
    return CaseRecord.create(
        source="BraTS-MET",
        release="heterogeneous-test",
        case_id=case_id,
        files=files,
    )


def test_audit_preserves_heterogeneous_case_grids_and_derives_per_case_1mm(
    tmp_path: Path,
) -> None:
    first_affine = np.eye(4)
    first_affine[:3, 3] = (-7.0, 2.0, 11.0)
    second_affine = np.diag((0.8, 0.8, 1.5, 1.0))
    second_affine[:3, 3] = (10.0, -4.0, 3.0)
    first = _case(
        tmp_path,
        "BraTS-MET-00001-000",
        (8, 9, 5),
        first_affine,
        include_seg=True,
    )
    second = _case(
        tmp_path,
        "BraTS-MET-00002-000",
        (10, 12, 4),
        second_affine,
    )
    manifest = DatasetManifest(cases=(first, second))

    catalog = audit_case_grids(manifest, tmp_path)
    catalog.validate_manifest(manifest)
    first_record = catalog.record_for_case(first)
    second_record = catalog.record_for_case(second)

    assert first_record.native_grid.shape == (8, 9, 5)
    assert np.array_equal(first_record.native_grid.affine, first_affine)
    assert first_record.prepared_grid.shape == (8, 9, 5)
    assert np.array_equal(first_record.prepared_grid.affine, first_affine)
    assert second_record.native_grid.shape == (10, 12, 4)
    assert second_record.native_grid.spacing_mm == pytest.approx((0.8, 0.8, 1.5))
    assert second_record.prepared_grid.shape == (8, 10, 6)
    assert np.asarray(second_record.prepared_grid.affine)[:3, 3] == pytest.approx(
        (10.1, -3.9, 2.75)
    )
    second_spec = catalog.extraction_spec_for_case(second)
    assert second_spec.canonical_shape == (8, 10, 6)
    assert np.diag(np.asarray(second_spec.canonical_affine))[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert second_spec.world_origin_policy == "preserve-case-physical-bounds"
    assert first_record.extraction_spec_sha256 != second_record.extraction_spec_sha256

    second_t1n = next(record for record in second.files if record.modality == "t1n")
    volume = prepare_canonical_volume(tmp_path / second_t1n.path, second_spec)
    assert volume.data.shape == second_record.prepared_grid.shape
    assert np.array_equal(volume.affine, np.asarray(second_record.prepared_grid.affine))
    assert volume.valid_support_mask.any()
    assert not volume.valid_support_mask.all()


def test_catalog_is_canonical_content_addressed_and_manifest_bound(tmp_path: Path) -> None:
    case = _case(tmp_path, "BraTS-MET-00001-000", (8, 9, 5), np.eye(4))
    manifest = DatasetManifest(cases=(case,))
    catalog = audit_case_grids(manifest, tmp_path)
    path = tmp_path / "case-grids.json"
    save_case_grid_manifest(catalog, path)

    assert path.read_bytes() == catalog.to_json().encode("utf-8")
    loaded = load_case_grid_manifest(path, expected_sha256=catalog.sha256)
    assert loaded == catalog

    changed_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=tuple(
            replace(record, sha256="0" * 64) if record.modality == "t1n" else record
            for record in case.files
        ),
    )
    with pytest.raises(CaseGridError, match="does not exactly match"):
        loaded.record_for_case(changed_case)
    with pytest.raises(CaseGridError, match="different data manifest"):
        loaded.validate_manifest(DatasetManifest(cases=(changed_case,)))


def test_audit_rejects_within_case_registration_drift(tmp_path: Path) -> None:
    case = _case(tmp_path, "BraTS-MET-00001-000", (8, 9, 5), np.eye(4))
    first = case.files[0]
    path = tmp_path / first.path
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    _save_image(path, (8, 9, 5), shifted)
    updated = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=(
            FileRecord(first.modality, first.path, sha256_file(path)),
            *case.files[1:],
        ),
    )

    with pytest.raises(CaseGridError, match="not exactly registered"):
        audit_case_grids(DatasetManifest(cases=(updated,)), tmp_path)


def test_audit_rehashes_mri_and_requires_explicit_mm_units(tmp_path: Path) -> None:
    case = _case(tmp_path, "BraTS-MET-00001-000", (8, 9, 5), np.eye(4))
    path = tmp_path / case.files[0].path
    with path.open("ab") as handle:
        handle.write(b"mutated")
    with pytest.raises(CaseGridError, match="file bytes changed"):
        audit_case_grids(DatasetManifest(cases=(case,)), tmp_path)

    other_root = tmp_path / "unknown"
    other_root.mkdir()
    unknown = _case(other_root, "BraTS-MET-00002-000", (8, 9, 5), np.eye(4))
    first = unknown.files[0]
    unknown_path = other_root / first.path
    _save_image(unknown_path, (8, 9, 5), np.eye(4), units="unknown")
    updated = CaseRecord.create(
        source=unknown.source,
        release=unknown.release,
        case_id=unknown.case_id,
        files=(
            FileRecord(first.modality, first.path, sha256_file(unknown_path)),
            *unknown.files[1:],
        ),
    )
    with pytest.raises(CaseGridError, match="spatial unit"):
        audit_case_grids(DatasetManifest(cases=(updated,)), other_root)


def test_oblique_native_bounds_are_covered_without_erasing_origin() -> None:
    angle = np.deg2rad(10.0)
    affine = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 17.0],
            [np.sin(angle), np.cos(angle), 0.0, -9.0],
            [0.0, 0.0, 1.5, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    native = SpatialGrid(
        shape=(10, 12, 4),
        affine=tuple(tuple(float(value) for value in row) for row in affine),
    )
    policy = ExtractionPolicy()
    prepared = derive_prepared_grid(native, policy)
    native_lower, native_upper = native.voxel_cell_bounds_mm
    prepared_lower, prepared_upper = prepared.voxel_cell_bounds_mm

    assert np.asarray(prepared_lower) == pytest.approx(native_lower)
    assert np.all(np.asarray(prepared_upper) + 1e-8 >= np.asarray(native_upper))
    assert prepared.affine[0][3] != 0.0


def test_integer_bound_roundoff_does_not_add_a_spurious_voxel() -> None:
    affine = np.diag((1.00000000001, 1.0, 1.0, 1.0))
    native = SpatialGrid(
        shape=(8, 9, 5),
        affine=tuple(tuple(float(value) for value in row) for row in affine),
    )
    assert derive_prepared_grid(native, ExtractionPolicy()).shape == (8, 9, 5)
