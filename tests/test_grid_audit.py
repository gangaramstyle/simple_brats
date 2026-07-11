import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simple_brats.data.grid_audit import GridAuditError, infer_common_extraction_spec
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord

MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(root: Path, case_id: str, affine: np.ndarray) -> CaseRecord:
    directory = root / case_id
    directory.mkdir()
    records = []
    for modality in MODALITIES:
        path = directory / f"{case_id}-{modality}.nii.gz"
        image = nib.Nifti1Image(np.ones((8, 9, 5), dtype=np.float32), affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        records.append(FileRecord(modality, path.relative_to(root).as_posix(), _digest(path)))
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=records,
    )


def test_common_axis_aligned_grid_becomes_hashed_extraction_spec(tmp_path) -> None:
    affine = np.eye(4)
    manifest = DatasetManifest(
        cases=(
            _case(tmp_path, "BraTS-MET-00001-000", affine),
            _case(tmp_path, "BraTS-MET-00002-000", affine),
        )
    )
    spec = infer_common_extraction_spec(manifest, tmp_path)
    assert spec.canonical_shape == (8, 9, 5)
    assert spec.canonical_affine == tuple(tuple(float(v) for v in row) for row in affine)
    assert len(spec.sha256) == 64


def test_grid_drift_fails_instead_of_silently_selecting_reference(tmp_path) -> None:
    first = _case(tmp_path, "BraTS-MET-00001-000", np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    second = _case(tmp_path, "BraTS-MET-00002-000", shifted)
    with pytest.raises(GridAuditError, match="does not share one exact canonical grid"):
        infer_common_extraction_spec(DatasetManifest(cases=(first, second)), tmp_path)


def test_grid_audit_rehashes_image_before_trusting_header(tmp_path) -> None:
    case = _case(tmp_path, "BraTS-MET-00001-000", np.eye(4))
    manifest = DatasetManifest(cases=(case,))
    path = tmp_path / case.files[0].path
    with path.open("ab") as handle:
        handle.write(b"post-manifest mutation")

    with pytest.raises(GridAuditError, match="image bytes changed"):
        infer_common_extraction_spec(manifest, tmp_path)


def test_grid_audit_requires_explicit_millimetre_units(tmp_path) -> None:
    case = _case(tmp_path, "BraTS-MET-00001-000", np.eye(4))
    first = case.files[0]
    path = tmp_path / first.path
    image = nib.load(path)
    image.header.set_xyzt_units("unknown")
    nib.save(image, path)
    replacement = FileRecord(first.modality, first.path, _digest(path))
    updated = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=(replacement, *case.files[1:]),
    )

    with pytest.raises(GridAuditError, match="spatial units must be explicitly"):
        infer_common_extraction_spec(DatasetManifest(cases=(updated,)), tmp_path)
