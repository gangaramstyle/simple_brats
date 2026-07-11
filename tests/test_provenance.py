import hashlib

import pytest

from simple_brats.config import ExperimentConfig
from simple_brats.data import (
    CaseRecord,
    DatasetManifest,
    FileRecord,
    create_subject_split,
    save_manifest,
    save_split,
)
from simple_brats.provenance import collect_locked_provenance, collect_provenance


def test_provenance_rejects_fake_real_manifest_digests(tmp_path) -> None:
    (tmp_path / "uv.lock").write_text("locked")
    with pytest.raises(ValueError, match="canonical SHA-256"):
        collect_provenance(
            ExperimentConfig(),
            execution={"positions": 8},
            data_manifest_sha256="not-a-sha",
            split_manifest_sha256=hashlib.sha256(b"split").hexdigest(),
            root=tmp_path,
        )


def test_synthetic_provenance_requires_explicit_identity(monkeypatch, tmp_path) -> None:
    (tmp_path / "uv.lock").write_text("locked")
    monkeypatch.setattr(
        "simple_brats.provenance.current_git_sha",
        lambda root: "a" * 40,
    )
    provenance = collect_provenance(
        ExperimentConfig(),
        execution={"positions": 8, "tiny_model": False},
        synthetic_dataset_id="synthetic-test-v0",
        root=tmp_path,
    )
    assert provenance.data_manifest_sha256 is None
    assert provenance.synthetic_dataset_id == "synthetic-test-v0"
    assert len(provenance.execution_sha256) == 64


def test_real_provenance_loads_and_pins_manifest_files(monkeypatch, tmp_path) -> None:
    cases = tuple(
        CaseRecord.create(
            source="BraTS-MET",
            release="r1",
            case_id=f"BraTS-MET-{index:05d}-000",
            files=(
                FileRecord(
                    "t1n",
                    f"case-{index}.nii.gz",
                    hashlib.sha256(f"case-{index}".encode()).hexdigest(),
                ),
            ),
        )
        for index in range(30)
    )
    manifest = DatasetManifest(cases=cases)
    split = create_subject_split(manifest, seed=0)
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    save_manifest(manifest, manifest_path)
    save_split(split, split_path)
    (tmp_path / "uv.lock").write_text("locked")
    monkeypatch.setattr(
        "simple_brats.provenance.current_git_sha",
        lambda root: "b" * 40,
    )

    provenance = collect_locked_provenance(
        ExperimentConfig(),
        execution={"command": "train"},
        data_manifest_path=manifest_path,
        split_manifest_path=split_path,
        expected_data_manifest_sha256=manifest.sha256,
        expected_split_manifest_sha256=split.sha256,
        root=tmp_path,
    )
    assert provenance.data_manifest_sha256 == manifest.sha256
    assert provenance.split_manifest_sha256 == split.sha256
    assert provenance.synthetic_dataset_id is None
