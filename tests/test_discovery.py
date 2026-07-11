from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simple_brats.data import DiscoveryError, discover_met_release

REQUIRED = ("t1n", "t1c", "t2w", "t2f")


def _write_case(parent: Path, case_id: str, *, modalities: tuple[str, ...] = REQUIRED) -> Path:
    case_directory = parent / case_id
    case_directory.mkdir(parents=True)
    for modality in modalities:
        (case_directory / f"{case_id}-{modality}.nii.gz").write_bytes(
            f"{case_id}:{modality}".encode()
        )
    return case_directory


def test_discovers_nested_and_direct_cases_with_stable_relative_paths(tmp_path: Path) -> None:
    direct_id = "BraTS-MET-00002-001"
    nested_id = "BraTS-MET-00001-002"
    _write_case(tmp_path, direct_id)
    nested = _write_case(tmp_path / "opaque-release-name" / "train", nested_id)
    seg = nested / f"{nested_id}-seg.nii.gz"
    seg.write_bytes(b"segmentation")

    manifest = discover_met_release(tmp_path, source="BraTS-MET", release="2026-training")

    assert [case.case_id for case in manifest.cases] == [nested_id, direct_id]
    assert {case.source for case in manifest.cases} == {"BraTS-MET"}
    assert {case.release for case in manifest.cases} == {"2026-training"}
    nested_case = next(case for case in manifest.cases if case.case_id == nested_id)
    assert set(nested_case.modalities) == {*REQUIRED, "seg"}
    assert {record.path for record in nested_case.files} == {
        f"opaque-release-name/train/{nested_id}/{nested_id}-{modality}.nii.gz"
        for modality in (*REQUIRED, "seg")
    }
    t1n = next(record for record in nested_case.files if record.modality == "t1n")
    assert t1n.sha256 == hashlib.sha256(f"{nested_id}:t1n".encode()).hexdigest()


def test_missing_required_modality_fails_closed(tmp_path: Path) -> None:
    _write_case(tmp_path, "BraTS-MET-00001-001", modalities=("t1n", "t1c", "t2w"))

    with pytest.raises(DiscoveryError, match="missing required modalities.*t2f"):
        discover_met_release(tmp_path, source="BraTS-MET", release="r1")


def test_unknown_case_nifti_fails_closed(tmp_path: Path) -> None:
    case_id = "BraTS-MET-00001-001"
    case_directory = _write_case(tmp_path, case_id)
    (case_directory / f"{case_id}-adc.nii.gz").write_bytes(b"unexpected")

    with pytest.raises(DiscoveryError, match="unknown NIfTI.*adc"):
        discover_met_release(tmp_path, source="BraTS-MET", release="r1")


def test_duplicate_case_directories_under_nested_releases_are_rejected(tmp_path: Path) -> None:
    case_id = "BraTS-MET-00001-001"
    _write_case(tmp_path / "release-a", case_id)
    _write_case(tmp_path / "release-b", case_id)

    with pytest.raises(DiscoveryError, match="duplicate case directory"):
        discover_met_release(tmp_path, source="BraTS-MET", release="r1")


def test_symlinked_image_escape_is_rejected(tmp_path: Path) -> None:
    case_id = "BraTS-MET-00001-001"
    case_directory = _write_case(tmp_path / "dataset", case_id, modalities=("t1c", "t2w", "t2f"))
    outside = tmp_path / "outside-t1n.nii.gz"
    outside.write_bytes(b"outside")
    (case_directory / f"{case_id}-t1n.nii.gz").symlink_to(outside)

    with pytest.raises(DiscoveryError, match="symlinked file|symlink inside case"):
        discover_met_release(tmp_path / "dataset", source="BraTS-MET", release="r1")


def test_no_cases_is_an_explicit_discovery_error(tmp_path: Path) -> None:
    (tmp_path / "metadata.txt").write_text("empty release")

    with pytest.raises(DiscoveryError, match="no canonical BraTS MET"):
        discover_met_release(tmp_path, source="BraTS-MET", release="r1")
