from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simple_brats.data.case_grids import audit_case_grids, save_case_grid_manifest
from simple_brats.data.manifest import (
    CaseRecord,
    DatasetManifest,
    FileRecord,
    canonical_json_bytes,
    save_manifest,
    sha256_file,
)
from simple_brats.data.splits import (
    SplitFraction,
    SplitManifest,
    SubjectAssignment,
    save_split,
)
from simple_brats.provenance import current_git_sha
from simple_brats.real_pilot import run_real_io_pilot
from simple_brats.sampling import load_patch_plan

IDENTITY_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def _write_nifti(
    path: Path,
    *,
    case_index: int,
    modality_index: int,
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> None:
    x, y, z = np.indices(shape, dtype=np.float32)
    values = (
        1.0
        + x
        + (modality_index + 1.0) * np.square(y) / 37.0
        + (case_index + 1.0) * 0.17 * x * y
        + z
    ).astype(np.float32)
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def _pilot_inputs(tmp_path: Path) -> dict[str, object]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    grids = (
        ((40, 40, 40), np.asarray(IDENTITY_AFFINE, dtype=np.float64)),
        ((20, 18, 12), np.diag((0.8, 0.8, 1.5, 1.0))),
        ((18, 20, 16), np.asarray(IDENTITY_AFFINE, dtype=np.float64)),
    )
    grids[0][1][:3, 3] = (-7.0, 2.0, 11.0)
    grids[1][1][:3, 3] = (10.0, -4.0, 3.0)
    grids[2][1][:3, 3] = (3.0, 8.0, -2.0)
    cases = []
    for case_index, (shape, affine) in enumerate(grids):
        case_id = f"BraTS-MET-{case_index + 1:05d}-000"
        case_dir = data_root / case_id
        case_dir.mkdir()
        files = []
        for modality_index, modality in enumerate(MODALITIES):
            path = case_dir / f"{case_id}-{modality}.nii.gz"
            _write_nifti(
                path,
                case_index=case_index,
                modality_index=modality_index,
                shape=shape,
                affine=affine,
            )
            files.append(
                FileRecord(
                    modality=modality,
                    path=path.relative_to(data_root).as_posix(),
                    sha256=sha256_file(path),
                )
            )
        cases.append(
            CaseRecord.create(
                source="BraTS-MET",
                release="synthetic-real-pilot",
                case_id=case_id,
                files=files,
            )
        )

    manifest = DatasetManifest(cases=tuple(cases))
    manifest_path = tmp_path / "filtered.manifest.json"
    save_manifest(manifest, manifest_path)
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=0,
        fractions=(
            SplitFraction("train", "0.34"),
            SplitFraction("validation", "0.33"),
            SplitFraction("test", "0.33"),
        ),
        assignments=(
            SubjectAssignment(cases[0].subject_id, "train"),
            SubjectAssignment(cases[1].subject_id, "validation"),
            SubjectAssignment(cases[2].subject_id, "test"),
        ),
    )
    split_path = tmp_path / "subject-split.json"
    save_split(split, split_path)
    case_grid_manifest = audit_case_grids(manifest, data_root)
    case_grid_manifest_path = tmp_path / "case-grid-manifest.json"
    save_case_grid_manifest(case_grid_manifest, case_grid_manifest_path)
    config_path = tmp_path / "tiny-real-pilot.toml"
    config_path.write_text(
        """\
seed = 11
checkpoint_every_steps = 1000
artifact_every_steps = 5000

[patch]
footprint_mm = 4.0
thin_mm = 4.0
tensor_shape = [16, 16, 16]

[model]
width = 24
depth = 2
heads = 3
mlp_ratio = 2.0
predictor_depth = 1
teacher_ema_momentum = 0.996

[task]
modalities = ["t1n", "t1c", "t2w", "t2f"]
prism_extent_mm = [32.0, 32.0, 32.0]
target_patches_per_bag = 32
context_patches_per_nontarget_modality = 30
context_patches_target_modality = 6
objective = "match"
allow_target_modality_elsewhere = true
allow_target_modality_at_target = false
pass_scan_statistics_to_teacher = false
"""
    )
    repo_root = Path(__file__).parents[1]
    return {
        "data_root": data_root,
        "manifest_path": manifest_path,
        "expected_manifest_sha256": manifest.sha256,
        "split_path": split_path,
        "expected_split_sha256": split.sha256,
        "case_grid_manifest_path": case_grid_manifest_path,
        "expected_case_grid_manifest_sha256": case_grid_manifest.sha256,
        "config_path": config_path,
        "output_dir": tmp_path / "pilot-output",
        "expected_git_sha": current_git_sha(repo_root),
        "repo_root": repo_root,
        "device": "cpu",
        "candidate_pool_size": 512,
    }


def test_real_pilot_runs_one_synthetic_nifti_step_and_persists_audit(
    tmp_path: Path,
) -> None:
    arguments = _pilot_inputs(tmp_path)

    report = run_real_io_pilot(**arguments)

    output_dir = Path(arguments["output_dir"])
    plan_path = output_dir / "materialized-patch-plan.json"
    audit_path = output_dir / "prepared-plan-audit.json"
    report_path = output_dir / "pilot-report.json"
    plan = load_patch_plan(
        plan_path,
        expected_sha256=report["hashes"]["materialized_patch_plan_sha256"],  # type: ignore[index]
    )
    audit = json.loads(audit_path.read_bytes())
    disk_report = json.loads(report_path.read_bytes())

    assert report["status"] == "ok"
    assert report["case"]["case_id"] == "BraTS-MET-00001-000"  # type: ignore[index]
    assert report["metrics"]["teacher_updates"] == 1  # type: ignore[index]
    assert report["metrics"]["gradient_norm_before_clip"] > 0  # type: ignore[index,operator]
    assert report["metrics"]["ema_update_norm"] > 0  # type: ignore[index,operator]
    assert report["metrics"]["chance"] == pytest.approx(1.0 / 32.0)  # type: ignore[index]
    assert report["shapes"]["source_patches"] == [1, 96, 16, 16, 16]  # type: ignore[index]
    assert report["shapes"]["target_patches"] == [1, 32, 16, 16, 16]  # type: ignore[index]
    assert len(report["volume_digests"]) == 4  # type: ignore[arg-type]
    assert (
        report["hashes"]["case_grid_manifest_sha256"]
        == arguments[  # type: ignore[index]
            "expected_case_grid_manifest_sha256"
        ]
    )
    assert len(report["hashes"]["extraction_policy_sha256"]) == 64  # type: ignore[index,arg-type]
    assert (
        report["hashes"]["extraction_policy_sha256"]
        == report["hashes"]["runtime_extraction_policy_sha256"]
    )
    assert (
        report["hashes"]["case_grid_policy_sha256"]
        != report["hashes"]["runtime_extraction_policy_sha256"]
    )
    assert len(report["hashes"]["case_grid_record_sha256"]) == 64  # type: ignore[index,arg-type]
    assert plan.extraction_spec_sha256 == report["hashes"]["extraction_spec_sha256"]  # type: ignore[index]
    assert report["case_grid"]["native_grid"]["affine"][0][3] == -7.0  # type: ignore[index]
    assert set(report["case_grid"]["native_grids_by_modality"]) == set(MODALITIES)  # type: ignore[index,arg-type]
    assert report["case_grid"]["prepared_grid"]["affine"][0][3] == -7.0  # type: ignore[index]
    assert plan.sha256 == report["hashes"]["materialized_patch_plan_sha256"]  # type: ignore[index]
    assert (
        audit["candidate_centers_sha256"]
        == report["hashes"][  # type: ignore[index]
            "candidate_centers_sha256"
        ]
    )
    assert disk_report == report
    assert report_path.read_bytes() == canonical_json_bytes(disk_report)
    assert plan_path.read_bytes() == plan.to_json().encode("utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_real_io_pilot(**arguments)


def test_real_pilot_rejects_wrong_artifact_pin_before_creating_output(tmp_path: Path) -> None:
    arguments = _pilot_inputs(tmp_path)
    arguments["expected_case_grid_manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA mismatch"):
        run_real_io_pilot(**arguments)

    assert not Path(arguments["output_dir"]).exists()
