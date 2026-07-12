from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_brats.config import PatchConfig
from simple_brats.data.extraction import ExtractionSpec, valid_patch_centers_mm
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord, sha256_file
from simple_brats.data.splits import SplitFraction, SplitManifest, SubjectAssignment
from simple_brats.evaluation.patches import (
    BinaryPatchLabelRule,
    EvaluationPatchManifest,
    EvaluationPatchRecord,
    PatchEvaluationError,
    SegmentationAuditRecord,
    label_candidate_centers,
    load_evaluation_patch_manifest,
    save_evaluation_patch_manifest,
    select_balanced_patch_records,
    verify_segmentation_label_audit,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(case_id: str) -> CaseRecord:
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=(FileRecord("t1n", f"{case_id}/t1n.nii.gz", _digest(case_id)),),
    )


def _record(case: CaseRecord, partition: str, label: int, offset: float):
    return EvaluationPatchRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        subject_id=case.subject_id,
        partition=partition,
        center_mm=(5.5 + offset, 5.5, 5.5),
        seg_positive_voxels=16 if label else 0,
        crop_voxels=64,
        halo_clear=not bool(label),
        label=label,
    )


def test_ternary_rule_excludes_boundary_and_near_tumor_negatives() -> None:
    spec = ExtractionSpec(
        canonical_shape=(20, 20, 20),
        canonical_affine=IDENTITY,
        patch_source_shape=(4, 4, 4),
        patch_physical_extent_mm=(4.0, 4.0, 4.0),
        model_visible_shape=(16, 16, 16),
    )
    centers = valid_patch_centers_mm(spec, np.ones(spec.canonical_shape, dtype=np.bool_))
    segmentation = np.zeros(spec.canonical_shape, dtype=np.bool_)
    segmentation[8:12, 8:12, 8:12] = True
    rule = BinaryPatchLabelRule(positive_minimum_fraction=0.25, negative_halo_mm=4)

    counts, clear = label_candidate_centers(
        spec=spec,
        centers_mm=centers,
        segmentation_mask=segmentation,
        rule=rule,
    )

    assert int(counts.max()) == 64
    assert bool(clear[counts > 0].any()) is False
    assert bool((clear & (counts == 0)).any())
    assert bool(((counts > 0) & (counts < 16)).any())  # deliberately ambiguous boundary


def test_balanced_selection_is_deterministic_and_rule_bound() -> None:
    case = _case("BraTS-MET-00001-000")
    centers = np.asarray([(float(index), 0.0, 0.0) for index in range(20)])
    counts = np.asarray([0] * 8 + [1] * 4 + [16] * 8)
    clear = np.asarray([True] * 8 + [False] * 12)
    rule = BinaryPatchLabelRule()

    first = select_balanced_patch_records(
        case=case,
        partition="probe_train",
        centers_mm=centers,
        crop_counts=counts,
        halo_clear=clear,
        rule=rule,
        seed=7,
        maximum_per_class=4,
        minimum_per_class=2,
        crop_voxels=64,
    )
    second = select_balanced_patch_records(
        case=case,
        partition="probe_train",
        centers_mm=centers,
        crop_counts=counts,
        halo_clear=clear,
        rule=rule,
        seed=7,
        maximum_per_class=4,
        minimum_per_class=2,
        crop_voxels=64,
    )

    assert first == second
    assert sum(record.label == 0 for record in first) == 4
    assert sum(record.label == 1 for record in first) == 4
    assert {record.seg_positive_voxels for record in first if record.label == 1} == {16}
    assert all(record.halo_clear for record in first if record.label == 0)


def test_manifest_round_trip_pins_audit_and_never_contains_test_records(tmp_path: Path) -> None:
    probe_case = _case("BraTS-MET-00001-000")
    validation_case = _case("BraTS-MET-00002-000")
    records = (
        _record(probe_case, "probe_train", 0, 0.0),
        _record(probe_case, "probe_train", 1, 1.0),
        _record(validation_case, "validation", 0, 0.0),
        _record(validation_case, "validation", 1, 1.0),
    )
    manifest = EvaluationPatchManifest(
        data_manifest_sha256=_digest("manifest"),
        subject_split_sha256=_digest("split"),
        case_grid_manifest_sha256=_digest("grids"),
        segmentation_label_audit_sha256=(
            "5dc6ead2008d6b8763a050af7de6e27deb77e2540a851e2cb1d6b7afb2977222"
        ),
        patch_config=PatchConfig(),
        seed=0,
        label_rule=BinaryPatchLabelRule(),
        probe_train_subjects=(probe_case.subject_id,),
        validation_subjects=(validation_case.subject_id,),
        ineligible_probe_train_subjects=(),
        locked_test_subject_count=73,
        segmentation_audit=(
            SegmentationAuditRecord(probe_case.case_id, _digest("seg1"), (1, 2, 3, 4, 6, 8)),
            SegmentationAuditRecord(validation_case.case_id, _digest("seg2"), (1, 2, 3, 4)),
        ),
        records=records,
    )
    path = tmp_path / "patches.json"
    save_evaluation_patch_manifest(manifest, path)
    loaded = load_evaluation_patch_manifest(path, expected_sha256=manifest.sha256)

    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.segmentation_label_audit_sha256.startswith("5dc6")
    assert loaded.to_dict()["locked_test_image_or_label_access"] is False
    assert {record.partition for record in loaded.records} == {"probe_train", "validation"}
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_evaluation_patch_manifest(manifest, path)
    assert path.read_bytes() == original_bytes


def test_manifest_rejects_ambiguous_patch_record() -> None:
    case = _case("BraTS-MET-00001-000")
    with pytest.raises(PatchEvaluationError, match="negative evaluation patch"):
        EvaluationPatchRecord.create(
            source=case.source,
            release=case.release,
            case_id=case.case_id,
            subject_id=case.subject_id,
            partition="probe_train",
            center_mm=(0.0, 0.0, 0.0),
            seg_positive_voxels=1,
            crop_voxels=64,
            halo_clear=False,
            label=0,
        )


def test_actual_label_audit_file_is_hashed_and_provenance_checked(tmp_path: Path) -> None:
    train = _case("BraTS-MET-00001-000")
    validation = _case("BraTS-MET-00002-000")
    test = _case("BraTS-MET-00003-000")
    manifest = DatasetManifest(cases=(train, validation, test))
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=0,
        fractions=(
            SplitFraction("train", "0.34"),
            SplitFraction("validation", "0.33"),
            SplitFraction("test", "0.33"),
        ),
        assignments=(
            SubjectAssignment(train.subject_id, "train"),
            SubjectAssignment(validation.subject_id, "validation"),
            SubjectAssignment(test.subject_id, "test"),
        ),
    )
    value = {
        "schema": "simple-brats.segmentation-label-audit",
        "schema_version": 1,
        "provenance": {
            "manifest_sha256": manifest.sha256,
            "split_sha256": split.sha256,
            "launch_sha": "a" * 40,
            "state_sha256": _digest("state"),
        },
        "results": {"numeric_label_values": [0, 1, 2, 3, 4, 6, 8]},
        "label_semantics": {"semantic_names_assigned": False},
    }
    path = tmp_path / "label-audit.json"
    path.write_text(json.dumps(value))
    audit = verify_segmentation_label_audit(
        path,
        expected_sha256=sha256_file(path),
        manifest=manifest,
        split=split,
    )

    assert audit.numeric_label_values == (0, 1, 2, 3, 4, 6, 8)
    assert audit.launch_sha == "a" * 40

    value["label_semantics"]["semantic_names_assigned"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(PatchEvaluationError, match="semantically unnamed"):
        verify_segmentation_label_audit(
            path,
            expected_sha256=sha256_file(path),
            manifest=manifest,
            split=split,
        )
