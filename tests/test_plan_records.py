from __future__ import annotations

import hashlib
import json
from itertools import product

import pytest

from simple_brats.sampling import (
    CandidatePosition,
    MaterializedPatchPlan,
    PatchPlanError,
    SingleModalityOrderingBatchPlan,
    SlabGeometry,
    canonical_json_bytes,
    canonical_sha256,
    load_patch_plan,
    plan_single_modality_ordering_batch,
    save_patch_plan,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _batch_plan() -> SingleModalityOrderingBatchPlan:
    candidates = tuple(
        CandidatePosition(
            position_id=index,
            center_mm=(float(x), float(y), float(z)),
        )
        for index, (x, y, z) in enumerate(product((-12, -7, -2, 3, 8, 13), repeat=3))
    )
    return plan_single_modality_ordering_batch(
        candidates,
        prism_anchor_mm=(0.0, 0.0, 0.0),
        prism_extent_mm=32.0,
        target_modality_id=3,
        geometry=SlabGeometry.cubic(4.0),
        rng=317,
    )


def _record(
    batch_plan: SingleModalityOrderingBatchPlan | None = None,
) -> MaterializedPatchPlan:
    return MaterializedPatchPlan.from_ordering_batch_plan(
        _batch_plan() if batch_plan is None else batch_plan,
        data_manifest_sha256=_digest("manifest"),
        source="BraTS-MET",
        release="BraTS2026",
        case_id="BraTS-MET-00730-001",
        subject_id="BraTS-MET-00730",
        visit_id="001",
        epoch=3,
        bag_index=29,
        seed=1234567,
        extraction_spec_sha256=canonical_sha256(
            {
                "interpolation": "trilinear",
                "normalization": "patch-only-v0",
                "align_corners": False,
            }
        ),
    )


def _rehash(record_dict: dict[str, object]) -> None:
    payload = {key: value for key, value in record_dict.items() if key != "payload_sha256"}
    record_dict["payload_sha256"] = canonical_sha256(payload)


def test_batch_plan_materializes_all_provenance_and_explicit_patch_roles() -> None:
    record = _record()

    assert record.data_manifest_sha256 == _digest("manifest")
    assert (record.case.case_id, record.case.subject_id, record.case.visit_id) == (
        "BraTS-MET-00730-001",
        "BraTS-MET-00730",
        "001",
    )
    assert (record.epoch, record.bag_index, record.seed) == (3, 29, 1234567)
    assert record.geometry.model_shape == (16, 16, 16)
    assert record.geometry.in_plane_footprint_mm == 4.0
    assert record.geometry.thin_extent_mm == 4.0
    assert record.geometry_sha256 == record.geometry.sha256

    assert record.queries == record.targets
    assert len(record.targets) == 32
    assert len(record.sources) == 96
    assert record.target_modality_id == 3
    source_keys = {source.key for source in record.sources}
    assert not source_keys.intersection(target.key for target in record.targets)
    assert sum(source.modality_id == record.target_modality_id for source in record.sources) == 6

    serialized = record.to_dict()
    assert serialized["sources"]
    assert serialized["queries"]
    assert serialized["targets"]
    assert serialized["payload_sha256"] == record.sha256


def test_record_hash_is_independent_of_sampler_tuple_order() -> None:
    batch_plan = _batch_plan()
    reversed_plan = SingleModalityOrderingBatchPlan(
        prism_anchor_mm=batch_plan.prism_anchor_mm,
        prism_extent_mm=batch_plan.prism_extent_mm,
        target_modality_id=batch_plan.target_modality_id,
        sources=tuple(reversed(batch_plan.sources)),
        targets=tuple(reversed(batch_plan.targets)),
        geometry=batch_plan.geometry,
        modality_names=batch_plan.modality_names,
    )

    first = _record(batch_plan)
    second = _record(reversed_plan)

    assert first.to_json() == second.to_json()
    assert first.sha256 == second.sha256


def test_strict_save_and_load_round_trip_with_pinned_hash(tmp_path) -> None:
    record = _record()
    path = tmp_path / "epoch-0003-bag-000029.patch-plan.json"

    saved_sha = save_patch_plan(record, path)
    loaded = load_patch_plan(path, expected_sha256=saved_sha)

    assert saved_sha == record.sha256
    assert loaded == record
    assert path.read_bytes() == canonical_json_bytes(record.to_dict())
    with pytest.raises(FileExistsError):
        save_patch_plan(record, path)
    assert list(tmp_path.glob(f".{path.name}.tmp-*")) == []


def test_loader_rejects_noncanonical_json_even_when_semantics_and_hash_are_valid() -> None:
    record = _record()
    pretty = json.dumps(record.to_dict(), indent=2, sort_keys=True)

    with pytest.raises(PatchPlanError, match="canonical byte form"):
        MaterializedPatchPlan.from_json(pretty)


def test_loader_rejects_payload_tampering_before_replay() -> None:
    record_dict = _record().to_dict()
    record_dict["epoch"] = 4
    tampered = canonical_json_bytes(record_dict)

    with pytest.raises(PatchPlanError, match="payload SHA mismatch"):
        MaterializedPatchPlan.from_json(tampered)


def test_rehashed_record_still_rejects_hidden_target_source_leak() -> None:
    record_dict = _record().to_dict()
    sources = record_dict["sources"]
    targets = record_dict["targets"]
    assert isinstance(sources, list) and isinstance(targets, list)
    target_modality_id = record_dict["target_modality_id"]
    source_index = next(
        index
        for index, source in enumerate(sources)
        if isinstance(source, dict) and source["modality_id"] == target_modality_id
    )
    sources[source_index] = targets[0]
    _rehash(record_dict)

    with pytest.raises(PatchPlanError, match="hidden target identities"):
        MaterializedPatchPlan.from_json(canonical_json_bytes(record_dict))


def test_rehashed_record_rejects_wrong_source_count() -> None:
    record_dict = _record().to_dict()
    sources = record_dict["sources"]
    assert isinstance(sources, list)
    sources.pop(0)
    _rehash(record_dict)

    with pytest.raises(PatchPlanError, match="exactly 96 sources"):
        MaterializedPatchPlan.from_json(canonical_json_bytes(record_dict))


def test_rehashed_record_still_rejects_query_target_identity_drift() -> None:
    record_dict = _record().to_dict()
    queries = record_dict["queries"]
    assert isinstance(queries, list) and isinstance(queries[0], dict)
    queries[0]["modality"] = "t1c" if queries[0]["modality"] != "t1c" else "t2w"
    _rehash(record_dict)

    with pytest.raises(PatchPlanError, match="modality mapping mismatch|queries and targets"):
        MaterializedPatchPlan.from_json(canonical_json_bytes(record_dict))


def test_geometry_digest_and_pinned_plan_digest_fail_closed(tmp_path) -> None:
    record_dict = _record().to_dict()
    geometry = record_dict["geometry"]
    assert isinstance(geometry, dict)
    geometry["in_plane_footprint_mm"] = 8.0
    _rehash(record_dict)
    with pytest.raises(PatchPlanError, match="geometry SHA mismatch"):
        MaterializedPatchPlan.from_json(canonical_json_bytes(record_dict))

    record = _record()
    path = tmp_path / "plan.json"
    save_patch_plan(record, path)
    with pytest.raises(PatchPlanError, match="patch-plan SHA mismatch"):
        load_patch_plan(path, expected_sha256=_digest("different plan"))


def test_strict_schema_and_case_identity_validation() -> None:
    record_dict = _record().to_dict()
    record_dict["objective_arm"] = "matching"
    with pytest.raises(PatchPlanError, match="unexpected"):
        MaterializedPatchPlan.from_json(canonical_json_bytes(record_dict))

    record_dict = _record().to_dict()
    case = record_dict["case"]
    assert isinstance(case, dict)
    case["subject_id"] = "BraTS-MET-WRONG"
    _rehash(record_dict)
    with pytest.raises(PatchPlanError, match="case identity"):
        MaterializedPatchPlan.from_json(canonical_json_bytes(record_dict))
