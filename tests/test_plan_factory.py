import hashlib

import pytest

from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.data.plan_factory import (
    PlanFactoryError,
    materialize_matching_plan,
    stateless_plan_seed,
)
from simple_brats.sampling import V0_SLAB_GEOMETRY


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case() -> CaseRecord:
    case_id = "BraTS-MET-00001-000"
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=tuple(
            FileRecord(modality, f"{case_id}-{modality}.nii.gz", _digest(modality))
            for modality in ("t1n", "t1c", "t2w", "t2f")
        ),
    )


def _centers() -> list[tuple[float, float, float]]:
    return [(float(6 * (index % 10)), float(6 * (index // 10)), 0.0) for index in range(100)]


def test_stateless_plan_is_reproducible_and_bag_specific() -> None:
    arguments = {
        "case": _case(),
        "data_manifest_sha256": _digest("manifest"),
        "candidate_centers_mm": list(reversed(_centers())),
        "geometry": V0_SLAB_GEOMETRY,
        "extraction_spec_sha256": _digest("extraction"),
        "epoch": 2,
        "bag_index": 7,
        "experiment_seed": 11,
        "target_count": 32,
        "candidate_pool_size": 64,
    }
    first = materialize_matching_plan(**arguments)
    second = materialize_matching_plan(**{**arguments, "candidate_centers_mm": _centers()})
    different_bag = materialize_matching_plan(**{**arguments, "bag_index": 8})

    assert first.sha256 == second.sha256
    assert first.sha256 != different_bag.sha256
    assert len(first.targets) == 32
    assert len(first.sources) == 96


def test_plan_seed_binds_case_and_manifest() -> None:
    case = _case()
    first = stateless_plan_seed(
        data_manifest_sha256=_digest("manifest"),
        case=case,
        epoch=0,
        bag_index=0,
        experiment_seed=0,
    )
    second = stateless_plan_seed(
        data_manifest_sha256=_digest("other"),
        case=case,
        epoch=0,
        bag_index=0,
        experiment_seed=0,
    )
    assert 0 <= first < 2**64
    assert first != second


def test_factory_fails_without_relaxing_safe_center_count() -> None:
    with pytest.raises(PlanFactoryError, match="safe centers"):
        materialize_matching_plan(
            case=_case(),
            data_manifest_sha256=_digest("manifest"),
            candidate_centers_mm=_centers()[:7],
            geometry=V0_SLAB_GEOMETRY,
            extraction_spec_sha256=_digest("extraction"),
            epoch=0,
            bag_index=0,
            experiment_seed=0,
            target_count=8,
        )
