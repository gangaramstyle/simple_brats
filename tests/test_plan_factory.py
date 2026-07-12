import hashlib

import numpy as np
import pytest

from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.data.plan_factory import (
    CanonicalCandidateCenters,
    PlanFactoryError,
    materialize_matching_plan,
    stateless_plan_seed,
)
from simple_brats.sampling import V0_CUBIC_GEOMETRY, V0_SLAB_GEOMETRY


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
    different_epoch = materialize_matching_plan(**{**arguments, "epoch": 3})

    assert first.sha256 == second.sha256
    assert first.sha256 != different_bag.sha256
    assert first.sha256 != different_epoch.sha256
    assert len(first.targets) == 32
    assert len(first.sources) == 96


def test_canonical_candidate_centers_are_immutable_and_plan_equivalent() -> None:
    raw = [*_centers(), _centers()[0], _centers()[1]]
    canonical = CanonicalCandidateCenters(np.asarray(list(reversed(raw))))
    arguments = {
        "case": _case(),
        "data_manifest_sha256": _digest("manifest"),
        "geometry": V0_SLAB_GEOMETRY,
        "extraction_spec_sha256": _digest("extraction"),
        "epoch": 2,
        "bag_index": 7,
        "experiment_seed": 11,
        "target_count": 32,
        "candidate_pool_size": 64,
    }

    generic_plan = materialize_matching_plan(candidate_centers_mm=raw, **arguments)
    cached_plan = materialize_matching_plan(candidate_centers_mm=canonical, **arguments)

    assert cached_plan.sha256 == generic_plan.sha256
    assert len(canonical) == len(_centers())
    assert canonical.values.dtype == np.dtype("<f8")
    assert canonical.values.flags.c_contiguous
    assert not canonical.values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        canonical.values[0, 0] = 999.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        canonical.values.setflags(write=True)


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


def test_registered_cubic_plan_preserves_pre_vectorization_golden_sha() -> None:
    plan = materialize_matching_plan(
        case=_case(),
        data_manifest_sha256=_digest("manifest"),
        candidate_centers_mm=list(reversed(_centers())),
        geometry=V0_CUBIC_GEOMETRY,
        extraction_spec_sha256=_digest("extraction"),
        epoch=2,
        bag_index=7,
        experiment_seed=11,
        target_count=32,
        candidate_pool_size=64,
    )

    assert plan.sha256 == (
        "74577adadc55d737658ca5a138517b82da6e70207d54d934de6458fbbc7fdd60"
    )
