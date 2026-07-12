import hashlib

import numpy as np
import pytest

from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.data.plan_factory import (
    CanonicalCandidateCenters,
    PlanFactoryError,
    balanced_target_modality_id,
    materialize_matching_plan,
    stateless_plan_seed,
)
from simple_brats.sampling import V0_CUBIC_GEOMETRY, SlabGeometry


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


def _centers(edge_mm: float = 4.0) -> list[tuple[float, float, float]]:
    spacing = edge_mm + 1.0
    offsets = tuple(spacing * index for index in range(7))
    return [(x, y, z) for z in offsets for y in offsets for x in offsets]


def test_stateless_plan_is_reproducible_and_bag_specific() -> None:
    arguments = {
        "case": _case(),
        "data_manifest_sha256": _digest("manifest"),
        "candidate_centers_mm": list(reversed(_centers())),
        "geometry": V0_CUBIC_GEOMETRY,
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
    assert first.prism_extent_mm == (32.0, 32.0, 32.0)
    assert {target.modality_id for target in first.targets} == {first.target_modality_id}


def test_canonical_candidate_centers_are_immutable_and_plan_equivalent() -> None:
    raw = [*_centers(), _centers()[0], _centers()[1]]
    canonical = CanonicalCandidateCenters(np.asarray(list(reversed(raw))))
    arguments = {
        "case": _case(),
        "data_manifest_sha256": _digest("manifest"),
        "geometry": V0_CUBIC_GEOMETRY,
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


def test_target_modality_is_an_exact_balanced_random_four_bag_cycle() -> None:
    arguments = {
        "data_manifest_sha256": _digest("manifest"),
        "case": _case(),
        "epoch": 3,
        "experiment_seed": 19,
    }

    first_block = [
        balanced_target_modality_id(**arguments, bag_index=bag_index) for bag_index in range(4)
    ]
    second_block = [
        balanced_target_modality_id(**arguments, bag_index=bag_index) for bag_index in range(4, 8)
    ]

    assert sorted(first_block) == [0, 1, 2, 3]
    assert sorted(second_block) == [0, 1, 2, 3]
    assert first_block == [
        balanced_target_modality_id(**arguments, bag_index=bag_index) for bag_index in range(4)
    ]


def test_factory_fails_without_relaxing_safe_center_count() -> None:
    with pytest.raises(PlanFactoryError, match="safe centers"):
        materialize_matching_plan(
            case=_case(),
            data_manifest_sha256=_digest("manifest"),
            candidate_centers_mm=_centers()[:7],
            geometry=V0_CUBIC_GEOMETRY,
            extraction_spec_sha256=_digest("extraction"),
            epoch=0,
            bag_index=0,
            experiment_seed=0,
            target_count=32,
        )


@pytest.mark.parametrize(
    ("edge_mm", "prism_extent_mm"),
    ((4.0, 32.0), (8.0, 64.0)),
)
def test_factory_materializes_only_registered_local_scale_pairs(
    edge_mm: float,
    prism_extent_mm: float,
) -> None:
    geometry = SlabGeometry.cubic(edge_mm)
    plan = materialize_matching_plan(
        case=_case(),
        data_manifest_sha256=_digest("manifest"),
        candidate_centers_mm=list(reversed(_centers(edge_mm))),
        geometry=geometry,
        extraction_spec_sha256=_digest("extraction"),
        epoch=2,
        bag_index=7,
        experiment_seed=11,
        target_count=32,
        prism_extent_mm=prism_extent_mm,
        candidate_pool_size=64,
    )

    assert plan.prism_extent_mm == (prism_extent_mm,) * 3
    assert plan.geometry.to_geometry() == geometry
    assert len(plan.targets) == 32
    assert len(plan.sources) == 96


def test_factory_rejects_unregistered_prism_patch_pair() -> None:
    with pytest.raises(ValueError, match="requires a 32 mm cubic prism"):
        materialize_matching_plan(
            case=_case(),
            data_manifest_sha256=_digest("manifest"),
            candidate_centers_mm=_centers(),
            geometry=V0_CUBIC_GEOMETRY,
            extraction_spec_sha256=_digest("extraction"),
            epoch=0,
            bag_index=0,
            experiment_seed=0,
            prism_extent_mm=64.0,
            candidate_pool_size=64,
        )
