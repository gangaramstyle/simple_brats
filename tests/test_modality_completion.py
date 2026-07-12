from __future__ import annotations

import hashlib
import json
from collections import Counter
from random import Random

import numpy as np
import pytest

from simple_brats.sampling import (
    ALL_MODALITY_IDS,
    V0_SLAB_GEOMETRY,
    CandidatePosition,
    ModalityCompletionPlanningError,
    PatchRole,
    plan_modality_completion_batch,
)
from simple_brats.sampling.geometry import AxisAlignedSlab, SlabGeometry
from simple_brats.sampling.modality_completion import _closed_patch_conflict_matrix


def _candidates(count: int) -> list[CandidatePosition]:
    return [
        CandidatePosition(
            position_id=100 + index,
            center_mm=(5.0 * index, 0.0, 0.0),
        )
        for index in range(count)
    ]


def test_plan_is_balanced_and_hides_exactly_one_available_modality_per_location() -> None:
    plan = plan_modality_completion_batch(_candidates(12), batch_size=8, rng=11)

    assert len(plan.locations) == 8
    assert plan.target_counts == {0: 2, 1: 2, 2: 2, 3: 2}
    for location in plan.locations:
        assert location.target.role is PatchRole.TARGET
        assert location.target_modality_id not in {
            patch.modality_id for patch in location.visible_sources
        }
        assert {patch.modality_id for patch in location.visible_sources} == (
            set(location.available_modality_ids) - {location.target_modality_id}
        )
        assert all(
            patch.role is PatchRole.VISIBLE_SOURCE and patch.position_id == location.position_id
            for patch in location.visible_sources
        )


def test_target_modality_is_visible_elsewhere_only_in_disjoint_slabs() -> None:
    plan = plan_modality_completion_batch(_candidates(8), batch_size=8, rng=3)

    for target in plan.targets:
        same_modality_context = [
            source for source in plan.visible_sources if source.modality_id == target.modality_id
        ]
        assert same_modality_context
        target_slab = V0_SLAB_GEOMETRY.slab(target.center_mm)
        assert all(
            source.position_id != target.position_id
            and not target_slab.intersects(V0_SLAB_GEOMETRY.slab(source.center_mm))
            for source in same_modality_context
        )


def test_available_modalities_are_respected_without_inventing_sources() -> None:
    candidates = [
        CandidatePosition(0, (0.0, 0.0, 0.0), (0, 1)),
        CandidatePosition(1, (5.0, 0.0, 0.0), (1, 2)),
        CandidatePosition(2, (10.0, 0.0, 0.0), (2, 3)),
        CandidatePosition(3, (15.0, 0.0, 0.0), (0, 3)),
    ]
    available_by_position = {
        candidate.position_id: set(candidate.available_modality_ids) for candidate in candidates
    }

    plan = plan_modality_completion_batch(candidates, batch_size=4, rng=0)

    assert Counter(target.modality_id for target in plan.targets) == Counter(ALL_MODALITY_IDS)
    for location in plan.locations:
        available = available_by_position[location.position_id]
        assert location.target_modality_id in available
        assert {patch.modality_id for patch in location.visible_sources} == (
            available - {location.target_modality_id}
        )


def test_patch_keys_make_metadata_permutation_safe() -> None:
    plan = plan_modality_completion_batch(_candidates(4), batch_size=4, rng=9)
    patches = [*plan.targets, *plan.visible_sources]
    Random(123).shuffle(patches)

    reconstructed = {patch.key: patch for patch in patches}
    assert reconstructed == plan.patches_by_key
    assert len(reconstructed) == 4 * 4
    assert all(
        patch.key.position_id == patch.position_id and patch.key.modality_id == patch.modality_id
        for patch in reconstructed.values()
    )


def test_planner_raises_when_overlap_prevents_a_complete_batch() -> None:
    candidates = [CandidatePosition(index, (0.25 * index, 0.0, 0.0)) for index in range(8)]

    with pytest.raises(ModalityCompletionPlanningError):
        plan_modality_completion_batch(candidates, batch_size=4, rng=0)


@pytest.mark.parametrize("batch_size", [0, 2, 5])
def test_batch_size_must_support_exact_four_modality_balance(batch_size: int) -> None:
    with pytest.raises(ValueError):
        plan_modality_completion_batch(_candidates(8), batch_size=batch_size, rng=0)


def test_duplicate_position_ids_are_rejected() -> None:
    candidates = [
        CandidatePosition(7, (0.0, 0.0, 0.0)),
        CandidatePosition(7, (5.0, 0.0, 0.0)),
        CandidatePosition(8, (10.0, 0.0, 0.0)),
        CandidatePosition(9, (15.0, 0.0, 0.0)),
    ]

    with pytest.raises(ValueError, match="position_id"):
        plan_modality_completion_batch(candidates, batch_size=4, rng=0)


@pytest.mark.parametrize(
    "geometry",
    (
        V0_SLAB_GEOMETRY,
        SlabGeometry.cubic(4.0),
        SlabGeometry(
            in_plane_axes=(1, 2),
            thin_axis=0,
            in_plane_footprint_mm=8.0,
            thin_extent_mm=4.0,
            model_shape=(16, 16, 16),
        ),
    ),
)
def test_vectorized_conflicts_exactly_match_scalar_closed_box_predicate(
    geometry: SlabGeometry,
) -> None:
    random = np.random.default_rng(71)
    coordinates = random.normal(size=(96, 3)) * 20.0
    extents = np.asarray(geometry.extents_mm, dtype=np.float64)
    coordinates[:8] = np.asarray(
        (
            (0.0, 0.0, 0.0),
            tuple(extents),
            (extents[0], 0.0, 0.0),
            (0.0, extents[1], 0.0),
            (0.0, 0.0, extents[2]),
            tuple(np.nextafter(extents, 0.0)),
            tuple(np.nextafter(extents, np.inf)),
            (-0.0, 0.0, -0.0),
        ),
        dtype=np.float64,
    )
    slabs = tuple(geometry.slab(center) for center in coordinates)
    scalar = np.asarray(
        [[first.intersects(second) for second in slabs] for first in slabs],
        dtype=np.bool_,
    )

    vectorized = _closed_patch_conflict_matrix(slabs)

    assert np.array_equal(vectorized, scalar)
    assert vectorized.dtype == np.dtype(np.bool_)
    assert vectorized.shape == (len(slabs), len(slabs))
    assert np.array_equal(vectorized, vectorized.T)
    assert bool(vectorized.diagonal().all())
    assert not vectorized.flags.writeable


def test_vectorized_conflicts_do_not_consume_or_reorder_planner_rng() -> None:
    candidates = [
        CandidatePosition(
            position_id=100 + index,
            center_mm=(
                float(6 * (index % 8)),
                float(6 * ((index // 8) % 8)),
                float(6 * (index // 64)),
            ),
        )
        for index in range(64)
    ]
    rng = Random(9182)

    plan = plan_modality_completion_batch(
        candidates,
        batch_size=32,
        geometry=SlabGeometry.cubic(4.0),
        rng=rng,
    )
    assignment_bytes = json.dumps(
        [
            (location.position_id, location.target_modality_id)
            for location in plan.locations
        ],
        separators=(",", ":"),
    ).encode()

    assert hashlib.sha256(assignment_bytes).hexdigest() == (
        "9432d3beb58ca9ede63151de2ad9e070d0b1f613cc9d6519c21154e45a10c30f"
    )
    assert rng.random() == 0.9147278789137382


def test_registered_512_conflict_build_avoids_scalar_pair_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = SlabGeometry.cubic(4.0)
    slabs = tuple(
        geometry.slab((float(5 * x), float(5 * y), float(5 * z)))
        for z in range(8)
        for y in range(8)
        for x in range(8)
    )

    def forbidden_scalar_intersection(
        _first: AxisAlignedSlab,
        _second: AxisAlignedSlab,
    ) -> bool:
        raise AssertionError("512-position conflict construction must stay vectorized")

    monkeypatch.setattr(AxisAlignedSlab, "intersects", forbidden_scalar_intersection)

    conflicts = _closed_patch_conflict_matrix(slabs)

    assert conflicts.shape == (512, 512)
    assert int(conflicts.sum()) == 512
