from __future__ import annotations

from collections import Counter
from random import Random

import pytest

from simple_brats.sampling import (
    ALL_MODALITY_IDS,
    V0_SLAB_GEOMETRY,
    CandidatePosition,
    ModalityCompletionPlanningError,
    PatchRole,
    plan_modality_completion_batch,
)


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
