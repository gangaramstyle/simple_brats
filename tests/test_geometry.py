from __future__ import annotations

import pytest

from simple_brats.sampling.geometry import (
    V0_CUBIC_GEOMETRY,
    V0_PATCH_GEOMETRY,
    V0_SLAB_GEOMETRY,
    NonOverlappingSelectionError,
    SlabGeometry,
    are_pairwise_non_overlapping,
    select_non_overlapping_indices,
)


def test_v0_geometry_has_declared_physical_and_model_extents() -> None:
    assert V0_SLAB_GEOMETRY.extents_mm == (4.0, 4.0, 1.0)
    assert V0_SLAB_GEOMETRY.model_shape == (16, 16, 1)

    coronal = SlabGeometry(in_plane_axes=(0, 2), thin_axis=1)
    assert coronal.extents_mm == (4.0, 1.0, 4.0)

    assert V0_CUBIC_GEOMETRY is V0_PATCH_GEOMETRY
    assert V0_CUBIC_GEOMETRY.extents_mm == (4.0, 4.0, 4.0)
    assert V0_CUBIC_GEOMETRY.model_shape == (16, 16, 16)
    assert SlabGeometry.cubic(8.0).extents_mm == (8.0, 8.0, 8.0)


@pytest.mark.parametrize(
    ("offset_mm", "expected"),
    [
        ((3.999, 0.0, 0.0), True),
        ((4.0, 0.0, 0.0), True),  # touching is excluded, fail-closed
        ((4.001, 0.0, 0.0), False),
        ((0.0, 0.0, 1.0), True),
        ((0.0, 0.0, 1.001), False),
        ((3.0, 5.0, 0.0), False),  # axis-aligned, not Euclidean distance
    ],
)
def test_exact_closed_box_intersection(offset_mm: tuple[float, ...], expected: bool) -> None:
    origin = V0_SLAB_GEOMETRY.slab((0.0, 0.0, 0.0))
    other = V0_SLAB_GEOMETRY.slab(offset_mm)
    assert origin.intersects(other) is expected
    assert other.intersects(origin) is expected


def test_intersection_respects_declared_thin_axis() -> None:
    geometry = SlabGeometry(in_plane_axes=(0, 2), thin_axis=1)
    origin = geometry.slab((0.0, 0.0, 0.0))

    assert origin.intersects(geometry.slab((0.0, 0.9, 0.0)))
    assert not origin.intersects(geometry.slab((0.0, 1.1, 0.0)))
    assert origin.intersects(geometry.slab((0.0, 0.0, 3.9)))


def test_random_selection_returns_only_pairwise_disjoint_slabs() -> None:
    centers = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),  # conflicts with the first
        (5.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
    ]
    selected = select_non_overlapping_indices(centers, 4, rng=7)

    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert are_pairwise_non_overlapping(V0_SLAB_GEOMETRY.slab(centers[index]) for index in selected)


def test_random_selection_raises_instead_of_returning_overlaps_or_partial_result() -> None:
    centers = [(float(index), 0.0, 0.0) for index in range(4)]

    with pytest.raises(NonOverlappingSelectionError):
        select_non_overlapping_indices(centers, 2, rng=0)


def test_forbidden_slabs_are_hard_exclusions() -> None:
    centers = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
    forbidden = (V0_SLAB_GEOMETRY.slab((0.0, 0.0, 0.0)),)

    assert select_non_overlapping_indices(
        centers,
        1,
        forbidden_slabs=forbidden,
        rng=0,
    ) == (1,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_plane_axes": (0, 0), "thin_axis": 2},
        {"in_plane_axes": (0, 1), "thin_axis": 1},
        {"in_plane_footprint_mm": 0.0},
        {"thin_extent_mm": -1.0},
    ],
)
def test_invalid_geometry_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SlabGeometry(**kwargs)  # type: ignore[arg-type]
