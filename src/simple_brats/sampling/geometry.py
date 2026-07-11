"""Physical geometry and fail-closed sampling for 2.5D MRI slabs.

The sampling geometry is expressed in physical coordinates, independently of
the later resampling to a model tensor.  Intersections use closed axis-aligned
boxes: slabs that merely touch at a face, edge, or corner are conservatively
treated as intersecting.  This keeps target exclusion fail-closed even when a
downstream interpolator samples patch boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import TypeAlias

Coordinate3D: TypeAlias = tuple[float, float, float]
RngLike: TypeAlias = int | Random | None


class NonOverlappingSelectionError(RuntimeError):
    """Raised when the requested number of disjoint slabs cannot be selected."""


def _coordinate3d(value: Iterable[float], *, name: str) -> Coordinate3D:
    coordinate = tuple(float(component) for component in value)
    if len(coordinate) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    if not all(isfinite(component) for component in coordinate):
        raise ValueError(f"{name} must contain only finite coordinates")
    return coordinate  # type: ignore[return-value]


def _random(rng: RngLike) -> Random:
    if isinstance(rng, Random):
        return rng
    return Random(rng)


@dataclass(frozen=True, slots=True)
class SlabGeometry:
    """Geometry of a 2.5D slab in a three-axis physical coordinate system.

    ``model_shape`` describes the local tensor layout after extraction: its
    first two dimensions correspond to ``in_plane_axes`` and its final
    dimension corresponds to ``thin_axis``.  It does not change the physical
    intersection calculation.
    """

    in_plane_axes: tuple[int, int] = (0, 1)
    thin_axis: int = 2
    in_plane_footprint_mm: float = 4.0
    thin_extent_mm: float = 1.0
    model_shape: tuple[int, int, int] = (16, 16, 1)

    def __post_init__(self) -> None:
        if len(self.in_plane_axes) != 2:
            raise ValueError("in_plane_axes must contain exactly two axes")
        axes = (*self.in_plane_axes, self.thin_axis)
        if set(axes) != {0, 1, 2}:
            raise ValueError("in_plane_axes and thin_axis must be distinct and cover axes 0, 1, 2")
        if not isfinite(self.in_plane_footprint_mm) or self.in_plane_footprint_mm <= 0:
            raise ValueError("in_plane_footprint_mm must be finite and positive")
        if not isfinite(self.thin_extent_mm) or self.thin_extent_mm <= 0:
            raise ValueError("thin_extent_mm must be finite and positive")
        if len(self.model_shape) != 3 or any(size <= 0 for size in self.model_shape):
            raise ValueError("model_shape must contain three positive dimensions")

    @property
    def extents_mm(self) -> Coordinate3D:
        """Return full slab extents ordered by physical coordinate axis."""

        extents = [0.0, 0.0, 0.0]
        for axis in self.in_plane_axes:
            extents[axis] = float(self.in_plane_footprint_mm)
        extents[self.thin_axis] = float(self.thin_extent_mm)
        return tuple(extents)  # type: ignore[return-value]

    @property
    def half_extents_mm(self) -> Coordinate3D:
        return tuple(extent / 2.0 for extent in self.extents_mm)  # type: ignore[return-value]

    def slab(self, center_mm: Iterable[float]) -> AxisAlignedSlab:
        return AxisAlignedSlab(center_mm=_coordinate3d(center_mm, name="center_mm"), geometry=self)


V0_SLAB_GEOMETRY = SlabGeometry()


@dataclass(frozen=True, slots=True)
class AxisAlignedSlab:
    """A physical slab centered at ``center_mm``."""

    center_mm: Coordinate3D
    geometry: SlabGeometry = V0_SLAB_GEOMETRY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_mm",
            _coordinate3d(self.center_mm, name="center_mm"),
        )

    @property
    def bounds_mm(self) -> tuple[Coordinate3D, Coordinate3D]:
        """Return inclusive lower and upper bounds ordered by physical axis."""

        lower = tuple(
            center - half_extent
            for center, half_extent in zip(
                self.center_mm, self.geometry.half_extents_mm, strict=True
            )
        )
        upper = tuple(
            center + half_extent
            for center, half_extent in zip(
                self.center_mm, self.geometry.half_extents_mm, strict=True
            )
        )
        return lower, upper  # type: ignore[return-value]

    def intersects(self, other: AxisAlignedSlab) -> bool:
        """Return whether two closed physical slabs intersect on every axis."""

        self_lower, self_upper = self.bounds_mm
        other_lower, other_upper = other.bounds_mm
        return all(
            max(first_lower, second_lower) <= min(first_upper, second_upper)
            for first_lower, first_upper, second_lower, second_upper in zip(
                self_lower,
                self_upper,
                other_lower,
                other_upper,
                strict=True,
            )
        )


def slabs_intersect(first: AxisAlignedSlab, second: AxisAlignedSlab) -> bool:
    """Functional form of :meth:`AxisAlignedSlab.intersects`."""

    return first.intersects(second)


def are_pairwise_non_overlapping(slabs: Iterable[AxisAlignedSlab]) -> bool:
    """Return ``True`` only when no pair of slabs intersects."""

    slab_tuple = tuple(slabs)
    return all(
        not first.intersects(second)
        for index, first in enumerate(slab_tuple)
        for second in slab_tuple[index + 1 :]
    )


def select_non_overlapping_indices(
    candidate_centers_mm: Sequence[Iterable[float]],
    count: int,
    *,
    geometry: SlabGeometry = V0_SLAB_GEOMETRY,
    forbidden_slabs: Iterable[AxisAlignedSlab] = (),
    rng: RngLike = None,
) -> tuple[int, ...]:
    """Randomly select indices whose slabs are guaranteed not to intersect.

    The randomized depth-first search backtracks instead of weakening the
    exclusion rule.  It either returns exactly ``count`` valid indices or
    raises :class:`NonOverlappingSelectionError`; it never returns a partial or
    overlapping selection.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")

    centers = tuple(
        _coordinate3d(center, name=f"candidate_centers_mm[{index}]")
        for index, center in enumerate(candidate_centers_mm)
    )
    if count == 0:
        return ()
    if count > len(centers):
        raise NonOverlappingSelectionError(
            f"requested {count} slabs from only {len(centers)} candidates"
        )

    forbidden = tuple(forbidden_slabs)
    slabs = tuple(geometry.slab(center) for center in centers)
    eligible = [
        index
        for index, slab in enumerate(slabs)
        if not any(slab.intersects(excluded) for excluded in forbidden)
    ]
    if len(eligible) < count:
        raise NonOverlappingSelectionError(
            f"only {len(eligible)} candidates remain after exclusion; {count} required"
        )

    random = _random(rng)
    random.shuffle(eligible)

    # Cache conflicts once.  The physical intersection test remains the single
    # source of truth rather than relying on Euclidean center distance.
    conflicts = {
        tuple(sorted((first_index, second_index))): slabs[first_index].intersects(
            slabs[second_index]
        )
        for offset, first_index in enumerate(eligible)
        for second_index in eligible[offset + 1 :]
    }

    def conflict(first_index: int, second_index: int) -> bool:
        key = (
            (first_index, second_index)
            if first_index < second_index
            else (second_index, first_index)
        )
        return conflicts[key]

    def search(pool: tuple[int, ...], selected: tuple[int, ...]) -> tuple[int, ...] | None:
        remaining = count - len(selected)
        if remaining == 0:
            return selected
        if len(pool) < remaining:
            return None

        for offset, candidate_index in enumerate(pool):
            later_compatible = tuple(
                other_index
                for other_index in pool[offset + 1 :]
                if not conflict(candidate_index, other_index)
            )
            if len(later_compatible) < remaining - 1:
                continue
            result = search(later_compatible, (*selected, candidate_index))
            if result is not None:
                return result
        return None

    selected = search(tuple(eligible), ())
    if selected is None:
        raise NonOverlappingSelectionError(
            f"could not select {count} mutually non-overlapping slabs from "
            f"{len(eligible)} eligible candidates"
        )

    selected_slabs = tuple(slabs[index] for index in selected)
    if len(selected) != count or not are_pairwise_non_overlapping(selected_slabs):
        raise AssertionError("internal error: invalid non-overlapping selection")
    return selected


def select_non_overlapping_centers(
    candidate_centers_mm: Sequence[Iterable[float]],
    count: int,
    *,
    geometry: SlabGeometry = V0_SLAB_GEOMETRY,
    forbidden_slabs: Iterable[AxisAlignedSlab] = (),
    rng: RngLike = None,
) -> tuple[Coordinate3D, ...]:
    """Return centers selected by :func:`select_non_overlapping_indices`."""

    centers = tuple(
        _coordinate3d(center, name=f"candidate_centers_mm[{index}]")
        for index, center in enumerate(candidate_centers_mm)
    )
    indices = select_non_overlapping_indices(
        centers,
        count,
        geometry=geometry,
        forbidden_slabs=forbidden_slabs,
        rng=rng,
    )
    return tuple(centers[index] for index in indices)
