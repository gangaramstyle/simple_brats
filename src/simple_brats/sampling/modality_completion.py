"""Planning for balanced leave-one-modality-out completion batches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real
from random import Random

import numpy as np

from .geometry import (
    V0_PATCH_GEOMETRY,
    AxisAlignedSlab,
    Coordinate3D,
    RngLike,
    SlabGeometry,
)

BRATS_MODALITIES: tuple[str, str, str, str] = ("t1n", "t1c", "t2w", "t2f")
ALL_MODALITY_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)
ORDERING_TARGET_COUNT = 32
ORDERING_TARGET_MODALITY_SOURCE_COUNT = 6
ORDERING_OTHER_MODALITY_SOURCE_COUNT = 30
ORDERING_SOURCE_COUNT = (
    ORDERING_TARGET_MODALITY_SOURCE_COUNT
    + (len(ALL_MODALITY_IDS) - 1) * ORDERING_OTHER_MODALITY_SOURCE_COUNT
)
REGISTERED_ORDERING_SCALE_PAIRS: tuple[tuple[float, float], ...] = (
    (32.0, 4.0),
    (64.0, 8.0),
)


class ModalityCompletionPlanningError(RuntimeError):
    """Raised when no valid balanced, non-overlapping batch can be planned."""


class PatchRole(StrEnum):
    TARGET = "target"
    VISIBLE_SOURCE = "visible_source"


def _center(value: Iterable[float], *, name: str) -> Coordinate3D:
    # SlabGeometry owns the canonical finite/length validation.
    return V0_PATCH_GEOMETRY.patch(value).center_mm


def _modality_ids(value: Iterable[int], *, name: str) -> tuple[int, ...]:
    modality_ids = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in modality_ids):
        raise ValueError(f"{name} must contain integer modality IDs")
    if len(set(modality_ids)) != len(modality_ids):
        raise ValueError(f"{name} must not contain duplicate modality IDs")
    if not set(modality_ids).issubset(ALL_MODALITY_IDS):
        raise ValueError(f"{name} must be a subset of {ALL_MODALITY_IDS}")
    if not modality_ids:
        raise ValueError(f"{name} must contain at least one available modality")
    return tuple(sorted(modality_ids))


def _rng(value: RngLike) -> Random:
    return value if isinstance(value, Random) else Random(value)


def _prism_extent(value: Real | Iterable[float]) -> Coordinate3D:
    if isinstance(value, bool):
        raise ValueError("prism_extent_mm must be a positive scalar or length-three extent")
    if isinstance(value, Real):
        extent = (float(value),) * 3
    else:
        try:
            extent = tuple(float(component) for component in value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "prism_extent_mm must be a positive scalar or length-three extent"
            ) from error
    if len(extent) != 3 or not all(isfinite(component) and component > 0 for component in extent):
        raise ValueError("prism_extent_mm must contain three finite positive extents")
    return extent  # type: ignore[return-value]


def registered_ordering_prism_extent(
    geometry: SlabGeometry,
    prism_extent_mm: Real | Iterable[float] | None = None,
) -> Coordinate3D:
    """Resolve and validate one registered prism/physical-patch scale pair."""

    if not isinstance(geometry, SlabGeometry):
        raise TypeError("geometry must be a SlabGeometry")
    patch_extents = geometry.extents_mm
    if len(set(patch_extents)) != 1:
        raise ValueError("ordering batches require an isotropic physical patch")
    patch_edge = patch_extents[0]
    inferred = {
        patch_size: (prism_size, prism_size, prism_size)
        for prism_size, patch_size in REGISTERED_ORDERING_SCALE_PAIRS
    }.get(patch_edge)
    if inferred is None:
        raise ValueError(
            "ordering batches register only 4 mm cubes in 32 mm prisms and "
            "8 mm cubes in 64 mm prisms"
        )
    extent = inferred if prism_extent_mm is None else _prism_extent(prism_extent_mm)
    if extent != inferred:
        raise ValueError(f"a {patch_edge:g} mm cube requires a {inferred[0]:g} mm cubic prism")
    return extent


def _patch_is_inside_prism(
    center_mm: Coordinate3D,
    *,
    geometry: SlabGeometry,
    prism_anchor_mm: Coordinate3D,
    prism_extent_mm: Coordinate3D,
) -> bool:
    return all(
        abs(center - anchor) + patch_half <= prism_extent / 2.0
        for center, anchor, patch_half, prism_extent in zip(
            center_mm,
            prism_anchor_mm,
            geometry.half_extents_mm,
            prism_extent_mm,
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class PatchKey:
    """Stable patch identity that does not depend on tensor or tuple ordering."""

    position_id: int
    modality_id: int

    def __post_init__(self) -> None:
        if isinstance(self.position_id, bool) or not isinstance(self.position_id, int):
            raise ValueError("position_id must be an integer")
        if self.modality_id not in ALL_MODALITY_IDS:
            raise ValueError(f"modality_id must be one of {ALL_MODALITY_IDS}")


@dataclass(frozen=True, slots=True)
class CandidatePosition:
    """A foreground center and the modalities that actually exist there."""

    position_id: int
    center_mm: Coordinate3D
    available_modality_ids: tuple[int, ...] = ALL_MODALITY_IDS

    def __post_init__(self) -> None:
        if isinstance(self.position_id, bool) or not isinstance(self.position_id, int):
            raise ValueError("position_id must be an integer")
        object.__setattr__(
            self,
            "center_mm",
            _center(self.center_mm, name="center_mm"),
        )
        object.__setattr__(
            self,
            "available_modality_ids",
            _modality_ids(
                self.available_modality_ids,
                name="available_modality_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class PatchMetadata:
    """Permutation-safe metadata carried beside every patch tensor."""

    key: PatchKey
    center_mm: Coordinate3D
    role: PatchRole

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_mm",
            _center(self.center_mm, name="center_mm"),
        )
        if not isinstance(self.role, PatchRole):
            object.__setattr__(self, "role", PatchRole(self.role))

    @property
    def position_id(self) -> int:
        return self.key.position_id

    @property
    def modality_id(self) -> int:
        return self.key.modality_id


@dataclass(frozen=True, slots=True)
class CompletionLocationPlan:
    """One hidden target and all other available modalities at one location."""

    target: PatchMetadata
    visible_sources: tuple[PatchMetadata, ...]
    available_modality_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        available = _modality_ids(
            self.available_modality_ids,
            name="available_modality_ids",
        )
        object.__setattr__(self, "available_modality_ids", available)
        object.__setattr__(self, "visible_sources", tuple(self.visible_sources))
        if self.target.role is not PatchRole.TARGET:
            raise ValueError("target metadata must have the target role")
        if self.target.modality_id not in available:
            raise ValueError("target modality must be available at its position")

        expected_visible = set(available) - {self.target.modality_id}
        observed_visible = {patch.modality_id for patch in self.visible_sources}
        if observed_visible != expected_visible:
            raise ValueError(
                "visible_sources must contain every available modality except the target"
            )
        if len(observed_visible) != len(self.visible_sources):
            raise ValueError("visible_sources must not contain duplicate modalities")
        for source in self.visible_sources:
            if source.role is not PatchRole.VISIBLE_SOURCE:
                raise ValueError("visible source metadata has an invalid role")
            if source.position_id != self.target.position_id:
                raise ValueError("all patches in a location must share one position_id")
            if source.center_mm != self.target.center_mm:
                raise ValueError("all patches in a location must share one physical center")

    @property
    def position_id(self) -> int:
        return self.target.position_id

    @property
    def center_mm(self) -> Coordinate3D:
        return self.target.center_mm

    @property
    def target_modality_id(self) -> int:
        return self.target.modality_id


@dataclass(frozen=True, slots=True)
class ModalityCompletionBatchPlan:
    """A validated, exactly balanced leave-one-modality-out batch plan."""

    locations: tuple[CompletionLocationPlan, ...]
    geometry: SlabGeometry = V0_PATCH_GEOMETRY
    modality_names: tuple[str, str, str, str] = BRATS_MODALITIES

    def __post_init__(self) -> None:
        object.__setattr__(self, "locations", tuple(self.locations))
        object.__setattr__(self, "modality_names", tuple(self.modality_names))
        self.validate()

    @property
    def targets(self) -> tuple[PatchMetadata, ...]:
        return tuple(location.target for location in self.locations)

    @property
    def visible_sources(self) -> tuple[PatchMetadata, ...]:
        return tuple(patch for location in self.locations for patch in location.visible_sources)

    @property
    def target_counts(self) -> Mapping[int, int]:
        counts = Counter(target.modality_id for target in self.targets)
        return {modality_id: counts[modality_id] for modality_id in ALL_MODALITY_IDS}

    @property
    def patches_by_key(self) -> Mapping[PatchKey, PatchMetadata]:
        """Return an explicit identity map, independent of patch sequence order."""

        patches = (*self.targets, *self.visible_sources)
        return {patch.key: patch for patch in patches}

    def validate(self) -> None:
        if len(self.modality_names) != 4 or len(set(self.modality_names)) != 4:
            raise ValueError("modality_names must contain four distinct names")
        if not self.locations or len(self.locations) % 4 != 0:
            raise ValueError("a batch plan must contain a positive multiple of four locations")

        position_ids = tuple(location.position_id for location in self.locations)
        if len(set(position_ids)) != len(position_ids):
            raise ValueError("position_id values must be unique within a batch")

        expected_per_modality = len(self.locations) // 4
        expected_counts = {modality_id: expected_per_modality for modality_id in ALL_MODALITY_IDS}
        if self.target_counts != expected_counts:
            raise ValueError("target modalities must be exactly balanced within a batch")

        all_patches = (*self.targets, *self.visible_sources)
        if len(self.patches_by_key) != len(all_patches):
            raise ValueError("every patch must have a unique (position_id, modality_id) key")

        slabs = tuple(self.geometry.slab(location.center_mm) for location in self.locations)
        for index, first in enumerate(slabs):
            for second in slabs[index + 1 :]:
                if first.intersects(second):
                    raise ValueError("planned target locations must not intersect")

        # Explicitly verify the key leakage invariant.  A target modality is
        # permitted as a visible source at another location only if its slab is
        # physically disjoint from the held target slab.
        visible = self.visible_sources
        for target in self.targets:
            target_slab = self.geometry.slab(target.center_mm)
            for source in visible:
                if (
                    source.modality_id == target.modality_id
                    and source.position_id != target.position_id
                    and target_slab.intersects(self.geometry.slab(source.center_mm))
                ):
                    raise ValueError("a target modality is visible in an intersecting context slab")


@dataclass(frozen=True, slots=True)
class SingleModalityOrderingBatchPlan:
    """One local ordering task with a single target modality.

    The encoder sources are sampled independently of the target positions.  A
    source may therefore be co-located with a target only when its modality is
    different from ``target_modality_id``.
    """

    prism_anchor_mm: Coordinate3D
    prism_extent_mm: Coordinate3D
    target_modality_id: int
    sources: tuple[PatchMetadata, ...]
    targets: tuple[PatchMetadata, ...]
    geometry: SlabGeometry = V0_PATCH_GEOMETRY
    modality_names: tuple[str, str, str, str] = BRATS_MODALITIES

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, SlabGeometry):
            raise TypeError("geometry must be a SlabGeometry")
        anchor = _center(self.prism_anchor_mm, name="prism_anchor_mm")
        extent = registered_ordering_prism_extent(self.geometry, self.prism_extent_mm)
        if (
            isinstance(self.target_modality_id, bool)
            or not isinstance(self.target_modality_id, int)
            or self.target_modality_id not in ALL_MODALITY_IDS
        ):
            raise ValueError(f"target_modality_id must be one of {ALL_MODALITY_IDS}")
        names = tuple(self.modality_names)
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError("modality_names must contain four distinct names")
        sources = tuple(self.sources)
        targets = tuple(self.targets)
        object.__setattr__(self, "prism_anchor_mm", anchor)
        object.__setattr__(self, "prism_extent_mm", extent)
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "targets", targets)
        self.validate()

    @property
    def source_counts(self) -> Mapping[int, int]:
        counts = Counter(source.modality_id for source in self.sources)
        return {modality_id: counts[modality_id] for modality_id in ALL_MODALITY_IDS}

    @property
    def target_counts(self) -> Mapping[int, int]:
        counts = Counter(target.modality_id for target in self.targets)
        return {modality_id: counts[modality_id] for modality_id in ALL_MODALITY_IDS}

    @property
    def patches_by_key(self) -> Mapping[PatchKey, PatchMetadata]:
        patches = (*self.sources, *self.targets)
        return {patch.key: patch for patch in patches}

    def validate(self) -> None:
        if len(self.targets) != ORDERING_TARGET_COUNT:
            raise ValueError(f"ordering batches require exactly {ORDERING_TARGET_COUNT} targets")
        if len(self.sources) != ORDERING_SOURCE_COUNT:
            raise ValueError(f"ordering batches require exactly {ORDERING_SOURCE_COUNT} sources")

        expected_sources = {
            modality_id: (
                ORDERING_TARGET_MODALITY_SOURCE_COUNT
                if modality_id == self.target_modality_id
                else ORDERING_OTHER_MODALITY_SOURCE_COUNT
            )
            for modality_id in ALL_MODALITY_IDS
        }
        if self.source_counts != expected_sources:
            raise ValueError(
                "source counts must be 6 for the target modality and 30 for every other modality"
            )
        expected_targets = {
            modality_id: ORDERING_TARGET_COUNT if modality_id == self.target_modality_id else 0
            for modality_id in ALL_MODALITY_IDS
        }
        if self.target_counts != expected_targets:
            raise ValueError("all 32 targets must use target_modality_id")

        if any(source.role is not PatchRole.VISIBLE_SOURCE for source in self.sources):
            raise ValueError("every source must have the visible-source role")
        if any(target.role is not PatchRole.TARGET for target in self.targets):
            raise ValueError("every target must have the target role")
        source_keys = tuple(source.key for source in self.sources)
        target_keys = tuple(target.key for target in self.targets)
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("source patch keys must be unique")
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("target patch keys must be unique")
        if set(source_keys) & set(target_keys):
            raise ValueError("target patch identities must not appear among sources")

        center_by_position: dict[int, Coordinate3D] = {}
        position_by_center: dict[Coordinate3D, int] = {}
        for patch in (*self.sources, *self.targets):
            prior_center = center_by_position.setdefault(patch.position_id, patch.center_mm)
            if prior_center != patch.center_mm:
                raise ValueError("one position_id must map to exactly one physical center")
            prior_position = position_by_center.setdefault(patch.center_mm, patch.position_id)
            if prior_position != patch.position_id:
                raise ValueError("one physical center must map to exactly one position_id")
            if not _patch_is_inside_prism(
                patch.center_mm,
                geometry=self.geometry,
                prism_anchor_mm=self.prism_anchor_mm,
                prism_extent_mm=self.prism_extent_mm,
            ):
                raise ValueError("every full source and target footprint must be inside the prism")

        target_slabs = tuple(self.geometry.slab(target.center_mm) for target in self.targets)
        target_conflicts = _closed_patch_conflict_matrix(target_slabs)
        if bool((target_conflicts & ~np.eye(len(target_slabs), dtype=np.bool_)).any()):
            raise ValueError("target patches must be pairwise non-intersecting")

        target_modality_sources = tuple(
            source for source in self.sources if source.modality_id == self.target_modality_id
        )
        for source in target_modality_sources:
            source_slab = self.geometry.slab(source.center_mm)
            if any(source_slab.intersects(target_slab) for target_slab in target_slabs):
                raise ValueError("a target-modality source footprint intersects a target footprint")


def _location_plan(
    candidate: CandidatePosition,
    target_modality_id: int,
) -> CompletionLocationPlan:
    target = PatchMetadata(
        key=PatchKey(candidate.position_id, target_modality_id),
        center_mm=candidate.center_mm,
        role=PatchRole.TARGET,
    )
    visible_sources = tuple(
        PatchMetadata(
            key=PatchKey(candidate.position_id, modality_id),
            center_mm=candidate.center_mm,
            role=PatchRole.VISIBLE_SOURCE,
        )
        for modality_id in candidate.available_modality_ids
        if modality_id != target_modality_id
    )
    return CompletionLocationPlan(
        target=target,
        visible_sources=visible_sources,
        available_modality_ids=candidate.available_modality_ids,
    )


def _closed_patch_conflict_matrix(
    slabs: Sequence[AxisAlignedSlab],
) -> np.ndarray:
    """Return the exact closed-box intersection table as an immutable bool array.

    The planner evaluates this table for a bounded 512-position candidate pool.
    Computing every pair through Python repeatedly reconstructed identical bounds
    and dominated host CPU time.  This implementation constructs each slab's
    bounds once, then applies the same ``max(lower) <= min(upper)`` predicate in
    vectorized float64 operations.  It intentionally does not replace the
    predicate with a center-distance approximation: face, edge, and corner
    contact must continue to count as conflict, including at floating-point
    boundary values.
    """

    slab_tuple = tuple(slabs)
    if not all(isinstance(slab, AxisAlignedSlab) for slab in slab_tuple):
        raise TypeError("slabs must contain AxisAlignedSlab values")
    if not slab_tuple:
        result = np.empty((0, 0), dtype=np.bool_)
        result.setflags(write=False)
        return result

    bounds = tuple(slab.bounds_mm for slab in slab_tuple)
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    conflicts = np.ones((len(slab_tuple), len(slab_tuple)), dtype=np.bool_)
    for axis in range(3):
        conflicts &= np.maximum(
            lower[:, axis, None],
            lower[None, :, axis],
        ) <= np.minimum(
            upper[:, axis, None],
            upper[None, :, axis],
        )
    conflicts.setflags(write=False)
    return conflicts


def plan_modality_completion_batch(
    candidates: Sequence[CandidatePosition],
    *,
    batch_size: int,
    geometry: SlabGeometry = V0_PATCH_GEOMETRY,
    modality_names: Sequence[str] = BRATS_MODALITIES,
    rng: RngLike = None,
) -> ModalityCompletionBatchPlan:
    """Plan an exactly balanced, non-overlapping completion batch.

    At each selected foreground position, exactly one available modality is the
    hidden target and every other available modality is a visible source.  The
    joint randomized backtracking search assigns target modalities while it
    selects physical locations, so it cannot silently relax either target
    balance or slab exclusion.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if batch_size <= 0 or batch_size % 4 != 0:
        raise ValueError("batch_size must be a positive multiple of four")

    candidate_tuple = tuple(candidates)
    if len(candidate_tuple) < batch_size:
        raise ModalityCompletionPlanningError(
            f"requested {batch_size} locations from only {len(candidate_tuple)} candidates"
        )
    position_ids = tuple(candidate.position_id for candidate in candidate_tuple)
    if len(set(position_ids)) != len(position_ids):
        raise ValueError("candidate position_id values must be unique")

    names = tuple(modality_names)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("modality_names must contain four distinct names")

    random = _rng(rng)
    randomized_indices = list(range(len(candidate_tuple)))
    random.shuffle(randomized_indices)
    randomized_modalities = list(ALL_MODALITY_IDS)
    random.shuffle(randomized_modalities)
    candidate_rank = {
        candidate_index: rank for rank, candidate_index in enumerate(randomized_indices)
    }
    modality_rank = {modality_id: rank for rank, modality_id in enumerate(randomized_modalities)}

    slabs: tuple[AxisAlignedSlab, ...] = tuple(
        geometry.slab(candidate.center_mm) for candidate in candidate_tuple
    )
    conflicts = _closed_patch_conflict_matrix(slabs)
    initial_remaining = {modality_id: batch_size // 4 for modality_id in ALL_MODALITY_IDS}

    def search(
        eligible_indices: tuple[int, ...],
        remaining: dict[int, int],
        chosen: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...] | None:
        if not any(remaining.values()):
            return chosen
        if len(eligible_indices) < sum(remaining.values()):
            return None

        eligible_by_modality = {
            modality_id: tuple(
                candidate_index
                for candidate_index in eligible_indices
                if modality_id in candidate_tuple[candidate_index].available_modality_ids
            )
            for modality_id, required in remaining.items()
            if required
        }
        if any(
            len(eligible_by_modality[modality_id]) < required
            for modality_id, required in remaining.items()
            if required
        ):
            return None

        # Fill the modality with the least candidate slack first.  Random ranks
        # break equivalent choices without making tuple order semantically
        # meaningful.
        target_modality_id = min(
            eligible_by_modality,
            key=lambda modality_id: (
                len(eligible_by_modality[modality_id]) - remaining[modality_id],
                modality_rank[modality_id],
            ),
        )
        modality_candidates = sorted(
            eligible_by_modality[target_modality_id],
            key=candidate_rank.__getitem__,
        )
        for candidate_index in modality_candidates:
            next_eligible = tuple(
                other_index
                for other_index in eligible_indices
                if other_index != candidate_index and not conflicts[candidate_index][other_index]
            )
            next_remaining = dict(remaining)
            next_remaining[target_modality_id] -= 1
            result = search(
                next_eligible,
                next_remaining,
                (*chosen, (candidate_index, target_modality_id)),
            )
            if result is not None:
                return result
        return None

    assignments = search(
        tuple(randomized_indices),
        initial_remaining,
        (),
    )
    if assignments is None:
        raise ModalityCompletionPlanningError(
            "could not jointly satisfy exact modality balance, availability, "
            "and non-overlapping slab constraints"
        )

    locations = [
        _location_plan(candidate_tuple[candidate_index], target_modality_id)
        for candidate_index, target_modality_id in assignments
    ]
    random.shuffle(locations)
    return ModalityCompletionBatchPlan(
        locations=tuple(locations),
        geometry=geometry,
        modality_names=names,  # type: ignore[arg-type]
    )


def plan_single_modality_ordering_batch(
    candidates: Sequence[CandidatePosition],
    *,
    prism_anchor_mm: Iterable[float],
    prism_extent_mm: Real | Iterable[float],
    target_modality_id: int,
    geometry: SlabGeometry = V0_PATCH_GEOMETRY,
    modality_names: Sequence[str] = BRATS_MODALITIES,
    rng: RngLike = None,
    max_target_attempts: int = 32,
) -> SingleModalityOrderingBatchPlan:
    """Plan the registered single-modality 32-way local ordering task.

    Targets and sources are selected independently from one already-local,
    bounded foreground candidate pool.  Failure never relaxes target separation,
    source counts, prism containment, or target-modality footprint exclusion.
    """

    if (
        isinstance(target_modality_id, bool)
        or not isinstance(target_modality_id, int)
        or target_modality_id not in ALL_MODALITY_IDS
    ):
        raise ValueError(f"target_modality_id must be one of {ALL_MODALITY_IDS}")
    if (
        isinstance(max_target_attempts, bool)
        or not isinstance(max_target_attempts, int)
        or max_target_attempts <= 0
    ):
        raise ValueError("max_target_attempts must be a positive integer")
    anchor = _center(prism_anchor_mm, name="prism_anchor_mm")
    extent = registered_ordering_prism_extent(geometry, prism_extent_mm)
    names = tuple(modality_names)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("modality_names must contain four distinct names")

    candidate_tuple = tuple(candidates)
    if not all(isinstance(candidate, CandidatePosition) for candidate in candidate_tuple):
        raise TypeError("candidates must contain CandidatePosition values")
    position_ids = tuple(candidate.position_id for candidate in candidate_tuple)
    if len(set(position_ids)) != len(position_ids):
        raise ValueError("candidate position_id values must be unique")
    if any(
        not _patch_is_inside_prism(
            candidate.center_mm,
            geometry=geometry,
            prism_anchor_mm=anchor,
            prism_extent_mm=extent,
        )
        for candidate in candidate_tuple
    ):
        raise ValueError("every candidate full footprint must be inside the prism")

    eligible_targets = tuple(
        index
        for index, candidate in enumerate(candidate_tuple)
        if target_modality_id in candidate.available_modality_ids
    )
    if len(eligible_targets) < ORDERING_TARGET_COUNT:
        raise ModalityCompletionPlanningError(
            f"only {len(eligible_targets)} target-modality candidates are available; "
            f"{ORDERING_TARGET_COUNT} required"
        )
    for modality_id in ALL_MODALITY_IDS:
        required = (
            ORDERING_TARGET_MODALITY_SOURCE_COUNT
            if modality_id == target_modality_id
            else ORDERING_OTHER_MODALITY_SOURCE_COUNT
        )
        available = sum(
            modality_id in candidate.available_modality_ids for candidate in candidate_tuple
        )
        if available < required:
            raise ModalityCompletionPlanningError(
                f"only {available} modality-{modality_id} source candidates are available; "
                f"{required} required"
            )

    slabs = tuple(geometry.slab(candidate.center_mm) for candidate in candidate_tuple)
    conflicts = _closed_patch_conflict_matrix(slabs)
    random = _rng(rng)
    selected_targets: tuple[int, ...] | None = None
    for _attempt in range(max_target_attempts):
        target_order = list(eligible_targets)
        random.shuffle(target_order)
        chosen: list[int] = []
        for candidate_index in target_order:
            if not any(conflicts[candidate_index, prior] for prior in chosen):
                chosen.append(candidate_index)
                if len(chosen) == ORDERING_TARGET_COUNT:
                    break
        if len(chosen) != ORDERING_TARGET_COUNT:
            continue
        allowed_target_modality_sources = tuple(
            index
            for index, candidate in enumerate(candidate_tuple)
            if target_modality_id in candidate.available_modality_ids
            and not bool(conflicts[index, chosen].any())
        )
        if len(allowed_target_modality_sources) >= ORDERING_TARGET_MODALITY_SOURCE_COUNT:
            selected_targets = tuple(chosen)
            break
    if selected_targets is None:
        raise ModalityCompletionPlanningError(
            "could not select 32 pairwise-disjoint targets while retaining six disjoint "
            "target-modality source candidates"
        )

    sources: list[PatchMetadata] = []
    for modality_id in ALL_MODALITY_IDS:
        required = (
            ORDERING_TARGET_MODALITY_SOURCE_COUNT
            if modality_id == target_modality_id
            else ORDERING_OTHER_MODALITY_SOURCE_COUNT
        )
        eligible_sources = [
            index
            for index, candidate in enumerate(candidate_tuple)
            if modality_id in candidate.available_modality_ids
            and (
                modality_id != target_modality_id
                or not bool(conflicts[index, selected_targets].any())
            )
        ]
        random.shuffle(eligible_sources)
        if len(eligible_sources) < required:
            raise ModalityCompletionPlanningError(
                f"only {len(eligible_sources)} valid modality-{modality_id} sources remain; "
                f"{required} required"
            )
        sources.extend(
            PatchMetadata(
                key=PatchKey(candidate_tuple[index].position_id, modality_id),
                center_mm=candidate_tuple[index].center_mm,
                role=PatchRole.VISIBLE_SOURCE,
            )
            for index in eligible_sources[:required]
        )

    targets = [
        PatchMetadata(
            key=PatchKey(candidate_tuple[index].position_id, target_modality_id),
            center_mm=candidate_tuple[index].center_mm,
            role=PatchRole.TARGET,
        )
        for index in selected_targets
    ]
    random.shuffle(sources)
    random.shuffle(targets)
    return SingleModalityOrderingBatchPlan(
        prism_anchor_mm=anchor,
        prism_extent_mm=extent,
        target_modality_id=target_modality_id,
        sources=tuple(sources),
        targets=tuple(targets),
        geometry=geometry,
        modality_names=names,  # type: ignore[arg-type]
    )
