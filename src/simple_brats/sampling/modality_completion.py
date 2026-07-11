"""Planning for balanced leave-one-modality-out completion batches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from random import Random

from .geometry import (
    V0_SLAB_GEOMETRY,
    AxisAlignedSlab,
    Coordinate3D,
    RngLike,
    SlabGeometry,
)

BRATS_MODALITIES: tuple[str, str, str, str] = ("t1n", "t1c", "t2w", "t2f")
ALL_MODALITY_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)


class ModalityCompletionPlanningError(RuntimeError):
    """Raised when no valid balanced, non-overlapping batch can be planned."""


class PatchRole(StrEnum):
    TARGET = "target"
    VISIBLE_SOURCE = "visible_source"


def _center(value: Iterable[float], *, name: str) -> Coordinate3D:
    # SlabGeometry owns the canonical finite/length validation.
    return V0_SLAB_GEOMETRY.slab(value).center_mm


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
    geometry: SlabGeometry = V0_SLAB_GEOMETRY
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


def plan_modality_completion_batch(
    candidates: Sequence[CandidatePosition],
    *,
    batch_size: int,
    geometry: SlabGeometry = V0_SLAB_GEOMETRY,
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
    conflicts = tuple(tuple(first.intersects(second) for second in slabs) for first in slabs)
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
