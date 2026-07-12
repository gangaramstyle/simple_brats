"""Deterministically materialize objective-agnostic patch plans from a safe lattice."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .manifest import CaseRecord

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PlanFactoryError(RuntimeError):
    """A valid materialized plan could not be constructed without relaxing invariants."""


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalCandidateCenters:
    """Canonical, immutable coordinates safe to reuse across materialized plans.

    Constructing this record performs the same numeric validation,
    deduplication, and lexicographic ordering historically performed by
    :func:`materialize_matching_plan`.  The resulting little-endian float64
    array is backed by immutable ``bytes`` so callers cannot re-enable writes
    through NumPy's ``setflags`` API.
    """

    values: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        centers: list[tuple[float, float, float]] = []
        for index, center in enumerate(self.values):
            try:
                normalized = tuple(float(component) for component in center)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"candidate center {index} must be numeric") from error
            if len(normalized) != 3 or not all(isfinite(component) for component in normalized):
                raise ValueError(f"candidate center {index} must contain three finite values")
            centers.append(normalized)  # type: ignore[arg-type]

        ordered = sorted(set(centers))
        canonical = np.empty((len(ordered), 3), dtype=np.dtype("<f8"), order="C")
        if ordered:
            canonical[:] = ordered
        immutable = np.frombuffer(canonical.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
            (-1, 3)
        )
        object.__setattr__(self, "values", immutable)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def center(self, index: int) -> tuple[float, float, float]:
        row = self.values[index]
        return tuple(float(component) for component in row)  # type: ignore[return-value]


def stateless_plan_seed(
    *,
    data_manifest_sha256: str,
    case: CaseRecord,
    epoch: int,
    bag_index: int,
    experiment_seed: int,
) -> int:
    """Derive one uint64 seed from immutable experiment and bag identity."""

    if _SHA256_RE.fullmatch(data_manifest_sha256) is None:
        raise ValueError("data_manifest_sha256 must be a lowercase SHA-256 digest")
    for value, name in (
        (epoch, "epoch"),
        (bag_index, "bag_index"),
        (experiment_seed, "experiment_seed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    payload = "\0".join(
        (
            "simple-brats-plan-v1",
            data_manifest_sha256,
            case.source,
            case.release,
            case.case_id,
            case.subject_id,
            case.visit_id,
            str(epoch),
            str(bag_index),
            str(experiment_seed),
        )
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def balanced_target_modality_id(
    *,
    data_manifest_sha256: str,
    case: CaseRecord,
    epoch: int,
    bag_index: int,
    experiment_seed: int,
) -> int:
    """Choose one target modality in a balanced-random four-bag cycle."""

    if _SHA256_RE.fullmatch(data_manifest_sha256) is None:
        raise ValueError("data_manifest_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(case, CaseRecord):
        raise TypeError("case must be a CaseRecord")
    for value, name in (
        (epoch, "epoch"),
        (bag_index, "bag_index"),
        (experiment_seed, "experiment_seed"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    block_index, cycle_slot = divmod(bag_index, 4)
    payload = "\0".join(
        (
            "simple-brats-ordering-modality-cycle-v1",
            data_manifest_sha256,
            case.subject_id,
            str(epoch),
            str(block_index),
            str(experiment_seed),
        )
    ).encode()
    modalities = [0, 1, 2, 3]
    random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")).shuffle(modalities)
    return modalities[cycle_slot]


def materialize_matching_plan(
    *,
    case: CaseRecord,
    data_manifest_sha256: str,
    candidate_centers_mm: Sequence[Sequence[float]] | CanonicalCandidateCenters,
    geometry: object,
    extraction_spec_sha256: str,
    epoch: int,
    bag_index: int,
    experiment_seed: int,
    target_count: int = 32,
    prism_extent_mm: float | Sequence[float] | None = None,
    candidate_pool_size: int = 512,
    max_attempts: int = 8,
) -> object:
    """Create a deterministic, bounded-cost plan or fail without fallback.

    ``candidate_centers_mm`` must already come from a modality-agnostic,
    label-free validity mask on the locked extraction lattice. The factory
    canonicalizes their order, chooses one foreground anchor, filters to the
    registered local prism, and only then samples a bounded candidate pool. The
    returned materialized record, rather than RNG replay, is the source of truth
    shared by objective arms.
    """

    from simple_brats.sampling import (  # local import avoids data-package cycles
        ORDERING_TARGET_MODALITY_SOURCE_COUNT,
        CandidatePosition,
        MaterializedPatchPlan,
        ModalityCompletionPlanningError,
        SlabGeometry,
        plan_single_modality_ordering_batch,
        registered_ordering_prism_extent,
    )

    if not isinstance(case, CaseRecord):
        raise TypeError("case must be a CaseRecord")
    if not isinstance(geometry, SlabGeometry):
        raise TypeError("geometry must be a SlabGeometry")
    if _SHA256_RE.fullmatch(extraction_spec_sha256) is None:
        raise ValueError("extraction_spec_sha256 must be a lowercase SHA-256 digest")
    if isinstance(target_count, bool) or not isinstance(target_count, int):
        raise ValueError("target_count must be an integer")
    if target_count != 32:
        raise ValueError("the registered ordering task requires exactly 32 targets")
    for value, name in (
        (candidate_pool_size, "candidate_pool_size"),
        (max_attempts, "max_attempts"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    minimum_pool_size = target_count + ORDERING_TARGET_MODALITY_SOURCE_COUNT
    if candidate_pool_size < minimum_pool_size:
        raise ValueError(
            f"candidate_pool_size must be at least {minimum_pool_size} for disjoint "
            "targets and target-modality sources"
        )

    centers = (
        candidate_centers_mm
        if isinstance(candidate_centers_mm, CanonicalCandidateCenters)
        else CanonicalCandidateCenters(candidate_centers_mm)  # type: ignore[arg-type]
    )
    if len(centers) < minimum_pool_size:
        raise PlanFactoryError(
            f"only {len(centers)} unique safe centers are available; "
            f"at least {minimum_pool_size} required"
        )

    plan_seed = stateless_plan_seed(
        data_manifest_sha256=data_manifest_sha256,
        case=case,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=experiment_seed,
    )
    resolved_prism_extent = registered_ordering_prism_extent(geometry, prism_extent_mm)
    target_modality_id = balanced_target_modality_id(
        data_manifest_sha256=data_manifest_sha256,
        case=case,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=experiment_seed,
    )
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        attempt_payload = plan_seed.to_bytes(8, "big") + attempt.to_bytes(4, "big")
        attempt_seed = int.from_bytes(hashlib.sha256(attempt_payload).digest()[:8], "big")
        attempt_random = random.Random(attempt_seed)
        anchor_index = attempt_random.randrange(len(centers))
        anchor_mm = centers.center(anchor_index)
        maximum_center_delta = np.asarray(
            resolved_prism_extent, dtype=np.float64
        ) / 2.0 - np.asarray(geometry.half_extents_mm, dtype=np.float64)
        local_mask = (
            np.abs(centers.values - np.asarray(anchor_mm, dtype=np.float64))
            <= maximum_center_delta[None, :]
        ).all(axis=1)
        local_indices = np.flatnonzero(local_mask).tolist()
        if len(local_indices) < target_count:
            last_error = ModalityCompletionPlanningError(
                f"anchor has only {len(local_indices)} fully-contained local centers"
            )
            continue
        pool_count = min(candidate_pool_size, len(local_indices))
        selected_indices = attempt_random.sample(local_indices, pool_count)
        candidates = tuple(
            CandidatePosition(position_id=index, center_mm=centers.center(index))
            for index in selected_indices
        )
        try:
            batch_plan = plan_single_modality_ordering_batch(
                candidates,
                prism_anchor_mm=anchor_mm,
                prism_extent_mm=resolved_prism_extent,
                target_modality_id=target_modality_id,
                geometry=geometry,
                rng=attempt_random,
            )
        except ModalityCompletionPlanningError as error:
            last_error = error
            continue
        return MaterializedPatchPlan.from_ordering_batch_plan(
            batch_plan,
            data_manifest_sha256=data_manifest_sha256,
            source=case.source,
            release=case.release,
            case_id=case.case_id,
            subject_id=case.subject_id,
            visit_id=case.visit_id,
            epoch=epoch,
            bag_index=bag_index,
            seed=plan_seed,
            extraction_spec_sha256=extraction_spec_sha256,
        )

    raise PlanFactoryError(
        f"could not construct a valid plan after {max_attempts} bounded attempts"
    ) from last_error


__all__ = [
    "balanced_target_modality_id",
    "CanonicalCandidateCenters",
    "PlanFactoryError",
    "materialize_matching_plan",
    "stateless_plan_seed",
]
