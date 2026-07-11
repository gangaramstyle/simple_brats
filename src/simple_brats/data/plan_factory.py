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
    candidate_pool_size: int = 512,
    max_attempts: int = 8,
) -> object:
    """Create a deterministic, bounded-cost plan or fail without fallback.

    ``candidate_centers_mm`` must already come from a modality-agnostic,
    label-free validity mask on the locked extraction lattice. The factory
    canonicalizes their order, samples a bounded candidate pool, and delegates
    exact physical nonintersection and target-modality balancing to the core
    planner. The returned materialized record, rather than RNG replay, is the
    source of truth shared by objective arms.
    """

    from simple_brats.sampling import (  # local import avoids data-package cycles
        CandidatePosition,
        MaterializedPatchPlan,
        ModalityCompletionPlanningError,
        SlabGeometry,
        plan_modality_completion_batch,
    )

    if not isinstance(case, CaseRecord):
        raise TypeError("case must be a CaseRecord")
    if not isinstance(geometry, SlabGeometry):
        raise TypeError("geometry must be a SlabGeometry")
    if _SHA256_RE.fullmatch(extraction_spec_sha256) is None:
        raise ValueError("extraction_spec_sha256 must be a lowercase SHA-256 digest")
    if isinstance(target_count, bool) or not isinstance(target_count, int):
        raise ValueError("target_count must be an integer")
    if target_count < 8 or target_count % 4:
        raise ValueError("target_count must be a multiple of four with two or more per modality")
    for value, name in (
        (candidate_pool_size, "candidate_pool_size"),
        (max_attempts, "max_attempts"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if candidate_pool_size < target_count:
        raise ValueError("candidate_pool_size must be at least target_count")

    centers = (
        candidate_centers_mm
        if isinstance(candidate_centers_mm, CanonicalCandidateCenters)
        else CanonicalCandidateCenters(candidate_centers_mm)  # type: ignore[arg-type]
    )
    if len(centers) < target_count:
        raise PlanFactoryError(
            f"only {len(centers)} unique safe centers are available; {target_count} required"
        )

    plan_seed = stateless_plan_seed(
        data_manifest_sha256=data_manifest_sha256,
        case=case,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=experiment_seed,
    )
    pool_count = min(candidate_pool_size, len(centers))
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        attempt_payload = plan_seed.to_bytes(8, "big") + attempt.to_bytes(4, "big")
        attempt_seed = int.from_bytes(hashlib.sha256(attempt_payload).digest()[:8], "big")
        selected_indices = random.Random(attempt_seed).sample(range(len(centers)), pool_count)
        candidates = tuple(
            CandidatePosition(position_id=index, center_mm=centers.center(index))
            for index in selected_indices
        )
        try:
            batch_plan = plan_modality_completion_batch(
                candidates,
                batch_size=target_count,
                geometry=geometry,
                rng=attempt_seed,
            )
        except ModalityCompletionPlanningError as error:
            last_error = error
            continue
        return MaterializedPatchPlan.from_batch_plan(
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
    "CanonicalCandidateCenters",
    "PlanFactoryError",
    "materialize_matching_plan",
    "stateless_plan_seed",
]
