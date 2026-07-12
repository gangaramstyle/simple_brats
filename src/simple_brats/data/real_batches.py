"""Replay materialized patch plans into leakage-checked real-data batches.

This module deliberately does not know how a NIfTI volume is loaded or
resampled.  It binds a strict manifest record and a materialized patch plan to
a tiny extraction interface, then constructs the explicit identity tables used
by the matching objective.  Keeping extraction behind this boundary makes it
possible to test provenance and ordering without depending on an imaging
backend.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

from simple_brats.config import MODALITIES

from .manifest import CaseRecord, FileRecord

if TYPE_CHECKING:
    from simple_brats.sampling import MaterializedPatchPlan, PatchIdentity, SlabGeometry
    from simple_brats.training.matching import MatchingBatch

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RealBatchAssemblyError(ValueError):
    """Raised when provenance or extracted tensors cannot be replayed safely."""


class PatchExtractor(Protocol):
    """Minimal callable contract for one normalized physical patch.

    ``extraction_spec_sha256`` identifies the complete extraction recipe.  The
    callable receives only the manifest-bound image record and physical patch
    geometry; labels and any other case files are never supplied.
    """

    extraction_spec_sha256: str

    def __call__(
        self,
        *,
        path: str,
        file_sha256: str,
        modality: str,
        center_mm: tuple[float, float, float],
        geometry: SlabGeometry,
    ) -> Tensor:
        """Extract one normalized patch in ``geometry.model_shape``."""


def _pinned_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RealBatchAssemblyError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _verify_provenance(
    *,
    case: CaseRecord,
    plan: MaterializedPatchPlan,
    extractor: PatchExtractor,
    data_manifest_sha256: str,
    plan_sha256: str,
    extraction_spec_sha256: str,
) -> dict[str, FileRecord]:
    # Imported lazily so this module can safely be re-exported by
    # ``simple_brats.data``.  Sampling records depend on ``data.manifest``, and
    # importing them while the data package is initializing would form a
    # package-level cycle.
    from simple_brats.sampling.records import MaterializedPatchPlan as PlanType

    if not isinstance(case, CaseRecord):
        raise TypeError("case must be a CaseRecord")
    if not isinstance(plan, PlanType):
        raise TypeError("plan must be a MaterializedPatchPlan")
    if not callable(extractor):
        raise TypeError("extractor must be callable")

    manifest_pin = _pinned_sha256(data_manifest_sha256, "data_manifest_sha256")
    plan_pin = _pinned_sha256(plan_sha256, "plan_sha256")
    extraction_pin = _pinned_sha256(
        extraction_spec_sha256,
        "extraction_spec_sha256",
    )
    extractor_sha = _pinned_sha256(
        getattr(extractor, "extraction_spec_sha256", None),
        "extractor.extraction_spec_sha256",
    )

    if plan.data_manifest_sha256 != manifest_pin:
        raise RealBatchAssemblyError("patch plan was not created from the pinned data manifest")
    if plan.sha256 != plan_pin:
        raise RealBatchAssemblyError("materialized patch plan does not match plan_sha256")
    if plan.extraction_spec_sha256 != extraction_pin:
        raise RealBatchAssemblyError(
            "patch plan was not created for the pinned extraction specification"
        )
    if extractor_sha != extraction_pin:
        raise RealBatchAssemblyError(
            "extractor implementation does not match extraction_spec_sha256"
        )

    case_identity = (
        case.source,
        case.release,
        case.case_id,
        case.subject_id,
        case.visit_id,
    )
    plan_identity = (
        plan.case.source,
        plan.case.release,
        plan.case.case_id,
        plan.case.subject_id,
        plan.case.visit_id,
    )
    if case_identity != plan_identity:
        raise RealBatchAssemblyError(
            "patch plan case identity does not match the supplied manifest case"
        )

    # ``seg`` may legitimately be present in a MET case manifest, but no label
    # or ancillary image can become a model input through this API.  The plan
    # itself must name exactly the reviewed v0 image modalities and ordering.
    if tuple(plan.modality_names) != MODALITIES:
        raise RealBatchAssemblyError(
            f"patch plan modalities must be the canonical image modalities {MODALITIES}"
        )
    files_by_modality = {record.modality: record for record in case.files}
    missing = [modality for modality in MODALITIES if modality not in files_by_modality]
    if missing:
        raise RealBatchAssemblyError(
            f"manifest case is missing planned image modality paths: {missing}"
        )
    return {modality: files_by_modality[modality] for modality in MODALITIES}


def _extract_patch(
    extractor: PatchExtractor,
    *,
    patch: PatchIdentity,
    file: FileRecord,
    geometry: SlabGeometry,
) -> Tensor:
    if file.modality != patch.modality:
        raise RealBatchAssemblyError(
            f"manifest path for {file.modality!r} cannot satisfy {patch.modality!r} patch"
        )
    tensor = extractor(
        path=file.path,
        file_sha256=file.sha256,
        modality=patch.modality,
        center_mm=patch.center_mm,
        geometry=geometry,
    )
    if not isinstance(tensor, Tensor):
        raise RealBatchAssemblyError("extractor must return a torch.Tensor")
    allowed_shapes = {geometry.model_shape, (1, *geometry.model_shape)}
    if tuple(tensor.shape) not in allowed_shapes:
        raise RealBatchAssemblyError(
            "extracted patch must have shape "
            f"{geometry.model_shape} or {(1, *geometry.model_shape)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise RealBatchAssemblyError("extracted patches must use a floating-point dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise RealBatchAssemblyError("extracted patches must contain only finite values")
    # A data extractor is not a differentiable model component.  Detaching here
    # prevents an accidental graph from crossing the replay boundary.
    return tensor.detach().contiguous()


def _id_table(patches: tuple[PatchIdentity, ...], field: str) -> Tensor:
    values = (
        [patch.modality_id for patch in patches]
        if field == "modality"
        else [patch.position_id for patch in patches]
    )
    return torch.tensor(values, dtype=torch.long).unsqueeze(0)


def _coordinate_table(patches: tuple[PatchIdentity, ...]) -> Tensor:
    return torch.tensor(
        [patch.center_mm for patch in patches],
        dtype=torch.float32,
    ).unsqueeze(0)


def _bag_table(plan: MaterializedPatchPlan, count: int) -> Tensor:
    return torch.full((1, count), plan.bag_index, dtype=torch.long)


def _pair_table(patches: tuple[PatchIdentity, ...]) -> Tensor:
    # A plan has one target per position, so position_id is a stable pair key.
    # Modality and bag IDs remain explicit separate columns in the objective.
    return torch.tensor(
        [patch.position_id for patch in patches],
        dtype=torch.long,
    ).unsqueeze(0)


def _teacher_target_permutation(
    plan: MaterializedPatchPlan,
) -> tuple[tuple[PatchIdentity, ...], tuple[int, ...]]:
    """Return a deterministic nonidentity teacher order bound to plan identity."""

    domain = b"simple-brats.teacher-target-order.v1\0" + bytes.fromhex(plan.sha256)
    ranked = sorted(
        range(len(plan.targets)),
        key=lambda index: (
            hashlib.sha256(
                domain
                + b"\0"
                + str(plan.targets[index].position_id).encode("ascii")
                + b"\0"
                + str(plan.targets[index].modality_id).encode("ascii")
            ).digest(),
            index,
        ),
    )
    if ranked == list(range(len(ranked))) and len(ranked) > 1:
        ranked = [*ranked[1:], ranked[0]]
    indices = tuple(ranked)
    return tuple(plan.targets[index] for index in indices), indices


def assemble_matching_batch_from_patch_tables(
    case: CaseRecord,
    plan: MaterializedPatchPlan,
    extractor: PatchExtractor,
    *,
    source_patches: Tensor,
    target_patches: Tensor,
    data_manifest_sha256: str,
    plan_sha256: str,
    extraction_spec_sha256: str,
) -> MatchingBatch:
    """Bind already-extracted patch tables to the exact plan identities.

    This is the shared final assembly boundary for the reference per-patch CPU
    extractor and the explicit optimized batched GPU extractor.  Supplying
    pixels cannot bypass the same manifest, plan, extraction, ordering, and
    leakage validation used by the reference path.
    """

    from simple_brats.training.matching import MatchingBatch, validate_matching_batch

    _verify_provenance(
        case=case,
        plan=plan,
        extractor=extractor,
        data_manifest_sha256=data_manifest_sha256,
        plan_sha256=plan_sha256,
        extraction_spec_sha256=extraction_spec_sha256,
    )
    geometry = plan.geometry.to_geometry()
    expected_source = (1, len(plan.sources), *geometry.model_shape)
    expected_target = (1, len(plan.targets), *geometry.model_shape)
    if tuple(source_patches.shape) != expected_source or tuple(target_patches.shape) != (
        expected_target
    ):
        raise RealBatchAssemblyError(
            "pre-extracted source/target patch tables do not match the exact plan"
        )
    if (
        not source_patches.is_floating_point()
        or not target_patches.is_floating_point()
        or not bool(torch.isfinite(source_patches).all())
        or not bool(torch.isfinite(target_patches).all())
    ):
        raise RealBatchAssemblyError("pre-extracted patch tables must be finite floating point")

    query_modality_ids = _id_table(plan.queries, "modality")
    query_position_ids = _id_table(plan.queries, "position")
    query_coordinates_mm = _coordinate_table(plan.queries)
    query_bag_ids = _bag_table(plan, len(plan.queries))
    query_pair_ids = _pair_table(plan.queries)
    teacher_targets, teacher_permutation = _teacher_target_permutation(plan)
    target_patches = target_patches[
        :,
        torch.tensor(teacher_permutation, dtype=torch.long, device=target_patches.device),
    ]
    target_modality_ids = _id_table(teacher_targets, "modality")
    target_position_ids = _id_table(teacher_targets, "position")
    target_coordinates_mm = _coordinate_table(teacher_targets)
    target_bag_ids = _bag_table(plan, len(teacher_targets))
    target_pair_ids = _pair_table(teacher_targets)

    metadata_device = source_patches.device
    batch = MatchingBatch(
        source_patches=source_patches.detach().contiguous(),
        source_modality_ids=_id_table(plan.sources, "modality").to(metadata_device),
        source_position_ids=_id_table(plan.sources, "position").to(metadata_device),
        source_coordinates_mm=_coordinate_table(plan.sources).to(metadata_device),
        query_modality_ids=query_modality_ids.to(metadata_device),
        query_position_ids=query_position_ids.to(metadata_device),
        query_coordinates_mm=query_coordinates_mm.to(metadata_device),
        query_bag_ids=query_bag_ids.to(metadata_device),
        query_pair_ids=query_pair_ids.to(metadata_device),
        target_patches=target_patches.detach().contiguous(),
        target_modality_ids=target_modality_ids.to(metadata_device),
        target_position_ids=target_position_ids.to(metadata_device),
        target_coordinates_mm=target_coordinates_mm.to(metadata_device),
        target_bag_ids=target_bag_ids.to(metadata_device),
        target_pair_ids=target_pair_ids.to(metadata_device),
        anchor_mm=torch.tensor(plan.prism_anchor_mm, dtype=torch.float32)
        .unsqueeze(0)
        .to(metadata_device),
    )
    validate_matching_batch(batch, geometry=geometry)
    return batch


def assemble_matching_batch(
    case: CaseRecord,
    plan: MaterializedPatchPlan,
    extractor: PatchExtractor,
    *,
    data_manifest_sha256: str,
    plan_sha256: str,
    extraction_spec_sha256: str,
) -> MatchingBatch:
    """Replay one case/plan pair into a validated one-bag matching batch.

    Every provenance digest is supplied independently and checked before the
    extractor is called.  Source, query, and target metadata are allocated as
    separate tensors, so callers may independently permute their tables when
    the corresponding pixels/metadata move together.  Query pixels are never
    extracted, and case label records such as ``seg`` are never exposed to the
    extractor.
    """

    files = _verify_provenance(
        case=case,
        plan=plan,
        extractor=extractor,
        data_manifest_sha256=data_manifest_sha256,
        plan_sha256=plan_sha256,
        extraction_spec_sha256=extraction_spec_sha256,
    )
    geometry = plan.geometry.to_geometry()

    source_patches = torch.stack(
        [
            _extract_patch(
                extractor,
                patch=patch,
                file=files[patch.modality],
                geometry=geometry,
            )
            for patch in plan.sources
        ]
    ).unsqueeze(0)
    target_patches = torch.stack(
        [
            _extract_patch(
                extractor,
                patch=patch,
                file=files[patch.modality],
                geometry=geometry,
            )
            for patch in plan.targets
        ]
    ).unsqueeze(0)

    return assemble_matching_batch_from_patch_tables(
        case,
        plan,
        extractor,
        source_patches=source_patches,
        target_patches=target_patches,
        data_manifest_sha256=data_manifest_sha256,
        plan_sha256=plan_sha256,
        extraction_spec_sha256=extraction_spec_sha256,
    )


__all__ = [
    "PatchExtractor",
    "RealBatchAssemblyError",
    "assemble_matching_batch",
    "assemble_matching_batch_from_patch_tables",
]
