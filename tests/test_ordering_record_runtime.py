from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from itertools import product

import pytest
import torch

from simple_brats.config import ExperimentConfig, ModelConfig
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.real_batches import assemble_matching_batch
from simple_brats.sampling import (
    CandidatePosition,
    GeometryRecord,
    MaterializedPatchPlan,
    PatchIdentity,
    PatchPlanError,
    PlanCaseIdentity,
    SlabGeometry,
    plan_single_modality_ordering_batch,
)
from simple_brats.training import build_matching_system, validate_matching_batch

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
TARGET_MODALITY_ID = 3


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _case() -> CaseRecord:
    case_id = "BraTS-MET-00730-001"
    return CaseRecord.create(
        source="BraTS-MET",
        release="BraTS2026",
        case_id=case_id,
        files=tuple(
            FileRecord(
                modality=modality,
                path=f"{case_id}/{case_id}-{modality}.nii.gz",
                sha256=_digest(f"{case_id}:{modality}"),
            )
            for modality in MODALITIES
        ),
    )


def _target_centers() -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (float(x), float(y), float(z))
        for z in (-5, 0)
        for y in (-10, -5, 0, 5)
        for x in (-10, -5, 0, 5)
    )


def _record() -> MaterializedPatchPlan:
    case = _case()
    geometry = GeometryRecord.from_geometry(SlabGeometry.cubic(4.0))
    targets = tuple(
        PatchIdentity(
            position_id=position_id,
            modality_id=TARGET_MODALITY_ID,
            modality=MODALITIES[TARGET_MODALITY_ID],
            center_mm=center,
        )
        for position_id, center in enumerate(_target_centers())
    )
    sources = [
        PatchIdentity(
            position_id=position_id,
            modality_id=modality_id,
            modality=MODALITIES[modality_id],
            center_mm=targets[position_id].center_mm,
        )
        for modality_id in range(3)
        for position_id in range(30)
    ]
    target_source_centers = (
        (-10.0, 10.0, 10.0),
        (-5.0, 10.0, 10.0),
        (0.0, 10.0, 10.0),
        (5.0, 10.0, 10.0),
        (10.0, 10.0, 10.0),
        (10.0, 5.0, 10.0),
    )
    sources.extend(
        PatchIdentity(
            position_id=100 + index,
            modality_id=TARGET_MODALITY_ID,
            modality=MODALITIES[TARGET_MODALITY_ID],
            center_mm=center,
        )
        for index, center in enumerate(target_source_centers)
    )
    return MaterializedPatchPlan(
        data_manifest_sha256=DatasetManifest(cases=(case,)).sha256,
        case=PlanCaseIdentity(
            source=case.source,
            release=case.release,
            case_id=case.case_id,
            subject_id=case.subject_id,
            visit_id=case.visit_id,
        ),
        epoch=2,
        bag_index=17,
        seed=811,
        modality_names=MODALITIES,
        geometry=geometry,
        geometry_sha256=geometry.sha256,
        extraction_spec_sha256=_digest("extraction"),
        prism_anchor_mm=(0.0, 0.0, 0.0),
        prism_extent_mm=(32.0, 32.0, 32.0),
        target_modality_id=TARGET_MODALITY_ID,
        sources=tuple(sources),
        queries=targets,
        targets=targets,
    )


class _Extractor:
    extraction_spec_sha256 = _digest("extraction")

    def __call__(
        self,
        *,
        path: str,
        file_sha256: str,
        modality: str,
        center_mm: tuple[float, float, float],
        geometry: SlabGeometry,
    ) -> torch.Tensor:
        del path, file_sha256
        modality_id = MODALITIES.index(modality)
        value = 1_000.0 * modality_id + 10.0 * center_mm[0] + center_mm[1] + 0.1 * center_mm[2]
        return torch.full(geometry.model_shape, value, dtype=torch.float32)


def _assemble():
    plan = _record()
    case = _case()
    return (
        assemble_matching_batch(
            case,
            plan,
            _Extractor(),
            data_manifest_sha256=plan.data_manifest_sha256,
            plan_sha256=plan.sha256,
            extraction_spec_sha256=plan.extraction_spec_sha256,
        ),
        plan,
    )


def test_ordering_record_v2_round_trips_prism_and_enforces_physical_bounds() -> None:
    plan = _record()
    loaded = MaterializedPatchPlan.from_json(plan.to_json())

    assert loaded == plan
    assert loaded.schema_version == 2
    assert loaded.prism_anchor_mm == (0.0, 0.0, 0.0)
    assert loaded.prism_extent_mm == (32.0, 32.0, 32.0)
    assert loaded.target_modality_id == TARGET_MODALITY_ID
    assert len(loaded.sources) == 96
    assert len(loaded.targets) == len(loaded.queries) == 32
    with pytest.raises(PatchPlanError, match="requires a 32 mm cubic prism"):
        replace(plan, prism_extent_mm=(64.0, 64.0, 64.0))

    target_source = next(
        source for source in plan.sources if source.modality_id == TARGET_MODALITY_ID
    )
    outside = replace(target_source, center_mm=(15.0, 10.0, 10.0))
    with pytest.raises(PatchPlanError, match="fully contained"):
        replace(
            plan,
            sources=tuple(
                outside if item == target_source else item for item in plan.sources
            ),
        )

    overlapping = replace(target_source, center_mm=plan.targets[0].center_mm)
    with pytest.raises(PatchPlanError, match="must not intersect"):
        replace(
            plan,
            sources=tuple(overlapping if item == target_source else item for item in plan.sources),
        )


def test_ordering_sampler_plan_materializes_into_schema_v2() -> None:
    candidates = tuple(
        CandidatePosition(
            position_id=index,
            center_mm=(float(x), float(y), float(z)),
        )
        for index, (x, y, z) in enumerate(product((-12, -7, -2, 3, 8, 13), repeat=3))
    )
    sampled = plan_single_modality_ordering_batch(
        candidates,
        prism_anchor_mm=(0.0, 0.0, 0.0),
        prism_extent_mm=32.0,
        target_modality_id=TARGET_MODALITY_ID,
        geometry=SlabGeometry.cubic(4.0),
        rng=19,
    )
    case = _case()
    materialized = MaterializedPatchPlan.from_ordering_batch_plan(
        sampled,
        data_manifest_sha256=DatasetManifest(cases=(case,)).sha256,
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        subject_id=case.subject_id,
        visit_id=case.visit_id,
        epoch=0,
        bag_index=0,
        seed=19,
        extraction_spec_sha256=_digest("extraction"),
    )

    assert materialized.target_modality_id == TARGET_MODALITY_ID
    assert materialized.prism_extent_mm == (32.0, 32.0, 32.0)
    assert len(materialized.sources) == 96
    assert len(materialized.targets) == 32


def test_cpu_assembly_uses_prism_anchor_and_permuted_aligned_teacher_table() -> None:
    batch, plan = _assemble()
    repeated, _ = _assemble()

    assert batch.source_patches.shape[:2] == (1, 96)
    assert batch.target_patches.shape[:2] == (1, 32)
    assert torch.equal(batch.anchor_mm, torch.tensor([[0.0, 0.0, 0.0]]))
    assert not torch.equal(batch.target_position_ids, batch.query_position_ids)
    assert set(batch.target_pair_ids[0].tolist()) == set(batch.query_pair_ids[0].tolist())
    assert torch.equal(batch.target_position_ids, repeated.target_position_ids)
    assert torch.equal(batch.target_patches, repeated.target_patches)
    assert torch.all(batch.target_modality_ids == plan.target_modality_id)
    for index, center in enumerate(batch.target_coordinates_mm[0].tolist()):
        expected = (
            1_000.0 * plan.target_modality_id
            + 10.0 * center[0]
            + center[1]
            + 0.1 * center[2]
        )
        torch.testing.assert_close(
            batch.target_patches[0, index],
            torch.full_like(batch.target_patches[0, index], expected),
        )
    validate_matching_batch(batch, geometry=plan.geometry.to_geometry())


def test_runtime_validator_rejects_copy_paths_duplicates_and_coordinate_aliases() -> None:
    batch, plan = _assemble()
    target_source_indices = torch.nonzero(
        batch.source_modality_ids[0] == plan.target_modality_id,
        as_tuple=False,
    ).squeeze(1)
    overlap_index = int(target_source_indices[0])
    overlapping_coordinates = batch.source_coordinates_mm.clone()
    overlapping_coordinates[0, overlap_index] = batch.query_coordinates_mm[0, 0]
    with pytest.raises(ValueError, match="intersects a held target"):
        validate_matching_batch(
            replace(batch, source_coordinates_mm=overlapping_coordinates),
            geometry=plan.geometry.to_geometry(),
        )

    same_modality_indices = torch.nonzero(
        batch.source_modality_ids[0] == 0,
        as_tuple=False,
    ).squeeze(1)
    duplicate_positions = batch.source_position_ids.clone()
    duplicate_coordinates = batch.source_coordinates_mm.clone()
    duplicate_positions[0, same_modality_indices[1]] = duplicate_positions[
        0, same_modality_indices[0]
    ]
    duplicate_coordinates[0, same_modality_indices[1]] = duplicate_coordinates[
        0, same_modality_indices[0]
    ]
    with pytest.raises(ValueError, match="identities must be unique"):
        validate_matching_batch(
            replace(
                batch,
                source_position_ids=duplicate_positions,
                source_coordinates_mm=duplicate_coordinates,
            ),
            geometry=plan.geometry.to_geometry(),
        )

    inconsistent = batch.source_coordinates_mm.clone()
    repeated_position = int(batch.source_position_ids[0, 0])
    repeated_indices = torch.nonzero(
        batch.source_position_ids[0] == repeated_position,
        as_tuple=False,
    ).squeeze(1)
    inconsistent[0, repeated_indices[0], 0] += 0.25
    with pytest.raises(ValueError, match="inconsistent coordinates"):
        validate_matching_batch(
            replace(batch, source_coordinates_mm=inconsistent),
            geometry=plan.geometry.to_geometry(),
        )


def test_full_system_is_invariant_to_another_independent_teacher_permutation() -> None:
    batch, plan = _assemble()
    permutation = torch.arange(31, -1, -1)
    permuted = replace(
        batch,
        target_patches=batch.target_patches[:, permutation],
        target_modality_ids=batch.target_modality_ids[:, permutation],
        target_position_ids=batch.target_position_ids[:, permutation],
        target_coordinates_mm=batch.target_coordinates_mm[:, permutation],
        target_bag_ids=batch.target_bag_ids[:, permutation],
        target_pair_ids=batch.target_pair_ids[:, permutation],
    )
    config = ExperimentConfig(
        model=ModelConfig(width=24, depth=1, heads=3, mlp_ratio=1.0)
    )
    reference_system = build_matching_system(config).eval()
    permuted_system = copy.deepcopy(reference_system)

    reference = reference_system(batch)
    independently_permuted = permuted_system(permuted)

    torch.testing.assert_close(independently_permuted.loss, reference.loss)
    torch.testing.assert_close(independently_permuted.predictions, reference.predictions)
    validate_matching_batch(permuted, geometry=plan.geometry.to_geometry())
