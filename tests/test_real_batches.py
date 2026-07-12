from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from itertools import product

import pytest
import torch
from torch import Tensor

from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.real_batches import (
    RealBatchAssemblyError,
    assemble_matching_batch,
)
from simple_brats.sampling import (
    CandidatePosition,
    MaterializedPatchPlan,
    SlabGeometry,
    canonical_sha256,
    plan_single_modality_ordering_batch,
)
from simple_brats.training import MatchingBatch, validate_matching_batch


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _case(*, case_id: str = "BraTS-MET-00730-001") -> CaseRecord:
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
            for modality in ("t1n", "t1c", "t2w", "t2f", "seg")
        ),
    )


def _plan(case: CaseRecord, manifest_sha256: str, extraction_sha256: str):
    candidates = tuple(
        CandidatePosition(
            position_id=index,
            center_mm=(float(x), float(y), float(z)),
        )
        for index, (x, y, z) in enumerate(product((-12, -7, -2, 3, 8, 13), repeat=3))
    )
    batch_plan = plan_single_modality_ordering_batch(
        candidates,
        prism_anchor_mm=(0.0, 0.0, 0.0),
        prism_extent_mm=32.0,
        target_modality_id=3,
        geometry=SlabGeometry.cubic(4.0),
        rng=811,
    )
    return MaterializedPatchPlan.from_ordering_batch_plan(
        batch_plan,
        data_manifest_sha256=manifest_sha256,
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        subject_id=case.subject_id,
        visit_id=case.visit_id,
        epoch=2,
        bag_index=17,
        seed=811,
        extraction_spec_sha256=extraction_sha256,
    )


@dataclass
class FakeExtractor:
    extraction_spec_sha256: str
    bad_shape: bool = False
    nonfinite: bool = False
    calls: list[dict[str, object]] = field(default_factory=list)

    def __call__(
        self,
        *,
        path: str,
        file_sha256: str,
        modality: str,
        center_mm: tuple[float, float, float],
        geometry: SlabGeometry,
    ) -> Tensor:
        self.calls.append(
            {
                "path": path,
                "file_sha256": file_sha256,
                "modality": modality,
                "center_mm": center_mm,
            }
        )
        shape = (8, 8, 1) if self.bad_shape else geometry.model_shape
        modality_id = ("t1n", "t1c", "t2w", "t2f").index(modality)
        value = 1_000.0 * modality_id + center_mm[0]
        result = torch.full(shape, value, dtype=torch.float32)
        if self.nonfinite:
            result.flatten()[0] = float("nan")
        return result


def _assemble(
    *,
    case: CaseRecord | None = None,
    manifest_sha256: str | None = None,
    plan: MaterializedPatchPlan | None = None,
    extractor: FakeExtractor | None = None,
    plan_sha256: str | None = None,
    extraction_sha256: str | None = None,
) -> tuple[MatchingBatch, MaterializedPatchPlan, FakeExtractor]:
    case = _case() if case is None else case
    manifest_sha256 = (
        DatasetManifest(cases=(case,)).sha256 if manifest_sha256 is None else manifest_sha256
    )
    extraction_sha256 = (
        canonical_sha256({"normalization": "patch-v0", "interpolation": "linear"})
        if extraction_sha256 is None
        else extraction_sha256
    )
    plan = _plan(case, manifest_sha256, extraction_sha256) if plan is None else plan
    extractor = FakeExtractor(extraction_sha256) if extractor is None else extractor
    batch = assemble_matching_batch(
        case,
        plan,
        extractor,
        data_manifest_sha256=manifest_sha256,
        plan_sha256=plan.sha256 if plan_sha256 is None else plan_sha256,
        extraction_spec_sha256=extraction_sha256,
    )
    return batch, plan, extractor


def _permute_batch_tables(batch: MatchingBatch) -> MatchingBatch:
    source_permutation = torch.randperm(
        batch.source_patches.shape[1], generator=torch.Generator().manual_seed(3)
    )
    target_permutation = torch.randperm(
        batch.target_patches.shape[1], generator=torch.Generator().manual_seed(5)
    )
    query_permutation = torch.randperm(
        batch.query_modality_ids.shape[1], generator=torch.Generator().manual_seed(7)
    )
    return replace(
        batch,
        source_patches=batch.source_patches[:, source_permutation],
        source_modality_ids=batch.source_modality_ids[:, source_permutation],
        source_position_ids=batch.source_position_ids[:, source_permutation],
        source_coordinates_mm=batch.source_coordinates_mm[:, source_permutation],
        query_modality_ids=batch.query_modality_ids[:, query_permutation],
        query_position_ids=batch.query_position_ids[:, query_permutation],
        query_coordinates_mm=batch.query_coordinates_mm[:, query_permutation],
        query_bag_ids=batch.query_bag_ids[:, query_permutation],
        query_pair_ids=batch.query_pair_ids[:, query_permutation],
        target_patches=batch.target_patches[:, target_permutation],
        target_modality_ids=batch.target_modality_ids[:, target_permutation],
        target_position_ids=batch.target_position_ids[:, target_permutation],
        target_coordinates_mm=batch.target_coordinates_mm[:, target_permutation],
        target_bag_ids=batch.target_bag_ids[:, target_permutation],
        target_pair_ids=batch.target_pair_ids[:, target_permutation],
    )


def test_replays_manifest_paths_without_ever_exposing_label_files() -> None:
    case = _case()
    batch, plan, extractor = _assemble(case=case)
    files = {record.modality: record for record in case.files}

    assert batch.source_patches.shape == (1, 96, 8, 8, 8)
    assert batch.target_patches.shape == (1, 32, 8, 8, 8)
    assert len(extractor.calls) == len(plan.sources) + len(plan.targets)
    assert {call["modality"] for call in extractor.calls} == {
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    }
    assert all(call["modality"] != "seg" for call in extractor.calls)
    for call in extractor.calls:
        modality = call["modality"]
        assert isinstance(modality, str)
        assert call["path"] == files[modality].path
        assert call["file_sha256"] == files[modality].sha256

    for index, patch in enumerate(plan.sources):
        expected = 1_000.0 * patch.modality_id + patch.center_mm[0]
        assert torch.all(batch.source_patches[0, index] == expected)
    for index in range(batch.target_patches.shape[1]):
        expected = (
            1_000.0 * int(batch.target_modality_ids[0, index])
            + float(batch.target_coordinates_mm[0, index, 0])
        )
        assert torch.all(batch.target_patches[0, index] == expected)


def test_query_and_target_identity_tables_are_independent_and_permutation_safe() -> None:
    batch, plan, _ = _assemble()
    geometry = plan.geometry.to_geometry()

    assert batch.query_modality_ids.data_ptr() != batch.target_modality_ids.data_ptr()
    assert batch.query_position_ids.data_ptr() != batch.target_position_ids.data_ptr()
    assert batch.query_coordinates_mm.data_ptr() != batch.target_coordinates_mm.data_ptr()
    assert batch.query_bag_ids.data_ptr() != batch.target_bag_ids.data_ptr()
    assert batch.query_pair_ids.data_ptr() != batch.target_pair_ids.data_ptr()
    validate_matching_batch(_permute_batch_tables(batch), geometry=geometry)


def test_anchor_is_stored_prism_center_and_permutation_invariant() -> None:
    batch, plan, _ = _assemble()
    expected = torch.tensor(plan.prism_anchor_mm, dtype=torch.float32)
    assert torch.equal(batch.anchor_mm[0], expected)

    permuted = _permute_batch_tables(batch)
    assert torch.equal(permuted.anchor_mm, batch.anchor_mm)
    validate_matching_batch(permuted, geometry=plan.geometry.to_geometry())


@pytest.mark.parametrize(
    ("pin", "message"),
    [
        ("manifest", "pinned data manifest"),
        ("plan", "does not match plan_sha256"),
        ("extraction", "pinned extraction specification"),
        ("extractor", "extractor implementation"),
    ],
)
def test_all_provenance_pins_fail_before_extraction(pin: str, message: str) -> None:
    case = _case()
    manifest_sha = DatasetManifest(cases=(case,)).sha256
    extraction_sha = canonical_sha256({"recipe": "v0"})
    plan = _plan(case, manifest_sha, extraction_sha)
    extractor = FakeExtractor(_digest("wrong extractor") if pin == "extractor" else extraction_sha)

    with pytest.raises(RealBatchAssemblyError, match=message):
        assemble_matching_batch(
            case,
            plan,
            extractor,
            data_manifest_sha256=(_digest("wrong manifest") if pin == "manifest" else manifest_sha),
            plan_sha256=_digest("wrong plan") if pin == "plan" else plan.sha256,
            extraction_spec_sha256=(
                _digest("wrong extraction") if pin == "extraction" else extraction_sha
            ),
        )
    assert extractor.calls == []


def test_case_identity_and_required_modality_paths_fail_before_extraction() -> None:
    original = _case()
    manifest_sha = DatasetManifest(cases=(original,)).sha256
    extraction_sha = _digest("extraction")
    plan = _plan(original, manifest_sha, extraction_sha)

    different_case = _case(case_id="BraTS-MET-00731-001")
    extractor = FakeExtractor(extraction_sha)
    with pytest.raises(RealBatchAssemblyError, match="case identity"):
        assemble_matching_batch(
            different_case,
            plan,
            extractor,
            data_manifest_sha256=manifest_sha,
            plan_sha256=plan.sha256,
            extraction_spec_sha256=extraction_sha,
        )
    assert extractor.calls == []

    missing_t2f = CaseRecord.create(
        source=original.source,
        release=original.release,
        case_id=original.case_id,
        files=(record for record in original.files if record.modality != "t2f"),
    )
    with pytest.raises(RealBatchAssemblyError, match="missing planned image modality paths"):
        assemble_matching_batch(
            missing_t2f,
            plan,
            extractor,
            data_manifest_sha256=manifest_sha,
            plan_sha256=plan.sha256,
            extraction_spec_sha256=extraction_sha,
        )
    assert extractor.calls == []


def test_label_modality_cannot_be_smuggled_through_a_valid_plan() -> None:
    case = _case()
    manifest_sha = DatasetManifest(cases=(case,)).sha256
    extraction_sha = _digest("extraction")
    plan = _plan(case, manifest_sha, extraction_sha)

    def relabel(patch):
        return replace(patch, modality="seg") if patch.modality_id == 3 else patch

    label_plan = replace(
        plan,
        modality_names=("t1n", "t1c", "t2w", "seg"),
        sources=tuple(relabel(patch) for patch in plan.sources),
        queries=tuple(relabel(patch) for patch in plan.queries),
        targets=tuple(relabel(patch) for patch in plan.targets),
    )
    extractor = FakeExtractor(extraction_sha)

    with pytest.raises(RealBatchAssemblyError, match="canonical image modalities"):
        assemble_matching_batch(
            case,
            label_plan,
            extractor,
            data_manifest_sha256=manifest_sha,
            plan_sha256=label_plan.sha256,
            extraction_spec_sha256=extraction_sha,
        )
    assert extractor.calls == []


@pytest.mark.parametrize(
    ("extractor", "message"),
    [
        (FakeExtractor(_digest("placeholder"), bad_shape=True), "must have shape"),
        (FakeExtractor(_digest("placeholder"), nonfinite=True), "only finite"),
    ],
)
def test_invalid_extractor_output_is_rejected(
    extractor: FakeExtractor,
    message: str,
) -> None:
    case = _case()
    manifest_sha = DatasetManifest(cases=(case,)).sha256
    extraction_sha = extractor.extraction_spec_sha256
    plan = _plan(case, manifest_sha, extraction_sha)
    with pytest.raises(RealBatchAssemblyError, match=message):
        assemble_matching_batch(
            case,
            plan,
            extractor,
            data_manifest_sha256=manifest_sha,
            plan_sha256=plan.sha256,
            extraction_spec_sha256=extraction_sha,
        )
