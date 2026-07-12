from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import simple_brats.data.pipeline as pipeline_module
from simple_brats.data.extraction import ExtractionSpec
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord, sha256_file
from simple_brats.data.pipeline import (
    CachedNiftiPatchExtractor,
    DataPipelineError,
    PreparedCasePlan,
    materialize_case_matching_plan_record,
    prepare_case_candidate_universe,
    prepare_case_matching_plan,
    prepare_case_matching_plan_record,
)
from simple_brats.data.plan_factory import materialize_matching_plan
from simple_brats.sampling import V0_SLAB_GEOMETRY, SlabGeometry

IDENTITY_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_nifti(path: Path, *, offset: float, shape: tuple[int, int, int]) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0 + offset
    image = nib.Nifti1Image(values, np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def _dataset(
    root: Path,
    *,
    shape: tuple[int, int, int] = (16, 16, 2),
    include_seg: bool = True,
    cubic: bool = False,
) -> tuple[CaseRecord, DatasetManifest, ExtractionSpec]:
    case_id = "BraTS-MET-00001-000"
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    records = []
    for index, modality in enumerate(MODALITIES):
        path = case_dir / f"{case_id}-{modality}.nii.gz"
        _write_nifti(path, offset=1000.0 * index, shape=shape)
        records.append(
            FileRecord(
                modality,
                path.relative_to(root).as_posix(),
                sha256_file(path),
            )
        )
    if include_seg:
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        segmentation = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4))
        segmentation.header.set_xyzt_units("mm")
        nib.save(segmentation, seg_path)
        records.append(
            FileRecord("seg", seg_path.relative_to(root).as_posix(), sha256_file(seg_path))
        )
    case = CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=records,
    )
    manifest = DatasetManifest(cases=(case,))
    spec = ExtractionSpec(
        canonical_shape=shape,
        canonical_affine=IDENTITY_AFFINE,
        **(
            {
                "patch_source_shape": (4, 4, 4),
                "patch_physical_extent_mm": (4.0, 4.0, 4.0),
                "model_visible_shape": (16, 16, 16),
            }
            if cubic
            else {}
        ),
    )
    return case, manifest, spec


def _extractor(
    root: Path,
    manifest: DatasetManifest,
    spec: ExtractionSpec,
    *,
    max_cached_volumes: int = 4,
) -> CachedNiftiPatchExtractor:
    return CachedNiftiPatchExtractor(
        data_root=root,
        manifest=manifest,
        data_manifest_sha256=manifest.sha256,
        extraction_spec=spec,
        max_cached_volumes=max_cached_volumes,
    )


def _file(case: CaseRecord, modality: str) -> FileRecord:
    return next(record for record in case.files if record.modality == modality)


def _call(
    extractor: CachedNiftiPatchExtractor,
    record: FileRecord,
    *,
    modality: str | None = None,
    file_sha256: str | None = None,
) -> torch.Tensor:
    return extractor(
        path=record.path,
        file_sha256=record.sha256 if file_sha256 is None else file_sha256,
        modality=record.modality if modality is None else modality,
        center_mm=(5.5, 5.5, 1.0),
        geometry=V0_SLAB_GEOMETRY,
    )


def test_returns_protocol_shaped_tensor_and_uses_bounded_lru(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, manifest, spec = _dataset(tmp_path)
    extractor = _extractor(tmp_path, manifest, spec, max_cached_volumes=2)
    original_prepare = pipeline_module.prepare_canonical_volume
    prepared_paths: list[str] = []

    def counted_prepare(path: Path, extraction_spec: ExtractionSpec):
        prepared_paths.append(path.name)
        return original_prepare(path, extraction_spec)

    monkeypatch.setattr(pipeline_module, "prepare_canonical_volume", counted_prepare)
    t1n = _file(case, "t1n")
    t1c = _file(case, "t1c")
    t2w = _file(case, "t2w")

    first = _call(extractor, t1n)
    second = _call(extractor, t1n)
    _call(extractor, t1c)
    _call(extractor, t2w)
    _call(extractor, t1c)
    _call(extractor, t1n)

    assert first.shape == (16, 16, 1)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert extractor.extraction_spec_sha256 == spec.sha256
    assert extractor.cache_size == 2
    assert prepared_paths.count(t1n.path.rsplit("/", 1)[-1]) == 2
    assert len(prepared_paths) == 4


def test_rejects_manifest_modality_digest_and_geometry_mismatch_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, manifest, spec = _dataset(tmp_path)
    extractor = _extractor(tmp_path, manifest, spec)
    t1n = _file(case, "t1n")

    def forbidden_prepare(*args: object, **kwargs: object):
        raise AssertionError("image preparation should not have been reached")

    monkeypatch.setattr(pipeline_module, "prepare_canonical_volume", forbidden_prepare)
    with pytest.raises(DataPipelineError, match="registered for 't1n', not 't1c'"):
        _call(extractor, t1n, modality="t1c")
    with pytest.raises(DataPipelineError, match="does not match the pinned manifest"):
        _call(extractor, t1n, file_sha256=_digest("wrong"))
    with pytest.raises(DataPipelineError, match="geometry does not match"):
        extractor(
            path=t1n.path,
            file_sha256=t1n.sha256,
            modality=t1n.modality,
            center_mm=(5.5, 5.5, 1.0),
            geometry=SlabGeometry(in_plane_footprint_mm=8.0),
        )


def test_raw_sha_is_verified_before_nifti_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, manifest, spec = _dataset(tmp_path)
    t1n = _file(case, "t1n")
    altered = FileRecord(t1n.modality, t1n.path, _digest("not the file"))
    altered_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=(altered, *[record for record in case.files if record.modality != "t1n"]),
    )
    altered_manifest = DatasetManifest(cases=(altered_case,))
    extractor = _extractor(tmp_path, altered_manifest, spec)

    def forbidden_prepare(*args: object, **kwargs: object):
        raise AssertionError("digest mismatch must fail before NIfTI loading")

    monkeypatch.setattr(pipeline_module, "prepare_canonical_volume", forbidden_prepare)
    with pytest.raises(DataPipelineError, match="SHA mismatch before loading"):
        _call(extractor, altered)
    assert extractor.cache_size == 0


@pytest.mark.parametrize("bad_path", ["../outside.nii.gz", "/absolute.nii.gz", "a/./b.nii.gz"])
def test_noncanonical_or_escaping_manifest_paths_are_rejected(
    tmp_path: Path,
    bad_path: str,
) -> None:
    case, _, spec = _dataset(tmp_path)
    t1n = _file(case, "t1n")
    bad_record = FileRecord("t1n", bad_path, t1n.sha256)
    bad_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=(bad_record, *[record for record in case.files if record.modality != "t1n"]),
    )
    manifest = DatasetManifest(cases=(bad_case,))

    with pytest.raises(DataPipelineError, match="canonical and data-root-relative"):
        _extractor(tmp_path, manifest, spec)


def test_symlink_in_manifest_path_is_rejected_at_first_access(tmp_path: Path) -> None:
    case, _, spec = _dataset(tmp_path)
    original = _file(case, "t1n")
    original_path = tmp_path / original.path
    link_path = original_path.with_name("linked-t1n.nii.gz")
    link_path.symlink_to(original_path)
    link_record = FileRecord(
        "t1n",
        link_path.relative_to(tmp_path).as_posix(),
        original.sha256,
    )
    linked_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=(link_record, *[record for record in case.files if record.modality != "t1n"]),
    )
    manifest = DatasetManifest(cases=(linked_case,))
    extractor = _extractor(tmp_path, manifest, spec)

    with pytest.raises(DataPipelineError, match="traverses a symlink"):
        _call(extractor, link_record)


def test_manifest_requires_exact_four_images_and_only_reviewed_optional_records(
    tmp_path: Path,
) -> None:
    case, _, spec = _dataset(tmp_path)
    missing_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=tuple(record for record in case.files if record.modality != "t2f"),
    )
    missing_manifest = DatasetManifest(cases=(missing_case,))
    with pytest.raises(DataPipelineError, match="exactly the four"):
        _extractor(tmp_path, missing_manifest, spec)

    adc_path = tmp_path / "adc.nii.gz"
    _write_nifti(adc_path, offset=9_000.0, shape=spec.canonical_shape)
    adc = FileRecord("adc", "adc.nii.gz", sha256_file(adc_path))
    unexpected_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=(*case.files, adc),
    )
    unexpected_manifest = DatasetManifest(cases=(unexpected_case,))
    with pytest.raises(DataPipelineError, match="unreviewed modalities"):
        _extractor(tmp_path, unexpected_manifest, spec)


def test_case_helper_materializes_deterministic_sha_bound_label_free_plan(
    tmp_path: Path,
) -> None:
    case, manifest, spec = _dataset(tmp_path, shape=(40, 40, 40), cubic=True)
    extractor = _extractor(tmp_path, manifest, spec)

    prepared = prepare_case_matching_plan_record(
        extractor,
        case,
        epoch=2,
        bag_index=17,
        experiment_seed=29,
        target_count=32,
        candidate_pool_size=512,
    )
    second_prepared = prepare_case_matching_plan_record(
        extractor,
        case,
        epoch=2,
        bag_index=17,
        experiment_seed=29,
        target_count=32,
        candidate_pool_size=512,
    )
    first = prepared.plan
    second = second_prepared.plan

    assert first.sha256 == second.sha256
    assert prepared.sha256 == second_prepared.sha256
    assert prepared.candidate_count >= 512
    assert len(prepared.candidate_centers_sha256) == 64
    assert tuple(item.modality for item in prepared.volume_digests) == MODALITIES
    assert prepared.to_json().startswith('{"candidate_centers_sha256"')
    assert first.data_manifest_sha256 == manifest.sha256
    assert first.extraction_spec_sha256 == spec.sha256
    assert len(first.targets) == 32
    assert len(first.sources) == 96
    assert {patch.modality_id for patch in first.targets} == {first.target_modality_id}
    assert extractor.cache_size == 4
    assert all("seg" not in patch.modality for patch in (*first.sources, *first.targets))

    replay_plan = prepare_case_matching_plan(
        extractor,
        case,
        epoch=2,
        bag_index=17,
        experiment_seed=29,
        target_count=32,
        candidate_pool_size=512,
    )
    assert replay_plan.sha256 == first.sha256


def test_prepared_candidate_universe_is_reusable_distinct_and_digest_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, manifest, spec = _dataset(tmp_path, shape=(40, 40, 40), cubic=True)
    extractor = _extractor(tmp_path, manifest, spec)
    universe = prepare_case_candidate_universe(extractor, case)
    volumes = extractor.canonical_volumes_for_case(case)
    raw_centers = pipeline_module.valid_patch_centers_mm(
        spec,
        pipeline_module.intersect_modality_foreground_support_masks(volumes, spec=spec),
    )
    arguments = {
        "epoch": 2,
        "bag_index": 17,
        "experiment_seed": 29,
        "target_count": 32,
        "candidate_pool_size": 512,
    }

    direct_plan = materialize_matching_plan(
        case=case,
        data_manifest_sha256=manifest.sha256,
        candidate_centers_mm=universe.candidate_centers.values,
        geometry=SlabGeometry.cubic(4.0),
        extraction_spec_sha256=spec.sha256,
        **arguments,
    )
    expected = PreparedCasePlan(
        plan=direct_plan,  # type: ignore[arg-type]
        candidate_count=universe.candidate_count,
        candidate_centers_sha256=universe.candidate_centers_sha256,
        volume_digests=universe.volume_digests,
    )

    def forbidden_reprepare(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("cached plan materialization must not rebuild the shared mask")

    monkeypatch.setattr(
        pipeline_module,
        "intersect_modality_foreground_support_masks",
        forbidden_reprepare,
    )
    prepared = materialize_case_matching_plan_record(
        extractor,
        case,
        universe,
        **arguments,
    )
    different_bag = materialize_case_matching_plan_record(
        extractor,
        case,
        universe,
        **{**arguments, "bag_index": 18},
    )
    different_epoch = materialize_case_matching_plan_record(
        extractor,
        case,
        universe,
        **{**arguments, "epoch": 3},
    )

    assert prepared.plan.sha256 == direct_plan.sha256
    assert prepared.sha256 == expected.sha256
    assert prepared.candidate_centers_sha256 == pipeline_module._candidate_centers_sha256(
        raw_centers
    )
    assert prepared.plan.sha256 != different_bag.plan.sha256
    assert prepared.plan.sha256 != different_epoch.plan.sha256
    assert not universe.candidate_centers.values.flags.writeable


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("data_manifest_sha256", "data manifest"),
        ("extraction_spec_sha256", "extraction specification"),
        ("geometry_sha256", "patch geometry"),
    ),
)
def test_prepared_candidate_universe_fails_closed_on_provenance_mismatch(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    case, manifest, spec = _dataset(tmp_path)
    extractor = _extractor(tmp_path, manifest, spec)
    universe = prepare_case_candidate_universe(extractor, case)
    mismatched = replace(universe, **{field: _digest(f"wrong-{field}")})

    with pytest.raises(DataPipelineError, match=message):
        materialize_case_matching_plan_record(
            extractor,
            case,
            mismatched,
            epoch=0,
            bag_index=0,
            experiment_seed=0,
            target_count=8,
            candidate_pool_size=128,
        )


def test_prepared_candidate_universe_fails_closed_on_case_mismatch(tmp_path: Path) -> None:
    case, manifest, spec = _dataset(tmp_path)
    extractor = _extractor(tmp_path, manifest, spec)
    universe = prepare_case_candidate_universe(extractor, case)
    other_case = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id="BraTS-MET-00002-000",
        files=case.files,
    )
    mismatched = replace(universe, case=other_case)

    with pytest.raises(DataPipelineError, match="exact manifest case"):
        materialize_case_matching_plan_record(
            extractor,
            case,
            mismatched,
            epoch=0,
            bag_index=0,
            experiment_seed=0,
            target_count=8,
            candidate_pool_size=128,
        )


def test_case_helper_rejects_case_not_exactly_in_bound_manifest(tmp_path: Path) -> None:
    case, manifest, spec = _dataset(tmp_path)
    extractor = _extractor(tmp_path, manifest, spec)
    changed = CaseRecord.create(
        source=case.source,
        release=case.release,
        case_id=case.case_id,
        files=tuple(
            FileRecord(record.modality, record.path, _digest("changed"))
            if record.modality == "t1n"
            else record
            for record in case.files
        ),
    )

    with pytest.raises(DataPipelineError, match="exactly match"):
        prepare_case_matching_plan(
            extractor,
            changed,
            epoch=0,
            bag_index=0,
            experiment_seed=0,
            target_count=8,
        )
