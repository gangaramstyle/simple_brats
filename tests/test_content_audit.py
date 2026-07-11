from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_brats.config import MODALITIES, ExperimentConfig, PatchConfig
from simple_brats.data.case_grids import (
    CaseGridManifest,
    CaseGridRecord,
    ExtractionPolicy,
    SpatialGrid,
    derive_prepared_grid,
)
from simple_brats.data.content_audit import ContentAuditError, run_content_audit
from simple_brats.data.extraction import CanonicalVolume, ExtractionSpec, NormalizationStats
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.splits import (
    SplitFraction,
    SplitManifest,
    SubjectAssignment,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(number: int, *, duplicate_t1n_raw: str | None = None) -> CaseRecord:
    case_id = f"BraTS-MET-{number:05d}-001"
    files = []
    for modality in MODALITIES:
        raw_digest = (
            duplicate_t1n_raw
            if modality == "t1n" and duplicate_t1n_raw is not None
            else _digest(f"{case_id}:{modality}:raw")
        )
        files.append(
            FileRecord(
                modality=modality,
                path=f"{case_id}/{case_id}-{modality}.nii.gz",
                sha256=raw_digest,
            )
        )
    # Labels are deliberately present so the test can prove they never enter
    # a case content record or affect duplicate reporting.
    files.append(
        FileRecord(
            modality="seg",
            path=f"{case_id}/{case_id}-seg.nii.gz",
            sha256=_digest(f"{case_id}:seg"),
        )
    )
    return CaseRecord.create(
        source="BraTS-MET",
        release="test",
        case_id=case_id,
        files=files,
    )


def _grid(
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
) -> SpatialGrid:
    return SpatialGrid(
        shape=shape,
        affine=(
            (1.0, 0.0, 0.0, origin[0]),
            (0.0, 1.0, 0.0, origin[1]),
            (0.0, 0.0, 1.0, origin[2]),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _case_grid_manifest(manifest: DatasetManifest) -> CaseGridManifest:
    policy = ExtractionPolicy()
    records = []
    for index, case in enumerate(manifest.cases):
        # Deliberately heterogeneous shape and scanner-world origin: the audit
        # must derive and use a different ExtractionSpec for every case.
        native = _grid(
            (4 + index, 4, 4 + index),
            (float(index * 11), float(-index * 3), float(index * 7)),
        )
        prepared = derive_prepared_grid(native, policy)
        spec = policy.extraction_spec(prepared)
        records.append(
            CaseGridRecord(
                data_manifest_sha256=manifest.sha256,
                case=case,
                declared_spatial_units=("mm", "mm", "mm", "mm"),
                extraction_policy_sha256=policy.sha256,
                native_grid=native,
                modality_native_grids=(native, native, native, native),
                prepared_grid=prepared,
                extraction_spec_sha256=spec.sha256,
            )
        )
    return CaseGridManifest(
        data_manifest_sha256=manifest.sha256,
        policy=policy,
        records=tuple(records),
    )


class _FakeExtractor:
    def __init__(
        self,
        manifest: DatasetManifest,
        spec: ExtractionSpec,
        *,
        expected_case: CaseRecord,
        duplicate_cases: frozenset[str],
        calls: list[str],
        clear_calls: list[str],
    ) -> None:
        self.data_manifest_sha256 = manifest.sha256
        self.extraction_spec = spec
        self.extraction_spec_sha256 = spec.sha256
        self.expected_case = expected_case
        self.duplicate_cases = duplicate_cases
        self.calls = calls
        self.clear_calls = clear_calls

    def canonical_volumes_for_case(
        self,
        case: CaseRecord,
    ) -> dict[str, CanonicalVolume]:
        assert case == self.expected_case
        self.calls.append(case.case_id)
        result = {}
        for modality in MODALITIES:
            duplicate = modality == "t1n" and case.case_id in self.duplicate_cases
            canonical = (
                _digest("shared:t1n:canonical")
                if duplicate
                else _digest(f"{case.case_id}:{modality}:canonical")
            )
            normalized = (
                _digest("shared:t1n:normalized")
                if duplicate
                else _digest(f"{case.case_id}:{modality}:normalized")
            )
            shape = self.extraction_spec.canonical_shape
            result[modality] = CanonicalVolume(
                data=np.ones(shape, dtype=np.float32),
                valid_support_mask=np.ones(shape, dtype=np.bool_),
                foreground_mask=np.ones(shape, dtype=np.bool_),
                affine=np.asarray(self.extraction_spec.canonical_affine),
                extraction_spec_sha256=self.extraction_spec_sha256,
                voxel_content_sha256=canonical,
                normalized_sha256=normalized,
                normalization_stats=NormalizationStats(
                    foreground_voxels=int(np.prod(shape)),
                    mean=float(len(case.case_id)),
                    std=1.0,
                ),
            )
        return result

    def clear_cache(self) -> None:
        self.clear_calls.append(self.expected_case.case_id)


class _FakeExtractorFactory:
    def __init__(
        self,
        manifest: DatasetManifest,
        catalog: CaseGridManifest,
        experiment_config: ExperimentConfig,
        *,
        duplicate_cases: tuple[str, str],
    ) -> None:
        self.manifest = manifest
        self.catalog = catalog
        self.experiment_config = experiment_config
        self.duplicate_cases = frozenset(duplicate_cases)
        self.constructed: list[tuple[str, str]] = []
        self.calls: list[str] = []
        self.clear_calls: list[str] = []

    def __call__(
        self,
        *,
        case: CaseRecord,
        extraction_spec: ExtractionSpec,
    ) -> _FakeExtractor:
        assert extraction_spec == self.catalog.extraction_spec_for_case(
            case,
            patch_config=self.experiment_config.patch,
        )
        self.constructed.append((case.case_id, extraction_spec.sha256))
        return _FakeExtractor(
            self.manifest,
            extraction_spec,
            expected_case=case,
            duplicate_cases=self.duplicate_cases,
            calls=self.calls,
            clear_calls=self.clear_calls,
        )


def _inputs() -> tuple[
    DatasetManifest,
    SplitManifest,
    CaseGridManifest,
    ExperimentConfig,
    _FakeExtractorFactory,
]:
    shared_raw = _digest("shared:t1n:raw")
    first = _case(1, duplicate_t1n_raw=shared_raw)
    second = _case(2, duplicate_t1n_raw=shared_raw)
    manifest = DatasetManifest(cases=(second, first))
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=0,
        fractions=(
            SplitFraction("train", "0.5"),
            SplitFraction("test", "0.5"),
        ),
        assignments=(
            SubjectAssignment(first.subject_id, "train"),
            SubjectAssignment(second.subject_id, "test"),
        ),
    )
    catalog = _case_grid_manifest(manifest)
    experiment_config = ExperimentConfig()
    factory = _FakeExtractorFactory(
        manifest,
        catalog,
        experiment_config,
        duplicate_cases=(first.case_id, second.case_id),
    )
    return manifest, split, catalog, experiment_config, factory


def test_content_audit_resumes_heterogeneous_case_grids_and_reports_duplicates(
    tmp_path: Path,
) -> None:
    manifest, split, catalog, experiment_config, factory = _inputs()
    state = tmp_path / "state"
    output = tmp_path / "audit.json"
    launch_sha = "1" * 40

    partial = run_content_audit(
        manifest=manifest,
        split=split,
        case_grid_manifest=catalog,
        experiment_config=experiment_config,
        extractor_factory=factory,
        launch_sha=launch_sha,
        state_dir=state,
        output_path=output,
        max_new_cases=1,
    )
    assert partial.complete is False
    assert partial.completed_cases == 1
    assert partial.newly_completed_cases == 1
    assert not output.exists()

    complete = run_content_audit(
        manifest=manifest,
        split=split,
        case_grid_manifest=catalog,
        experiment_config=experiment_config,
        extractor_factory=factory,
        launch_sha=launch_sha,
        state_dir=state,
        output_path=output,
    )
    assert complete.complete is True
    assert complete.completed_cases == 2
    assert complete.newly_completed_cases == 1
    assert complete.output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert factory.calls == [case.case_id for case in manifest.cases]
    assert factory.clear_calls == factory.calls
    assert factory.constructed == [
        (
            case.case_id,
            catalog.extraction_spec_for_case(
                case,
                patch_config=experiment_config.patch,
            ).sha256,
        )
        for case in manifest.cases
    ]

    state_header = json.loads((state / "state.json").read_bytes())
    assert state_header["case_grid_manifest_sha256"] == catalog.sha256
    assert state_header["case_grid_policy_sha256"] == catalog.policy.sha256
    assert state_header["experiment_config_sha256"] == experiment_config.sha256
    assert state_header["runtime_extraction_policy_sha256"] == (
        catalog.policy.for_patch_config(experiment_config.patch).sha256
    )
    assert state_header["runtime_extraction_policy_sha256"] != catalog.policy.sha256
    assert state_header["patch_config"] == {
        "footprint_mm": 4.0,
        "tensor_shape": [16, 16, 16],
        "thin_mm": 4.0,
    }
    shards = [
        json.loads(path.read_bytes())
        for path in sorted((state / "cases").glob("*.json"))
    ]
    assert {shard["case_grid_record_sha256"] for shard in shards} == {
        catalog.record_for_case(case).sha256 for case in manifest.cases
    }
    assert {shard["extraction_spec_sha256"] for shard in shards} == {
        catalog.extraction_spec_for_case(
            case,
            patch_config=experiment_config.patch,
        ).sha256
        for case in manifest.cases
    }

    report = json.loads(output.read_bytes())
    assert output.read_bytes() == json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert report["case_grid_manifest_sha256"] == catalog.sha256
    assert report["case_grid_policy_sha256"] == catalog.policy.sha256
    assert report["experiment_config_sha256"] == experiment_config.sha256
    assert report["runtime_extraction_policy_sha256"] == (
        catalog.policy.for_patch_config(experiment_config.patch).sha256
    )
    assert report["counts"]["distinct_extraction_specs"] == 2
    assert report["counts"]["cross_subject_duplicate_components"] == 3
    assert report["counts"]["cross_split_duplicate_components"] == 3
    assert {
        component["representation"]
        for component in report["duplicates"]["cross_split_components"]
    } == {"raw_file", "canonical_voxel", "normalized_voxel"}
    assert all(
        len(component["subjects"]) == 2
        for component in report["duplicates"]["cross_subject_components"]
    )
    for case_record, case in zip(report["cases"], manifest.cases, strict=True):
        assert (
            case_record["case_grid_record_sha256"]
            == catalog.record_for_case(case).sha256
        )
        assert (
            case_record["extraction_spec_sha256"]
            == catalog.extraction_spec_for_case(
                case,
                patch_config=experiment_config.patch,
            ).sha256
        )
        assert all(item["modality"] != "seg" for item in case_record["modalities"])
    assert _digest(f"{manifest.cases[0].case_id}:seg") not in output.read_text()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_content_audit(
            manifest=manifest,
            split=split,
            case_grid_manifest=catalog,
            experiment_config=experiment_config,
            extractor_factory=factory,
            launch_sha=launch_sha,
            state_dir=state,
            output_path=output,
        )


def test_content_audit_rejects_resume_with_different_git_pin(tmp_path: Path) -> None:
    manifest, split, catalog, experiment_config, factory = _inputs()
    state = tmp_path / "state"
    run_content_audit(
        manifest=manifest,
        split=split,
        case_grid_manifest=catalog,
        experiment_config=experiment_config,
        extractor_factory=factory,
        launch_sha="1" * 40,
        state_dir=state,
        output_path=tmp_path / "first.json",
        max_new_cases=1,
    )

    with pytest.raises(ContentAuditError, match="different provenance"):
        run_content_audit(
            manifest=manifest,
            split=split,
            case_grid_manifest=catalog,
            experiment_config=experiment_config,
            extractor_factory=factory,
            launch_sha="2" * 40,
            state_dir=state,
            output_path=tmp_path / "second.json",
            max_new_cases=1,
        )


def test_content_audit_rejects_resume_with_different_patch_policy(tmp_path: Path) -> None:
    manifest, split, catalog, experiment_config, factory = _inputs()
    state = tmp_path / "state"
    run_content_audit(
        manifest=manifest,
        split=split,
        case_grid_manifest=catalog,
        experiment_config=experiment_config,
        extractor_factory=factory,
        launch_sha="1" * 40,
        state_dir=state,
        output_path=tmp_path / "first.json",
        max_new_cases=1,
    )
    eight_mm_config = ExperimentConfig(
        patch=PatchConfig(
            footprint_mm=8.0,
            thin_mm=8.0,
            tensor_shape=(16, 16, 16),
        )
    )

    with pytest.raises(ContentAuditError, match="different provenance"):
        run_content_audit(
            manifest=manifest,
            split=split,
            case_grid_manifest=catalog,
            experiment_config=eight_mm_config,
            extractor_factory=factory,
            launch_sha="1" * 40,
            state_dir=state,
            output_path=tmp_path / "second.json",
            max_new_cases=1,
        )


def test_content_audit_rejects_noncanonical_resume_shard(tmp_path: Path) -> None:
    manifest, split, catalog, experiment_config, factory = _inputs()
    state = tmp_path / "state"
    run_content_audit(
        manifest=manifest,
        split=split,
        case_grid_manifest=catalog,
        experiment_config=experiment_config,
        extractor_factory=factory,
        launch_sha="1" * 40,
        state_dir=state,
        output_path=tmp_path / "audit.json",
        max_new_cases=1,
    )
    shard = next((state / "cases").glob("*.json"))
    shard.write_bytes(shard.read_bytes() + b"\n")

    with pytest.raises(ContentAuditError, match="canonical JSON"):
        run_content_audit(
            manifest=manifest,
            split=split,
            case_grid_manifest=catalog,
            experiment_config=experiment_config,
            extractor_factory=factory,
            launch_sha="1" * 40,
            state_dir=state,
            output_path=tmp_path / "audit.json",
        )
