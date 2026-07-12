from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest
import torch

from simple_brats.config import MODALITIES
from simple_brats.data.extraction import (
    CanonicalVolume,
    ExtractionSpec,
    NormalizationStats,
    extract_patch,
    valid_patch_centers_mm,
)
from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.data.pipeline import (
    CanonicalVolumeDigest,
    PreparedCaseCandidateUniverse,
    _candidate_centers_sha256,
)
from simple_brats.data.plan_factory import CanonicalCandidateCenters
from simple_brats.data.scheduled_cache import (
    DEFAULT_GPU_CACHE_BYTES,
    DEFAULT_PREFETCH_DEPTH,
    DEFAULT_PREFETCH_REFILL_BATCH_SIZE,
    DEFAULT_PREFETCH_WORKERS,
    OptimizedRuntimeConfig,
    ScheduleKeyedPrefetcher,
    batched_patch_table_from_prepared_volumes,
)

IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_optimized_runtime_defaults_are_explicit_and_recorded() -> None:
    config = OptimizedRuntimeConfig()

    assert config.prefetch_workers == DEFAULT_PREFETCH_WORKERS == 8
    assert config.prefetch_depth == DEFAULT_PREFETCH_DEPTH == 16
    assert config.prefetch_refill_batch_size == DEFAULT_PREFETCH_REFILL_BATCH_SIZE == 4
    assert config.prefetch_refill_low_watermark == 12
    assert config.to_dict()["prefetch_refill_low_watermark"] == 12
    assert config.gpu_cache_bytes == DEFAULT_GPU_CACHE_BYTES
    assert config.to_dict()["selection_authority"] == "external_absolute_step_schedule_only"
    assert config.to_dict()["worker_cuda_access"] is False
    assert "lookahead_ready" in str(config.to_dict()["startup_prefetch_barrier"])
    assert config.to_dict()["failure_policy"].startswith("raise_for_exact_scheduled_key")


def test_schedule_keyed_prefetch_never_substitutes_failed_key() -> None:
    calls: list[int] = []

    def load(key: int) -> str:
        calls.append(key)
        if key == 2:
            raise RuntimeError("scheduled failure")
        return f"case-{key}"

    prefetch = ScheduleKeyedPrefetcher(load, workers=2, depth=3)
    try:
        assert prefetch.prime((1, 2, 3, 4)) == (1, 2, 3)
        assert prefetch.get(1) == "case-1"
        with pytest.raises(RuntimeError, match="scheduled failure"):
            prefetch.get(2)
        assert prefetch.get(3) == "case-3"
        assert 4 not in calls
        assert prefetch.submitted_count == 3
        assert prefetch.consumed_count == 3
    finally:
        prefetch.close(cancel_pending=True)


def test_schedule_keyed_prefetch_completion_order_cannot_change_lookup() -> None:
    release_first = threading.Event()
    second_finished = threading.Event()

    def load(key: str) -> str:
        if key == "first":
            release_first.wait(timeout=5)
        else:
            second_finished.set()
        return key.upper()

    prefetch = ScheduleKeyedPrefetcher(load, workers=2, depth=2)
    try:
        prefetch.prime(("first", "second"))
        assert second_finished.wait(timeout=5)
        # Although the second future completed first, exact keyed consumption
        # still returns only the requested scheduled key.
        assert prefetch.get("second") == "SECOND"
        release_first.set()
        assert prefetch.get("first") == "FIRST"
    finally:
        release_first.set()
        prefetch.close(cancel_pending=True)


def test_readiness_barrier_retains_exact_keys_for_ready_consumption() -> None:
    prefetch = ScheduleKeyedPrefetcher(lambda key: f"case-{key}", workers=2, depth=3)
    try:
        assert prefetch.prime((4, 7, 9)) == (4, 7, 9)

        assert prefetch.wait_pending() == (4, 7, 9)

        assert prefetch.pending_keys == (4, 7, 9)
        assert prefetch.consumed_count == 0
        assert prefetch.get(4) == "case-4"
        assert prefetch.get(7) == "case-7"
        assert prefetch.get(9) == "case-9"
        stats = prefetch.to_dict()
        assert stats["readiness_barrier_count"] == 1
        assert stats["readiness_barrier_key_count"] == 3
        assert stats["ready_hit_count"] == 3
        assert stats["stall_count"] == 0
        assert stats["ready_pending_count"] == 0
        assert stats["failed_pending_count"] == 0
        assert stats["running_pending_count"] == 0
    finally:
        prefetch.close(cancel_pending=True)


def test_prefetch_refills_only_at_low_watermark_and_fills_to_depth() -> None:
    prefetch = ScheduleKeyedPrefetcher(
        lambda key: key,
        workers=8,
        depth=16,
        refill_batch_size=4,
    )
    try:
        assert prefetch.refill_low_watermark == 12
        for start in range(0, 16, 4):
            assert prefetch.prime(range(start, 16)) == tuple(range(start, start + 4))
        prefetch.wait_pending()

        for key in range(3):
            assert prefetch.get(key) == key
            assert prefetch.prime(range(16, 20)) == ()
        assert len(prefetch.pending_keys) == 13

        assert prefetch.get(3) == 3
        assert len(prefetch.pending_keys) == 12
        assert prefetch.prime(range(16, 20)) == (16, 17, 18, 19)
        assert len(prefetch.pending_keys) == 16
        prefetch.wait_pending()
        stats = prefetch.to_dict()
        assert stats["ready_pending_count"] == 16
        assert stats["failed_pending_count"] == 0
        assert stats["running_pending_count"] == 0
        assert stats["ready_prefix_count"] == 16
    finally:
        prefetch.close(cancel_pending=True)


def test_overlapping_refill_batches_are_bounded_by_executor_workers() -> None:
    release = threading.Event()
    all_eight_active = threading.Event()
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def load(key: int) -> int:
        nonlocal active, peak_active
        if key >= 16:
            with lock:
                active += 1
                peak_active = max(peak_active, active)
                if active == 8:
                    all_eight_active.set()
            release.wait(timeout=5)
            with lock:
                active -= 1
        return key

    prefetch = ScheduleKeyedPrefetcher(
        load,
        workers=8,
        depth=16,
        refill_batch_size=4,
    )
    try:
        for start in range(0, 16, 4):
            prefetch.prime(range(start, 16))
        prefetch.wait_pending()
        for key in range(4):
            assert prefetch.get(key) == key
        assert prefetch.prime(range(16, 20)) == (16, 17, 18, 19)
        for key in range(4, 8):
            assert prefetch.get(key) == key
        assert prefetch.prime(range(20, 24)) == (20, 21, 22, 23)
        assert all_eight_active.wait(timeout=5)

        stats = prefetch.to_dict()
        assert stats["running_pending_count"] == 8
        assert stats["ready_prefix_count"] == 8
        assert peak_active == 8
    finally:
        release.set()
        prefetch.close(cancel_pending=True)


def test_pending_failure_is_not_reported_as_ready() -> None:
    second_finished = threading.Event()

    def fail(key: int) -> int:
        if key == 7:
            raise RuntimeError(f"failed-{key}")
        second_finished.set()
        return key

    prefetch = ScheduleKeyedPrefetcher(fail, workers=1, depth=2)
    try:
        assert prefetch.prime((7, 8)) == (7, 8)
        with pytest.raises(RuntimeError, match="failed-7"):
            prefetch.wait_pending()
        assert second_finished.wait(timeout=5)
        stats = prefetch.to_dict()
        assert stats["pending_count"] == 2
        assert stats["ready_pending_count"] == 1
        assert stats["failed_pending_count"] == 1
        assert stats["running_pending_count"] == 0
        assert stats["ready_prefix_count"] == 0
    finally:
        prefetch.close(cancel_pending=True)


def test_discard_pending_removes_stale_lookahead_without_loading_replacements() -> None:
    release = threading.Event()

    def load(key: int) -> int:
        release.wait(timeout=5)
        return key

    prefetch = ScheduleKeyedPrefetcher(load, workers=1, depth=3)
    try:
        prefetch.prime((10, 11, 12))
        assert prefetch.discard_pending() == (10, 11, 12)
        assert prefetch.pending_keys == ()
        release.set()
        assert prefetch.get(20) == 20
    finally:
        release.set()
        prefetch.close(cancel_pending=True)


def _prepared_case() -> tuple[
    ExtractionSpec,
    CaseRecord,
    dict[str, CanonicalVolume],
    PreparedCaseCandidateUniverse,
]:
    spec = ExtractionSpec(
        canonical_shape=(8, 8, 8),
        canonical_affine=IDENTITY,
        patch_source_shape=(4, 4, 4),
        patch_physical_extent_mm=(4.0, 4.0, 4.0),
        model_visible_shape=(8, 8, 8),
    )
    files = tuple(
        FileRecord(modality, f"case/{modality}.nii.gz", _digest(f"raw-{modality}"))
        for modality in MODALITIES
    )
    case = CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id="BraTS-MET-00001-000",
        files=files,
    )
    mask = np.ones(spec.canonical_shape, dtype=np.bool_)
    grid = np.arange(np.prod(spec.canonical_shape), dtype=np.float32).reshape(
        spec.canonical_shape
    )
    volumes: dict[str, CanonicalVolume] = {}
    digests: list[CanonicalVolumeDigest] = []
    for modality_id, modality in enumerate(MODALITIES):
        data = grid + modality_id * 1000.0
        volume = CanonicalVolume(
            data=data,
            valid_support_mask=mask,
            foreground_mask=mask,
            affine=np.asarray(IDENTITY),
            extraction_spec_sha256=spec.sha256,
            voxel_content_sha256=_digest(f"canonical-{modality}"),
            normalized_sha256=_digest(f"normalized-{modality}"),
            normalization_stats=NormalizationStats(
                foreground_voxels=int(mask.sum()), mean=0.0, std=1.0
            ),
        )
        volumes[modality] = volume
        digests.append(
            CanonicalVolumeDigest(
                modality=modality,
                raw_file_sha256=files[modality_id].sha256,
                canonical_voxel_sha256=volume.voxel_content_sha256,
                normalized_voxel_sha256=volume.normalized_sha256,
            )
        )
    centers = CanonicalCandidateCenters(valid_patch_centers_mm(spec, mask))
    universe = PreparedCaseCandidateUniverse(
        case=case,
        data_manifest_sha256=_digest("manifest"),
        extraction_spec_sha256=spec.sha256,
        geometry_sha256=_digest("geometry"),
        candidate_centers=centers,
        candidate_count=len(centers),
        candidate_centers_sha256=_candidate_centers_sha256(centers.values),
        volume_digests=tuple(digests),
    )
    return spec, case, volumes, universe


def test_batched_device_extraction_matches_reference_cpu_axis_and_values() -> None:
    spec, _, volumes, universe = _prepared_case()
    positions = (0, 17, len(universe.candidate_centers) - 1)
    centers = tuple(universe.candidate_centers.center(index) for index in positions)
    stacked = torch.stack(
        [torch.from_numpy(np.array(volumes[modality].data, copy=True)) for modality in MODALITIES]
    )

    actual = batched_patch_table_from_prepared_volumes(
        volumes=stacked,
        extraction_spec=spec,
        candidate_universe=universe,
        position_ids=positions,
        centers_mm=centers,
    )
    expected = torch.stack(
        [
            torch.stack(
                [
                    torch.from_numpy(
                        extract_patch(volumes[modality], center, spec=spec).data.copy()
                    )
                    for modality in MODALITIES
                ]
            )
            for center in centers
        ]
    )

    assert actual.shape == (3, 4, 8, 8, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_batched_extraction_rejects_center_position_mismatch() -> None:
    spec, _, volumes, universe = _prepared_case()
    stacked = torch.stack(
        [torch.from_numpy(np.array(volumes[modality].data, copy=True)) for modality in MODALITIES]
    )
    wrong_center = universe.candidate_centers.center(1)

    with pytest.raises(Exception, match="does not address"):
        batched_patch_table_from_prepared_volumes(
            volumes=stacked,
            extraction_spec=spec,
            candidate_universe=universe,
            position_ids=(0,),
            centers_mm=(wrong_center,),
        )


def test_optimized_config_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="gpu_cache_bytes"):
        OptimizedRuntimeConfig(gpu_cache_bytes=0)
