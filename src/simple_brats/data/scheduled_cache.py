"""Schedule-keyed prefetch and byte-bounded GPU patch extraction.

The cache in this module is deliberately incapable of selecting data.  Callers
must provide exact keys produced by their experiment schedule.  Background
workers may prepare those keys out of order, but a consumer can retrieve only
the key it explicitly requests; failures are propagated without substitution.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from simple_brats.config import MODALITIES

from .extraction import CanonicalVolume, ExtractionSpec
from .manifest import CaseRecord
from .pipeline import PreparedCaseCandidateUniverse

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

OPTIMIZED_RUNTIME_SCHEMA = "simple-brats.schedule-keyed-optimized-runtime"
OPTIMIZED_RUNTIME_SCHEMA_VERSION = 1
DEFAULT_PREFETCH_WORKERS = 8
DEFAULT_PREFETCH_DEPTH = 16
DEFAULT_GPU_CACHE_BYTES = 4 * 1024**3


class ScheduledCacheError(RuntimeError):
    """The optimized runtime could not preserve its schedule/cache contract."""


@dataclass(frozen=True, slots=True)
class OptimizedRuntimeConfig:
    """Explicit, provenance-ready optimized data runtime configuration."""

    prefetch_workers: int = DEFAULT_PREFETCH_WORKERS
    prefetch_depth: int = DEFAULT_PREFETCH_DEPTH
    gpu_cache_bytes: int = DEFAULT_GPU_CACHE_BYTES
    batched_gpu_extraction: bool = True
    schema: str = OPTIMIZED_RUNTIME_SCHEMA
    schema_version: int = OPTIMIZED_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("prefetch_workers", "prefetch_depth", "gpu_cache_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.batched_gpu_extraction, bool):
            raise TypeError("batched_gpu_extraction must be boolean")
        if self.schema != OPTIMIZED_RUNTIME_SCHEMA or (
            self.schema_version != OPTIMIZED_RUNTIME_SCHEMA_VERSION
        ):
            raise ValueError("unsupported optimized-runtime schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "prefetch_workers": self.prefetch_workers,
            "prefetch_depth": self.prefetch_depth,
            "gpu_cache_bytes": self.gpu_cache_bytes,
            "batched_gpu_extraction": self.batched_gpu_extraction,
            "selection_authority": "external_absolute_step_schedule_only",
            "worker_cuda_access": False,
            "startup_prefetch_barrier": (
                "all_scheduled_lookahead_ready_before_first_optimizer_step"
            ),
            "failure_policy": "raise_for_exact_scheduled_key_without_replacement",
            "gpu_eviction_policy": "byte_bounded_lru_latency_only",
            "patch_extraction": (
                "batched_integer_crop_then_trilinear_gpu_resize"
                if self.batched_gpu_extraction
                else "reference_cpu_per_patch"
            ),
        }


class ScheduleKeyedPrefetcher(Generic[K, V]):
    """Bounded single-flight preparation for caller-supplied schedule keys.

    ``prime`` preserves the supplied key order when submitting work.  Futures
    are keyed, so worker completion order cannot affect consumption.  The class
    has no sampling API and never invents a replacement key after failure.
    """

    def __init__(
        self,
        loader: Callable[[K], V],
        *,
        workers: int = DEFAULT_PREFETCH_WORKERS,
        depth: int = DEFAULT_PREFETCH_DEPTH,
        thread_name_prefix: str = "simple-brats-prefetch",
    ) -> None:
        if not callable(loader):
            raise TypeError("loader must be callable")
        for value, name in ((workers, "workers"), (depth, "depth")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._loader = loader
        self.workers = workers
        self.depth = depth
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._futures: OrderedDict[K, Future[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False
        self.submitted_count = 0
        self.consumed_count = 0
        self.ready_hit_count = 0
        self.stall_count = 0
        self.wait_seconds = 0.0
        self.discarded_count = 0
        self.readiness_barrier_count = 0
        self.readiness_barrier_key_count = 0
        self.readiness_wait_seconds = 0.0

    def _require_open(self) -> None:
        if self._closed:
            raise ScheduledCacheError("prefetcher is closed")

    def prime(self, keys: Iterable[K]) -> tuple[K, ...]:
        """Submit the first unseen keys that fit the bounded future table."""

        submitted: list[K] = []
        with self._lock:
            self._require_open()
            for key in keys:
                if key in self._futures:
                    continue
                if len(self._futures) >= self.depth:
                    break
                self._futures[key] = self._executor.submit(self._loader, key)
                self.submitted_count += 1
                submitted.append(key)
        return tuple(submitted)

    def get(self, key: K) -> V:
        """Return exactly ``key`` or re-raise its preparation exception."""

        with self._lock:
            self._require_open()
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(self._loader, key)
                self._futures[key] = future
                self.submitted_count += 1
            ready = future.done()
        start = time.perf_counter()
        try:
            value = future.result()
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._futures.pop(key, None)
                self.consumed_count += 1
                self.wait_seconds += elapsed
                if ready:
                    self.ready_hit_count += 1
                else:
                    self.stall_count += 1
        return value

    @property
    def pending_keys(self) -> tuple[K, ...]:
        with self._lock:
            return tuple(self._futures)

    def wait_pending(self) -> tuple[K, ...]:
        """Wait for exact submitted lookahead without consuming any key.

        This startup barrier lets compilation and calibration overlap cold
        loads, then retains the completed futures for normal keyed consumption.
        It never invents, removes, or reorders a schedule key.
        """

        with self._lock:
            self._require_open()
            items = tuple(self._futures.items())
        start = time.perf_counter()
        try:
            for _, future in items:
                future.result()
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self.readiness_barrier_count += 1
                self.readiness_barrier_key_count += len(items)
                self.readiness_wait_seconds += elapsed
        return tuple(key for key, _ in items)

    def discard_pending(self) -> tuple[K, ...]:
        """Drop unconsumed lookahead without changing any consumed schedule key."""

        with self._lock:
            self._require_open()
            items = tuple(self._futures.items())
            self._futures.clear()
        for _, future in items:
            future.cancel()
        with self._lock:
            self.discarded_count += len(items)
        return tuple(key for key, _ in items)

    def to_dict(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "workers": self.workers,
                "depth": self.depth,
                "pending_count": len(self._futures),
                "submitted_count": self.submitted_count,
                "consumed_count": self.consumed_count,
                "ready_hit_count": self.ready_hit_count,
                "stall_count": self.stall_count,
                "wait_seconds": self.wait_seconds,
                "discarded_count": self.discarded_count,
                "readiness_barrier_count": self.readiness_barrier_count,
                "readiness_barrier_key_count": self.readiness_barrier_key_count,
                "readiness_wait_seconds": self.readiness_wait_seconds,
            }

    def close(self, *, cancel_pending: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = tuple(self._futures.values())
            if cancel_pending:
                self.discarded_count += len(futures)
            self._futures.clear()
        if cancel_pending:
            for future in futures:
                future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)

    def __enter__(self) -> ScheduleKeyedPrefetcher[K, V]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class GpuPreparedCase:
    """Four verified normalized volumes resident on one device."""

    case_key: tuple[str, str, str]
    extraction_spec_sha256: str
    volumes: Tensor
    nbytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.volumes, Tensor)
            or self.volumes.ndim != 4
            or self.volumes.shape[0] != len(MODALITIES)
            or self.volumes.dtype != torch.float32
            or self.volumes.device.type != "cuda"
        ):
            raise ScheduledCacheError(
                "GPU prepared volumes must be CUDA float32 with shape [4, X, Y, Z]"
            )
        expected = self.volumes.numel() * self.volumes.element_size()
        if self.nbytes != expected:
            raise ScheduledCacheError("GPU prepared-case byte count is inconsistent")


class ByteBoundedGpuCaseCache:
    """Main-thread CUDA cache whose LRU affects latency, never sample choice."""

    def __init__(self, *, byte_budget: int = DEFAULT_GPU_CACHE_BYTES) -> None:
        if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget <= 0:
            raise ValueError("byte_budget must be a positive integer")
        self.byte_budget = byte_budget
        self._entries: OrderedDict[tuple[tuple[str, str, str], str], GpuPreparedCase] = (
            OrderedDict()
        )
        self.resident_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.peak_resident_bytes = 0

    @staticmethod
    def _key(
        case: CaseRecord, extraction_spec_sha256: str
    ) -> tuple[tuple[str, str, str], str]:
        return case.key, extraction_spec_sha256

    def get_or_upload(
        self,
        *,
        case: CaseRecord,
        extraction_spec: ExtractionSpec,
        canonical_volumes: Mapping[str, CanonicalVolume],
        candidate_universe: PreparedCaseCandidateUniverse,
        device: torch.device,
    ) -> GpuPreparedCase:
        if device.type != "cuda":
            raise ScheduledCacheError("optimized GPU case cache requires a CUDA device")
        key = self._key(case, extraction_spec.sha256)
        cached = self._entries.get(key)
        if cached is not None:
            self.hit_count += 1
            self._entries.move_to_end(key)
            return cached
        self.miss_count += 1
        if candidate_universe.case != case or (
            candidate_universe.extraction_spec_sha256 != extraction_spec.sha256
        ):
            raise ScheduledCacheError("candidate universe does not match GPU cache upload")
        if tuple(canonical_volumes) != MODALITIES:
            raise ScheduledCacheError("canonical volumes must use canonical modality order")
        arrays: list[Tensor] = []
        digests = {
            item.modality: item for item in candidate_universe.volume_digests
        }
        for modality in MODALITIES:
            volume = canonical_volumes[modality]
            if volume.extraction_spec_sha256 != extraction_spec.sha256:
                raise ScheduledCacheError("canonical volume extraction provenance differs")
            digest = digests[modality]
            if (
                volume.voxel_content_sha256 != digest.canonical_voxel_sha256
                or volume.normalized_sha256 != digest.normalized_voxel_sha256
            ):
                raise ScheduledCacheError(
                    "canonical volume bytes differ from the candidate-universe provenance"
                )
            arrays.append(torch.from_numpy(np.array(volume.data, copy=True, order="C")))
        host = torch.stack(arrays).to(dtype=torch.float32).contiguous()
        nbytes = host.numel() * host.element_size()
        if nbytes > self.byte_budget:
            raise ScheduledCacheError(
                f"one prepared case needs {nbytes} bytes, exceeding GPU cache budget "
                f"{self.byte_budget}"
            )
        while self._entries and self.resident_bytes + nbytes > self.byte_budget:
            _, evicted = self._entries.popitem(last=False)
            self.resident_bytes -= evicted.nbytes
            self.eviction_count += 1
        resident = GpuPreparedCase(
            case_key=case.key,
            extraction_spec_sha256=extraction_spec.sha256,
            volumes=host.to(device=device),
            nbytes=nbytes,
        )
        self._entries[key] = resident
        self.resident_bytes += nbytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return resident

    def to_dict(self) -> dict[str, int]:
        return {
            "byte_budget": self.byte_budget,
            "resident_bytes": self.resident_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "resident_case_count": len(self._entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
        }


def _plan_starts(
    *,
    centers_mm: np.ndarray,
    spec: ExtractionSpec,
) -> np.ndarray:
    centers = np.asarray(centers_mm, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ScheduledCacheError("planned centers must be a finite Nx3 array")
    affine = np.asarray(spec.canonical_affine, dtype=np.float64)
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-6, rtol=0):
        raise ScheduledCacheError("optimized extraction requires an axis-aligned canonical grid")
    voxel = (centers - affine[:3, 3]) / np.diag(linear)
    start_float = voxel - (np.asarray(spec.patch_source_shape, dtype=np.float64) - 1.0) / 2.0
    starts = np.rint(start_float).astype(np.int64)
    maximum = np.asarray(spec.canonical_shape) - np.asarray(spec.patch_source_shape)
    if not np.allclose(starts, start_float, atol=1e-6, rtol=0) or bool(
        ((starts < 0) | (starts > maximum)).any()
    ):
        raise ScheduledCacheError("planned center is off-lattice or outside the canonical grid")
    return starts


def batched_patch_table_from_prepared_volumes(
    *,
    volumes: Tensor,
    extraction_spec: ExtractionSpec,
    candidate_universe: PreparedCaseCandidateUniverse,
    position_ids: Iterable[int],
    centers_mm: Iterable[tuple[float, float, float]],
) -> Tensor:
    """Extract all modalities at planned locations in one device operation.

    Returns ``[positions, modalities, X, Y, Z]`` in the same prepared-grid axis
    order as the reference CPU extractor.  Production supplies CUDA volumes;
    allowing CPU tensors here supports exact reference/parity tests.
    """

    if (
        not isinstance(volumes, Tensor)
        or volumes.ndim != 4
        or volumes.shape[0] != len(MODALITIES)
        or volumes.dtype != torch.float32
    ):
        raise ScheduledCacheError(
            "prepared volumes must be float32 with shape [4, X, Y, Z]"
        )
    positions = tuple(position_ids)
    centers = tuple(centers_mm)
    if len(positions) != len(centers) or not positions:
        raise ScheduledCacheError("position IDs and centers must be equally non-empty")
    if len(set(positions)) != len(positions) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in positions
    ):
        raise ScheduledCacheError("planned position IDs must be unique integers")
    # Plan position IDs are the original indices into the immutable candidate
    # universe (they are intentionally not renumbered to 0..31).  Bind both the
    # index and exact center so a reordered or substituted plan cannot gather a
    # different crop.
    universe_centers = candidate_universe.candidate_centers
    for position_id, center in zip(positions, centers, strict=True):
        if not 0 <= position_id < len(universe_centers) or (
            universe_centers.center(position_id) != tuple(center)
        ):
            raise ScheduledCacheError(
                "plan position/center does not address the bound candidate universe"
            )
    starts_np = _plan_starts(
        centers_mm=np.asarray(centers, dtype=np.float64),
        spec=extraction_spec,
    )
    device = volumes.device
    starts = torch.as_tensor(starts_np, dtype=torch.long, device=device)
    sx, sy, sz = extraction_spec.patch_source_shape
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(sx, device=device),
            torch.arange(sy, device=device),
            torch.arange(sz, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    indices = starts[:, None, :] + offsets[None, :, :]
    position_count = len(positions)
    modality = torch.arange(len(MODALITIES), device=device)[None, :, None]
    x = indices[:, None, :, 0]
    y = indices[:, None, :, 1]
    z = indices[:, None, :, 2]
    crops = volumes[modality, x, y, z].reshape(
        position_count, len(MODALITIES), sx, sy, sz
    )
    # Match the reference path: canonical X/Y/Z -> Conv3d Z/Y/X, interpolate,
    # then restore canonical model-visible X/Y/Z axis order.
    resized = F.interpolate(
        crops.permute(0, 1, 4, 3, 2).reshape(
            position_count * len(MODALITIES), 1, sz, sy, sx
        ),
        size=(
            extraction_spec.model_visible_shape[2],
            extraction_spec.model_visible_shape[1],
            extraction_spec.model_visible_shape[0],
        ),
        mode="trilinear",
        align_corners=False,
    )[:, 0]
    result = resized.reshape(
        position_count,
        len(MODALITIES),
        extraction_spec.model_visible_shape[2],
        extraction_spec.model_visible_shape[1],
        extraction_spec.model_visible_shape[0],
    ).permute(0, 1, 4, 3, 2)
    if not bool(torch.isfinite(result).all()):
        raise ScheduledCacheError("batched GPU patch extraction produced non-finite values")
    return result.contiguous()


def batched_gpu_patches(
    *,
    gpu_case: GpuPreparedCase,
    extraction_spec: ExtractionSpec,
    candidate_universe: PreparedCaseCandidateUniverse,
    position_ids: Iterable[int],
    centers_mm: Iterable[tuple[float, float, float]],
) -> Tensor:
    """CUDA-only wrapper around the parity-testable batched extraction core."""

    return batched_patch_table_from_prepared_volumes(
        volumes=gpu_case.volumes,
        extraction_spec=extraction_spec,
        candidate_universe=candidate_universe,
        position_ids=position_ids,
        centers_mm=centers_mm,
    )


def assemble_batched_gpu_matching_batch(
    *,
    case: CaseRecord,
    plan: object,
    extractor: object,
    gpu_case: GpuPreparedCase,
    extraction_spec: ExtractionSpec,
    candidate_universe: PreparedCaseCandidateUniverse,
    data_manifest_sha256: str,
    plan_sha256: str,
    extraction_spec_sha256: str,
) -> object:
    """Build the exact matching batch from one coalesced GPU crop operation."""

    from simple_brats.sampling import MaterializedPatchPlan

    from .real_batches import assemble_matching_batch_from_patch_tables

    if not isinstance(plan, MaterializedPatchPlan):
        raise TypeError("plan must be a MaterializedPatchPlan")
    if gpu_case.case_key != case.key or (
        gpu_case.extraction_spec_sha256 != extraction_spec.sha256
    ):
        raise ScheduledCacheError("GPU prepared case differs from the scheduled plan case")
    targets = tuple(plan.targets)
    patches = batched_gpu_patches(
        gpu_case=gpu_case,
        extraction_spec=extraction_spec,
        candidate_universe=candidate_universe,
        position_ids=(target.position_id for target in targets),
        centers_mm=(target.center_mm for target in targets),
    )
    row_by_position = {
        target.position_id: row for row, target in enumerate(targets)
    }
    try:
        source_rows = torch.tensor(
            [row_by_position[item.position_id] for item in plan.sources],
            dtype=torch.long,
            device=patches.device,
        )
        source_modalities = torch.tensor(
            [item.modality_id for item in plan.sources],
            dtype=torch.long,
            device=patches.device,
        )
        target_rows = torch.tensor(
            [row_by_position[item.position_id] for item in targets],
            dtype=torch.long,
            device=patches.device,
        )
        target_modalities = torch.tensor(
            [item.modality_id for item in targets],
            dtype=torch.long,
            device=patches.device,
        )
    except KeyError as error:
        raise ScheduledCacheError("source position has no exact planned target row") from error
    source_table = patches[source_rows, source_modalities].unsqueeze(0)
    target_table = patches[target_rows, target_modalities].unsqueeze(0)
    return assemble_matching_batch_from_patch_tables(
        case,
        plan,
        extractor,  # type: ignore[arg-type]
        source_patches=source_table,
        target_patches=target_table,
        data_manifest_sha256=data_manifest_sha256,
        plan_sha256=plan_sha256,
        extraction_spec_sha256=extraction_spec_sha256,
    )


__all__ = [
    "DEFAULT_GPU_CACHE_BYTES",
    "DEFAULT_PREFETCH_DEPTH",
    "DEFAULT_PREFETCH_WORKERS",
    "OPTIMIZED_RUNTIME_SCHEMA",
    "OPTIMIZED_RUNTIME_SCHEMA_VERSION",
    "ByteBoundedGpuCaseCache",
    "GpuPreparedCase",
    "OptimizedRuntimeConfig",
    "ScheduleKeyedPrefetcher",
    "ScheduledCacheError",
    "assemble_batched_gpu_matching_batch",
    "batched_gpu_patches",
    "batched_patch_table_from_prepared_volumes",
]
