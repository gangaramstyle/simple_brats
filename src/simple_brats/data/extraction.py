"""Deterministic, fail-closed NIfTI preparation and v0 patch extraction.

There are deliberately two resampling stages with different responsibilities:

1. A complete native volume is reoriented to RAS and resampled once onto the
   physical grid pinned by :class:`ExtractionSpec`.
2. A registered integer crop from that grid is resized to the model-visible
   tensor.  The primary crops are 4 x 4 x 4 and 8 x 8 x 8 voxels, both shown
   to the model as 8 x 8 x 8.  The second stage never reads outside the
   integer crop.

Consequently, ``PatchInterpolationSupport`` names the exact canonical voxels
that can affect a patch.  Callers can prohibit overlap with a held target's
support instead of approximating interpolation leakage with center distance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import TypeAlias

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .manifest import canonical_json_bytes

EXTRACTION_SCHEMA = "simple_brats.extraction"
EXTRACTION_SCHEMA_VERSION = 2
V0_MODALITIES = ("t1n", "t1c", "t2w", "t2f")

Int3: TypeAlias = tuple[int, int, int]
Float3: TypeAlias = tuple[float, float, float]
Affine4: TypeAlias = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LATTICE_TOLERANCE = 1e-6
_AFFINE_TOLERANCE = 1e-7


class ExtractionError(ValueError):
    """Raised when image preparation cannot satisfy the pinned contract."""


def _int3(value: Iterable[int], *, name: str) -> Int3:
    try:
        result = tuple(value)
    except TypeError as error:
        raise ExtractionError(f"{name} must contain three integers") from error
    if len(result) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in result
    ):
        raise ExtractionError(f"{name} must contain three integers")
    if any(item <= 0 for item in result):
        raise ExtractionError(f"{name} must contain three positive integers")
    return result  # type: ignore[return-value]


def _float3(value: Iterable[float], *, name: str) -> Float3:
    try:
        raw = tuple(value)
    except TypeError as error:
        raise ExtractionError(f"{name} must contain three finite numbers") from error
    if len(raw) != 3 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw
    ):
        raise ExtractionError(f"{name} must contain three finite numbers")
    result = tuple(float(item) for item in raw)
    if not all(isfinite(item) for item in result):
        raise ExtractionError(f"{name} must contain three finite numbers")
    return tuple(0.0 if item == 0.0 else item for item in result)  # type: ignore[return-value]


def _affine4(value: object, *, name: str) -> Affine4:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ExtractionError(f"{name} must be a numeric 4x4 matrix") from error
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ExtractionError(f"{name} must be a finite numeric 4x4 matrix")
    if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=_AFFINE_TOLERANCE, rtol=0):
        raise ExtractionError(f"{name} must have homogeneous final row [0, 0, 0, 1]")
    linear = array[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not isfinite(determinant) or abs(determinant) <= 1e-8:
        raise ExtractionError(f"{name} has a singular or near-singular spatial transform")
    condition = float(np.linalg.cond(linear))
    if not isfinite(condition) or condition > 1e8:
        raise ExtractionError(f"{name} has an ill-conditioned spatial transform")
    rows = tuple(tuple(float(item) for item in row) for row in array)
    return rows  # type: ignore[return-value]


def _array_read_only(value: np.ndarray, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ExtractionSpec:
    """Immutable provenance for the complete v0 data-generating transform.

    The canonical grid must be chosen once after inspecting the release and
    then reused verbatim for every case and objective arm.  It is intentionally
    required rather than inferred per image.  V0 pins an axis-aligned RAS+
    1 mm grid; this makes each registered millimetre extent an exact integer
    voxel crop before the model-only resize.
    """

    canonical_shape: Int3
    canonical_affine: Affine4
    orientation: str = "RAS+"
    source_spatial_unit_policy: str = "mm-or-unknown-after-case-mm-consensus"
    world_origin_policy: str = "preserve-case-physical-bounds"
    volume_interpolation: str = "trilinear"
    volume_align_corners: bool = True
    volume_padding: str = "zeros"
    volume_dtype: str = "float32"
    foreground_rule: str = "valid-and-nonzero"
    normalization: str = "foreground-zscore-clip"
    normalization_clip: tuple[float, float] = (-5.0, 5.0)
    normalization_epsilon: float = 1e-6
    patch_source_shape: Int3 = (4, 4, 1)
    patch_physical_extent_mm: Float3 = (4.0, 4.0, 1.0)
    model_visible_shape: Int3 = (16, 16, 1)
    patch_interpolation: str = "trilinear"
    patch_align_corners: bool = False
    patch_padding: str = "forbid"
    schema: str = EXTRACTION_SCHEMA
    schema_version: int = EXTRACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        shape = _int3(self.canonical_shape, name="canonical_shape")
        affine = _affine4(self.canonical_affine, name="canonical_affine")
        source_shape = _int3(self.patch_source_shape, name="patch_source_shape")
        model_shape = _int3(self.model_visible_shape, name="model_visible_shape")
        extent = _float3(self.patch_physical_extent_mm, name="patch_physical_extent_mm")
        clip = tuple(self.normalization_clip)
        if len(clip) != 2 or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and isfinite(float(item))
            for item in clip
        ):
            raise ExtractionError("normalization_clip must contain two finite numbers")
        clip_float = (float(clip[0]), float(clip[1]))
        if clip_float[0] >= clip_float[1]:
            raise ExtractionError("normalization_clip lower bound must be below its upper bound")

        fixed_strings = {
            "orientation": (self.orientation, "RAS+"),
            "source_spatial_unit_policy": (
                self.source_spatial_unit_policy,
                "mm-or-unknown-after-case-mm-consensus",
            ),
            "world_origin_policy": (
                self.world_origin_policy,
                "preserve-case-physical-bounds",
            ),
            "volume_interpolation": (self.volume_interpolation, "trilinear"),
            "volume_padding": (self.volume_padding, "zeros"),
            "volume_dtype": (self.volume_dtype, "float32"),
            "foreground_rule": (self.foreground_rule, "valid-and-nonzero"),
            "normalization": (self.normalization, "foreground-zscore-clip"),
            "patch_interpolation": (self.patch_interpolation, "trilinear"),
            "patch_padding": (self.patch_padding, "forbid"),
            "schema": (self.schema, EXTRACTION_SCHEMA),
        }
        for field_name, (actual, expected) in fixed_strings.items():
            if actual != expected:
                raise ExtractionError(f"v0 requires {field_name}={expected!r}")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != EXTRACTION_SCHEMA_VERSION
        ):
            raise ExtractionError(f"v0 requires schema_version={EXTRACTION_SCHEMA_VERSION}")
        if self.volume_align_corners is not True:
            raise ExtractionError("v0 whole-volume resampling requires align_corners=True")
        if self.patch_align_corners is not False:
            raise ExtractionError("v0 patch resizing requires align_corners=False")
        if (
            isinstance(self.normalization_epsilon, bool)
            or not isinstance(self.normalization_epsilon, (int, float))
            or not isfinite(float(self.normalization_epsilon))
            or self.normalization_epsilon <= 0
        ):
            raise ExtractionError("normalization_epsilon must be finite and positive")

        affine_array = np.asarray(affine)
        linear = affine_array[:3, :3]
        diagonal = np.diag(np.diag(linear))
        if not np.allclose(linear, diagonal, atol=_AFFINE_TOLERANCE, rtol=0):
            raise ExtractionError(
                "canonical_affine must be axis-aligned; oblique grids are forbidden"
            )
        spacing = tuple(float(item) for item in np.diag(linear))
        if any(item <= 0 for item in spacing) or tuple(nib.aff2axcodes(affine_array)) != (
            "R",
            "A",
            "S",
        ):
            raise ExtractionError("canonical_affine must use positive RAS+ axes")
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=_AFFINE_TOLERANCE, rtol=0):
            raise ExtractionError("v0 canonical grid spacing must be exactly 1 mm isotropic")
        registered_geometry = {
            ((4, 4, 1), (4.0, 4.0, 1.0), (16, 16, 1)),
            ((4, 4, 4), (4.0, 4.0, 4.0), (8, 8, 8)),
            ((8, 8, 8), (8.0, 8.0, 8.0), (8, 8, 8)),
            ((4, 4, 4), (4.0, 4.0, 4.0), (16, 16, 16)),
            ((8, 8, 8), (8.0, 8.0, 8.0), (16, 16, 16)),
        }
        if (source_shape, extent, model_shape) not in registered_geometry:
            raise ExtractionError(
                "patch geometry must be a 4 or 8 mm isotropic cube resized to "
                "8x8x8 or legacy 16x16x16, or the load-only 4x4x1 mm / 16x16x1 slab"
            )
        if any(
            grid_size < crop_size for grid_size, crop_size in zip(shape, source_shape, strict=True)
        ):
            raise ExtractionError("canonical grid must contain at least one complete source crop")
        object.__setattr__(self, "canonical_shape", shape)
        object.__setattr__(self, "canonical_affine", affine)
        object.__setattr__(self, "patch_source_shape", source_shape)
        object.__setattr__(self, "patch_physical_extent_mm", extent)
        object.__setattr__(self, "model_visible_shape", model_shape)
        object.__setattr__(self, "normalization_clip", clip_float)
        object.__setattr__(self, "normalization_epsilon", float(self.normalization_epsilon))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "canonical_shape": list(self.canonical_shape),
            "canonical_affine": [list(row) for row in self.canonical_affine],
            "orientation": self.orientation,
            "source_spatial_unit_policy": self.source_spatial_unit_policy,
            "world_origin_policy": self.world_origin_policy,
            "volume_interpolation": self.volume_interpolation,
            "volume_align_corners": self.volume_align_corners,
            "volume_padding": self.volume_padding,
            "volume_dtype": self.volume_dtype,
            "foreground_rule": self.foreground_rule,
            "normalization": self.normalization,
            "normalization_clip": list(self.normalization_clip),
            "normalization_epsilon": self.normalization_epsilon,
            "patch_source_shape": list(self.patch_source_shape),
            "patch_physical_extent_mm": list(self.patch_physical_extent_mm),
            "model_visible_shape": list(self.model_visible_shape),
            "patch_interpolation": self.patch_interpolation,
            "patch_align_corners": self.patch_align_corners,
            "patch_padding": self.patch_padding,
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExtractionSpec:
        if not isinstance(value, Mapping):
            raise ExtractionError("extraction spec must be a JSON object")
        expected = {
            "schema",
            "schema_version",
            "canonical_shape",
            "canonical_affine",
            "orientation",
            "source_spatial_unit_policy",
            "world_origin_policy",
            "volume_interpolation",
            "volume_align_corners",
            "volume_padding",
            "volume_dtype",
            "foreground_rule",
            "normalization",
            "normalization_clip",
            "normalization_epsilon",
            "patch_source_shape",
            "patch_physical_extent_mm",
            "model_visible_shape",
            "patch_interpolation",
            "patch_align_corners",
            "patch_padding",
        }
        if set(value) != expected:
            raise ExtractionError(
                "extraction spec keys differ from the pinned schema: "
                f"missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
            )
        return cls(**value)  # type: ignore[arg-type]


def _decode_extraction_spec_json(payload: str | bytes | bytearray) -> object:
    if not isinstance(payload, (str, bytes, bytearray)):
        raise ExtractionError("extraction-spec JSON must be str, bytes, or bytearray")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ExtractionError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> object:
        raise ExtractionError(f"non-finite JSON number {token!r} is forbidden")

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except ExtractionError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExtractionError(f"invalid extraction-spec JSON: {error}") from error


def save_extraction_spec(
    spec: ExtractionSpec,
    path: str | os.PathLike[str],
) -> None:
    """Write exactly the canonical bytes addressed by ``spec.sha256``."""

    if not isinstance(spec, ExtractionSpec):
        raise TypeError("spec must be an ExtractionSpec")
    Path(path).write_bytes(canonical_json_bytes(spec.to_dict()))


def load_extraction_spec(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> ExtractionSpec:
    """Load a strict extraction spec and optionally require its canonical SHA."""

    value = _decode_extraction_spec_json(Path(path).read_bytes())
    if not isinstance(value, Mapping):
        raise ExtractionError("extraction spec must be a JSON object")
    spec = ExtractionSpec.from_dict(value)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ExtractionError("expected_sha256 must be a lowercase SHA-256 digest")
        if spec.sha256 != expected_sha256:
            raise ExtractionError(
                f"extraction-spec SHA mismatch: expected {expected_sha256}, got {spec.sha256}"
            )
    return spec


@dataclass(frozen=True, slots=True, eq=False)
class LoadedNifti:
    """Finite three-dimensional image after lossless RAS reorientation."""

    data: np.ndarray = field(repr=False)
    affine: np.ndarray = field(repr=False)
    source_path: str
    source_orientation: tuple[str, str, str]

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3 or any(size <= 0 for size in data.shape):
            raise ExtractionError("loaded NIfTI data must be a non-empty three-dimensional array")
        if not np.isfinite(data).all():
            raise ExtractionError("loaded NIfTI contains non-finite voxel values")
        affine_tuple = _affine4(self.affine, name="loaded affine")
        if tuple(nib.aff2axcodes(np.asarray(affine_tuple))) != ("R", "A", "S"):
            raise ExtractionError("LoadedNifti must be reoriented to RAS")
        orientation = tuple(self.source_orientation)
        if len(orientation) != 3 or any(not isinstance(item, str) for item in orientation):
            raise ExtractionError("source_orientation must contain three axis codes")
        object.__setattr__(self, "data", _array_read_only(data, dtype=np.dtype(np.float32)))
        object.__setattr__(
            self,
            "affine",
            _array_read_only(np.asarray(affine_tuple), dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "source_orientation", orientation)


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    foreground_voxels: int
    mean: float
    std: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.foreground_voxels, bool)
            or not isinstance(self.foreground_voxels, int)
            or self.foreground_voxels < 2
        ):
            raise ExtractionError("normalization requires at least two foreground voxels")
        if not isfinite(self.mean) or not isfinite(self.std) or self.std <= 0:
            raise ExtractionError("normalization statistics must be finite with positive std")


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalVolume:
    """One normalized modality on the immutable physical grid."""

    data: np.ndarray = field(repr=False)
    valid_support_mask: np.ndarray = field(repr=False)
    foreground_mask: np.ndarray = field(repr=False)
    affine: np.ndarray = field(repr=False)
    extraction_spec_sha256: str
    voxel_content_sha256: str
    normalized_sha256: str
    normalization_stats: NormalizationStats

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        mask = np.asarray(self.valid_support_mask)
        foreground = np.asarray(self.foreground_mask)
        if data.ndim != 3 or data.shape != mask.shape or data.shape != foreground.shape:
            raise ExtractionError(
                "canonical data, support mask, and foreground mask must have the same 3D shape"
            )
        if not np.isfinite(data).all():
            raise ExtractionError("canonical volume contains non-finite voxel values")
        if mask.dtype != np.bool_:
            raise ExtractionError("valid_support_mask must have boolean dtype")
        if foreground.dtype != np.bool_:
            raise ExtractionError("foreground_mask must have boolean dtype")
        if np.any(foreground & ~mask):
            raise ExtractionError("foreground_mask must be a subset of valid_support_mask")
        affine = _affine4(self.affine, name="canonical volume affine")
        for name in (
            "extraction_spec_sha256",
            "voxel_content_sha256",
            "normalized_sha256",
        ):
            digest = getattr(self, name)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ExtractionError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.normalization_stats, NormalizationStats):
            raise ExtractionError("normalization_stats must be a NormalizationStats record")
        object.__setattr__(self, "data", _array_read_only(data, dtype=np.dtype(np.float32)))
        object.__setattr__(
            self,
            "valid_support_mask",
            _array_read_only(mask, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "foreground_mask",
            _array_read_only(foreground, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "affine",
            _array_read_only(np.asarray(affine), dtype=np.dtype(np.float64)),
        )


@dataclass(frozen=True, slots=True)
class PatchInterpolationSupport:
    """Closed physical support of the canonical voxels influencing a patch."""

    start_ijk: Int3
    stop_ijk: Int3
    lower_mm: Float3
    upper_mm: Float3

    def __post_init__(self) -> None:
        start = tuple(self.start_ijk)
        stop = tuple(self.stop_ijk)
        if (
            len(start) != 3
            or len(stop) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in (*start, *stop))
        ):
            raise ExtractionError("patch support indices must contain three integers")
        if any(first < 0 or second <= first for first, second in zip(start, stop, strict=True)):
            raise ExtractionError("patch support must contain non-empty, non-negative index ranges")
        lower = _float3(self.lower_mm, name="support.lower_mm")
        upper = _float3(self.upper_mm, name="support.upper_mm")
        if any(first >= second for first, second in zip(lower, upper, strict=True)):
            raise ExtractionError("patch support physical bounds must have positive extent")
        object.__setattr__(self, "start_ijk", start)
        object.__setattr__(self, "stop_ijk", stop)
        object.__setattr__(self, "lower_mm", lower)
        object.__setattr__(self, "upper_mm", upper)

    @property
    def source_shape(self) -> Int3:
        return tuple(
            stop - start for start, stop in zip(self.start_ijk, self.stop_ijk, strict=True)
        )  # type: ignore[return-value]

    def intersects(self, other: PatchInterpolationSupport) -> bool:
        """Conservatively treat support boxes that touch at a boundary as intersecting."""

        return all(
            max(first_lower, second_lower) <= min(first_upper, second_upper)
            for first_lower, first_upper, second_lower, second_upper in zip(
                self.lower_mm,
                self.upper_mm,
                other.lower_mm,
                other.upper_mm,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True, eq=False)
class ExtractedPatch:
    data: np.ndarray = field(repr=False)
    center_mm: Float3
    support: PatchInterpolationSupport

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3 or any(size <= 0 for size in data.shape) or not np.isfinite(data).all():
            raise ExtractionError("extracted patch must be a finite, non-empty 3D tensor")
        if not isinstance(self.support, PatchInterpolationSupport):
            raise ExtractionError("support must be a PatchInterpolationSupport")
        object.__setattr__(self, "center_mm", _float3(self.center_mm, name="center_mm"))
        object.__setattr__(self, "data", _array_read_only(data, dtype=np.dtype(np.float32)))


def load_nifti_ras(
    path: str | os.PathLike[str],
    *,
    allow_unknown_spatial_unit: bool = False,
) -> LoadedNifti:
    """Load a finite 3D NIfTI and losslessly reorient its axes to RAS.

    Symlinks are rejected to preserve the discovery manifest's path boundary.
    Arbitrary obliquity is allowed here and handled by the subsequent physical
    resampling, but singular, non-finite, or ill-conditioned affines fail.
    """

    filesystem_path = Path(path).expanduser()
    if filesystem_path.is_symlink():
        raise ExtractionError(f"NIfTI path must not be a symlink: {filesystem_path}")
    try:
        resolved = filesystem_path.resolve(strict=True)
    except OSError as error:
        raise ExtractionError(f"NIfTI path is unavailable: {filesystem_path}") from error
    if not resolved.is_file():
        raise ExtractionError(f"NIfTI path is not a regular file: {filesystem_path}")

    try:
        image = nib.load(os.fspath(resolved), mmap="r")
    except Exception as error:
        raise ExtractionError(f"unable to load NIfTI {filesystem_path}: {error}") from error
    if len(image.shape) != 3 or any(size <= 0 for size in image.shape):
        raise ExtractionError("NIfTI must contain exactly one non-empty 3D volume")
    if not isinstance(allow_unknown_spatial_unit, bool):
        raise TypeError("allow_unknown_spatial_unit must be boolean")
    spatial_unit, _ = image.header.get_xyzt_units()
    if spatial_unit != "mm" and not (spatial_unit == "unknown" and allow_unknown_spatial_unit):
        raise ExtractionError(
            f"NIfTI spatial units must be explicitly millimetres, got {spatial_unit!r}"
        )
    source_affine = _affine4(image.affine, name="NIfTI affine")
    source_orientation_raw = nib.aff2axcodes(np.asarray(source_affine))
    if any(item is None for item in source_orientation_raw):
        raise ExtractionError("NIfTI affine does not define three anatomical axes")
    source_orientation = tuple(str(item) for item in source_orientation_raw)

    try:
        canonical_image = nib.as_closest_canonical(image, enforce_diag=False)
        canonical_affine = _affine4(canonical_image.affine, name="RAS NIfTI affine")
        data = canonical_image.get_fdata(dtype=np.float32, caching="unchanged")
    except Exception as error:
        raise ExtractionError(
            f"unable to reorient/read NIfTI {filesystem_path}: {error}"
        ) from error
    if tuple(nib.aff2axcodes(np.asarray(canonical_affine))) != ("R", "A", "S"):
        raise ExtractionError("NIfTI could not be represented in RAS orientation")
    if not np.isfinite(data).all():
        raise ExtractionError("NIfTI contains non-finite voxel values after header scaling")
    return LoadedNifti(
        data=data,
        affine=np.asarray(canonical_affine),
        source_path=os.fspath(resolved),
        source_orientation=source_orientation,  # type: ignore[arg-type]
    )


def _normalized_grid_coordinate(coordinate: torch.Tensor, size: int) -> torch.Tensor:
    if size == 1:
        return torch.zeros_like(coordinate)
    return 2.0 * coordinate / float(size - 1) - 1.0


def resample_to_canonical_grid(
    image: LoadedNifti,
    spec: ExtractionSpec,
    *,
    chunk_depth: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Trilinearly resample an entire RAS image and return its support-valid mask.

    A canonical output voxel is marked valid only when the complete linear
    interpolation support lies inside the native image.  Padded values can
    therefore never pass an extraction bounds check merely because zero is a
    plausible MRI intensity.
    """

    if not isinstance(image, LoadedNifti) or not isinstance(spec, ExtractionSpec):
        raise TypeError("image and spec must be LoadedNifti and ExtractionSpec")
    if isinstance(chunk_depth, bool) or not isinstance(chunk_depth, int) or chunk_depth <= 0:
        raise ValueError("chunk_depth must be a positive integer")

    target_affine = np.asarray(spec.canonical_affine, dtype=np.float64)
    if image.data.shape == spec.canonical_shape and np.array_equal(image.affine, target_affine):
        data = np.array(image.data, dtype=np.float32, order="C", copy=True)
        support = np.ones(spec.canonical_shape, dtype=np.bool_)
        return data, support

    source_shape = tuple(int(item) for item in image.data.shape)
    source_tensor = (
        torch.from_numpy(np.array(image.data, dtype=np.float32, order="C", copy=True))
        .permute(2, 1, 0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    transform = torch.as_tensor(
        np.linalg.inv(image.affine) @ target_affine,
        dtype=torch.float64,
    )
    output = np.empty(spec.canonical_shape, dtype=np.float32)
    valid_output = np.empty(spec.canonical_shape, dtype=np.bool_)
    target_x, target_y, target_z = spec.canonical_shape

    with torch.no_grad():
        for z_start in range(0, target_z, chunk_depth):
            z_stop = min(z_start + chunk_depth, target_z)
            zz, yy, xx = torch.meshgrid(
                torch.arange(z_start, z_stop, dtype=torch.float64),
                torch.arange(target_y, dtype=torch.float64),
                torch.arange(target_x, dtype=torch.float64),
                indexing="ij",
            )
            source_x = (
                transform[0, 0] * xx + transform[0, 1] * yy + transform[0, 2] * zz + transform[0, 3]
            )
            source_y = (
                transform[1, 0] * xx + transform[1, 1] * yy + transform[1, 2] * zz + transform[1, 3]
            )
            source_z = (
                transform[2, 0] * xx + transform[2, 1] * yy + transform[2, 2] * zz + transform[2, 3]
            )
            valid = (
                (source_x >= -_LATTICE_TOLERANCE)
                & (source_x <= source_shape[0] - 1 + _LATTICE_TOLERANCE)
                & (source_y >= -_LATTICE_TOLERANCE)
                & (source_y <= source_shape[1] - 1 + _LATTICE_TOLERANCE)
                & (source_z >= -_LATTICE_TOLERANCE)
                & (source_z <= source_shape[2] - 1 + _LATTICE_TOLERANCE)
            )
            source_x = source_x.clamp(0, source_shape[0] - 1)
            source_y = source_y.clamp(0, source_shape[1] - 1)
            source_z = source_z.clamp(0, source_shape[2] - 1)
            grid = torch.stack(
                (
                    _normalized_grid_coordinate(source_x, source_shape[0]),
                    _normalized_grid_coordinate(source_y, source_shape[1]),
                    _normalized_grid_coordinate(source_z, source_shape[2]),
                ),
                dim=-1,
            ).to(dtype=torch.float32)
            sampled = F.grid_sample(
                source_tensor,
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0]
            sampled = sampled.masked_fill(~valid, 0.0)
            output[:, :, z_start:z_stop] = sampled.permute(2, 1, 0).cpu().numpy()
            valid_output[:, :, z_start:z_stop] = valid.permute(2, 1, 0).cpu().numpy()

    if not np.isfinite(output).all():
        raise ExtractionError("whole-volume resampling produced non-finite values")
    return output, valid_output


def canonical_voxel_digest(
    data: np.ndarray,
    valid_support_mask: np.ndarray,
    affine: object,
) -> str:
    """Hash canonical voxel values independent of NIfTI container bytes."""

    values = np.asarray(data)
    support = np.asarray(valid_support_mask)
    if values.ndim != 3 or values.shape != support.shape or support.dtype != np.bool_:
        raise ExtractionError("digest data and boolean support mask must have the same 3D shape")
    if not np.isfinite(values).all():
        raise ExtractionError("cannot hash non-finite canonical voxels")
    affine_tuple = _affine4(affine, name="digest affine")
    canonical_values = np.array(values, dtype=np.dtype("<f4"), order="C", copy=True)
    canonical_values[canonical_values == 0] = 0.0  # collapse negative zero
    metadata = {
        "schema": "simple_brats.canonical_voxels",
        "schema_version": 1,
        "shape": list(canonical_values.shape),
        "affine": [list(row) for row in affine_tuple],
        "value_dtype": "little-endian-float32",
        "support_dtype": "uint8",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(canonical_values.tobytes(order="C"))
    digest.update(np.asarray(support, dtype=np.uint8, order="C").tobytes(order="C"))
    return digest.hexdigest()


def normalize_canonical_volume(
    data: np.ndarray,
    valid_support_mask: np.ndarray,
    spec: ExtractionSpec,
) -> tuple[np.ndarray, NormalizationStats]:
    """Apply the pinned valid-nonzero foreground z-score and clipping rule."""

    values = np.asarray(data)
    support = np.asarray(valid_support_mask)
    if values.shape != spec.canonical_shape or support.shape != spec.canonical_shape:
        raise ExtractionError("canonical array shape does not match ExtractionSpec")
    if support.dtype != np.bool_:
        raise ExtractionError("valid_support_mask must have boolean dtype")
    if not np.isfinite(values).all():
        raise ExtractionError("cannot normalize a non-finite canonical volume")
    foreground = support & (values != 0)
    count = int(foreground.sum())
    if count < 2:
        raise ExtractionError("normalization foreground requires at least two valid nonzero voxels")
    foreground_values = np.asarray(values[foreground], dtype=np.float64)
    mean = float(foreground_values.mean(dtype=np.float64))
    std = float(foreground_values.std(dtype=np.float64))
    if not isfinite(mean) or not isfinite(std) or std <= spec.normalization_epsilon:
        raise ExtractionError("foreground standard deviation is zero or below the pinned epsilon")
    normalized = np.zeros(spec.canonical_shape, dtype=np.float32)
    normalized[foreground] = np.clip(
        (foreground_values - mean) / std,
        spec.normalization_clip[0],
        spec.normalization_clip[1],
    ).astype(np.float32)
    return normalized, NormalizationStats(foreground_voxels=count, mean=mean, std=std)


def prepare_canonical_volume(
    path: str | os.PathLike[str],
    spec: ExtractionSpec,
) -> CanonicalVolume:
    """Load, RAS-orient, resample, digest, and normalize one modality."""

    image = load_nifti_ras(path, allow_unknown_spatial_unit=True)
    resampled, valid_support = resample_to_canonical_grid(image, spec)
    voxel_digest = canonical_voxel_digest(resampled, valid_support, spec.canonical_affine)
    foreground = valid_support & (resampled != 0)
    normalized, stats = normalize_canonical_volume(resampled, valid_support, spec)
    normalized_digest = canonical_voxel_digest(
        normalized,
        valid_support,
        spec.canonical_affine,
    )
    return CanonicalVolume(
        data=normalized,
        valid_support_mask=valid_support,
        foreground_mask=foreground,
        affine=np.asarray(spec.canonical_affine),
        extraction_spec_sha256=spec.sha256,
        voxel_content_sha256=voxel_digest,
        normalized_sha256=normalized_digest,
        normalization_stats=stats,
    )


def intersect_modality_foreground_support_masks(
    volumes: Mapping[str, CanonicalVolume],
    *,
    spec: ExtractionSpec,
) -> np.ndarray:
    """Return the strict label-free foreground/support intersection for v0.

    Exactly the four registered MRI modalities are required.  A voxel is true
    only when it is post-resample foreground *and* has fully native-supported
    interpolation in every modality.  The returned read-only mask is suitable
    for :func:`valid_patch_centers_mm` and cannot encode segmentation labels.
    """

    if not isinstance(volumes, Mapping):
        raise TypeError("volumes must map canonical modality names to CanonicalVolume records")
    if set(volumes) != set(V0_MODALITIES):
        raise ExtractionError(
            f"v0 foreground intersection requires exactly modalities {V0_MODALITIES}"
        )
    if not isinstance(spec, ExtractionSpec):
        raise TypeError("spec must be an ExtractionSpec")
    intersection = np.ones(spec.canonical_shape, dtype=np.bool_)
    for modality in V0_MODALITIES:
        volume = volumes[modality]
        if not isinstance(volume, CanonicalVolume):
            raise TypeError(f"volume {modality!r} must be a CanonicalVolume")
        if volume.extraction_spec_sha256 != spec.sha256:
            raise ExtractionError(
                f"volume {modality!r} was produced by a different extraction spec"
            )
        if volume.data.shape != spec.canonical_shape or not np.array_equal(
            volume.affine, np.asarray(spec.canonical_affine)
        ):
            raise ExtractionError(f"volume {modality!r} is not on the pinned canonical grid")
        intersection &= volume.valid_support_mask & volume.foreground_mask
    intersection.setflags(write=False)
    return intersection


def _window_sums(mask: np.ndarray, window: Int3) -> np.ndarray:
    """Return all valid box sums without materializing a strided window view."""

    result = np.asarray(mask, dtype=np.int32)
    for axis, width in enumerate(window):
        cumulative = np.cumsum(result, axis=axis, dtype=np.int32)
        zero_shape = list(cumulative.shape)
        zero_shape[axis] = 1
        padded = np.concatenate(
            (np.zeros(zero_shape, dtype=np.int32), cumulative),
            axis=axis,
        )
        upper = [slice(None)] * 3
        lower = [slice(None)] * 3
        upper[axis] = slice(width, None)
        lower[axis] = slice(None, -width)
        result = padded[tuple(upper)] - padded[tuple(lower)]
    return result


def valid_patch_centers_mm(
    spec: ExtractionSpec,
    modality_agnostic_foreground_support_mask: np.ndarray,
) -> np.ndarray:
    """Enumerate fixed-lattice centers whose complete source crop is eligible.

    ``modality_agnostic_foreground_support_mask`` should normally come from
    :func:`intersect_modality_foreground_support_masks`.  Every voxel in the
    complete 3D integer crop must be true; a center touching invalid support,
    padding, or any modality's non-foreground region is omitted.

    The returned read-only array has shape ``(candidate_count, 3)`` and stores
    physical RAS millimeters, ready for ``CandidatePosition.center_mm``.
    """

    if not isinstance(spec, ExtractionSpec):
        raise TypeError("spec must be an ExtractionSpec")
    mask = np.asarray(modality_agnostic_foreground_support_mask)
    if mask.shape != spec.canonical_shape or mask.dtype != np.bool_:
        raise ExtractionError(
            "modality-agnostic foreground/support mask must be boolean and match the canonical grid"
        )
    window_volume = int(np.prod(spec.patch_source_shape))
    eligible_starts = np.argwhere(_window_sums(mask, spec.patch_source_shape) == window_volume)
    if eligible_starts.size == 0:
        centers = np.empty((0, 3), dtype=np.float64)
    else:
        center_voxels = (
            eligible_starts.astype(np.float64)
            + (np.asarray(spec.patch_source_shape, dtype=np.float64) - 1.0) / 2.0
        )
        # ExtractionSpec guarantees an axis-aligned affine.  Applying its
        # diagonal and translation directly is both clearer and avoids a
        # platform BLAS warning seen for wide 4xN homogeneous products.
        affine = np.asarray(spec.canonical_affine, dtype=np.float64)
        centers = center_voxels * np.diag(affine[:3, :3]) + affine[:3, 3]
    centers = np.array(centers, dtype=np.float64, order="C", copy=True)
    centers.setflags(write=False)
    return centers


def patch_interpolation_support(
    spec: ExtractionSpec,
    center_mm: Iterable[float],
) -> PatchInterpolationSupport:
    """Return the exact integer canonical crop influencing ``center_mm``.

    Even crop widths require half-integer canonical indices on their axes;
    odd widths require integer indices.  Rejecting every other phase prevents
    subvoxel interpolation phase from encoding location.
    """

    if not isinstance(spec, ExtractionSpec):
        raise TypeError("spec must be an ExtractionSpec")
    center = _float3(center_mm, name="center_mm")
    world = np.asarray((*center, 1.0), dtype=np.float64)
    voxel = (np.linalg.inv(np.asarray(spec.canonical_affine)) @ world)[:3]
    source_shape = np.asarray(spec.patch_source_shape, dtype=np.float64)
    start_float = voxel - (source_shape - 1.0) / 2.0
    start = np.rint(start_float).astype(np.int64)
    if not np.allclose(start_float, start, atol=_LATTICE_TOLERANCE, rtol=0):
        phase = tuple(float(item % 1.0) for item in voxel)
        expected_phase = tuple(float(item % 1.0) for item in (source_shape - 1.0) / 2.0)
        raise ExtractionError(
            "patch center is off the pinned canonical lattice; expected voxel-index phases "
            f"{expected_phase}, got {phase}"
        )
    stop = start + np.asarray(spec.patch_source_shape, dtype=np.int64)
    if np.any(start < 0) or np.any(stop > np.asarray(spec.canonical_shape)):
        raise ExtractionError("patch interpolation support extends outside the canonical grid")

    # Canonical axes are positive and diagonal by spec.  Voxel i occupies the
    # closed physical cell [i - 0.5, i + 0.5], so a crop [start, stop) has
    # physical bounds at start - 0.5 and stop - 0.5.
    affine = np.asarray(spec.canonical_affine)
    lower_index = np.asarray((*((start - 0.5).tolist()), 1.0))
    upper_index = np.asarray((*((stop - 0.5).tolist()), 1.0))
    lower = tuple(float(item) for item in (affine @ lower_index)[:3])
    upper = tuple(float(item) for item in (affine @ upper_index)[:3])
    return PatchInterpolationSupport(
        start_ijk=tuple(int(item) for item in start),  # type: ignore[arg-type]
        stop_ijk=tuple(int(item) for item in stop),  # type: ignore[arg-type]
        lower_mm=lower,  # type: ignore[arg-type]
        upper_mm=upper,  # type: ignore[arg-type]
    )


def assert_interpolation_supports_disjoint(
    first: PatchInterpolationSupport,
    second: PatchInterpolationSupport,
) -> None:
    """Raise rather than permit overlapping or boundary-touching support."""

    if first.intersects(second):
        raise ExtractionError("patch interpolation supports overlap or touch at a boundary")


def assert_pairwise_interpolation_support_disjoint(
    supports: Sequence[PatchInterpolationSupport],
) -> None:
    for index, first in enumerate(supports):
        for second in supports[index + 1 :]:
            assert_interpolation_supports_disjoint(first, second)


def extract_patch(
    volume: CanonicalVolume,
    center_mm: Iterable[float],
    *,
    spec: ExtractionSpec,
    forbidden_supports: Iterable[PatchInterpolationSupport] = (),
) -> ExtractedPatch:
    """Extract one finite model tensor without reading outside its support."""

    if not isinstance(volume, CanonicalVolume) or not isinstance(spec, ExtractionSpec):
        raise TypeError("volume and spec must be CanonicalVolume and ExtractionSpec")
    if volume.extraction_spec_sha256 != spec.sha256:
        raise ExtractionError("canonical volume was produced by a different extraction spec")
    if volume.data.shape != spec.canonical_shape or not np.array_equal(
        volume.affine, np.asarray(spec.canonical_affine)
    ):
        raise ExtractionError("canonical volume grid does not match the extraction spec")
    center = _float3(center_mm, name="center_mm")
    support = patch_interpolation_support(spec, center)
    for forbidden in tuple(forbidden_supports):
        if not isinstance(forbidden, PatchInterpolationSupport):
            raise TypeError("forbidden_supports must contain PatchInterpolationSupport records")
        assert_interpolation_supports_disjoint(support, forbidden)

    slices = tuple(
        slice(start, stop) for start, stop in zip(support.start_ijk, support.stop_ijk, strict=True)
    )
    valid = volume.valid_support_mask[slices]
    if valid.shape != spec.patch_source_shape or not bool(valid.all()):
        raise ExtractionError(
            "patch reads canonical voxels with padded native interpolation support"
        )
    foreground = volume.foreground_mask[slices]
    if foreground.shape != spec.patch_source_shape or not bool(foreground.all()):
        raise ExtractionError(
            "complete patch crop must remain inside the modality foreground; "
            "background voxels are forbidden"
        )
    crop = volume.data[slices]
    if crop.shape != spec.patch_source_shape or not np.isfinite(crop).all():
        raise ExtractionError("canonical crop is incomplete or non-finite")
    tensor = (
        torch.from_numpy(np.array(crop, dtype=np.float32, order="C", copy=True))
        .permute(2, 1, 0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    with torch.no_grad():
        resized = F.interpolate(
            tensor,
            size=(
                spec.model_visible_shape[2],
                spec.model_visible_shape[1],
                spec.model_visible_shape[0],
            ),
            mode="trilinear",
            align_corners=False,
        )[0, 0].permute(2, 1, 0)
    patch = resized.cpu().numpy()
    if patch.shape != spec.model_visible_shape or not np.isfinite(patch).all():
        raise ExtractionError("patch resizing violated the model-visible tensor contract")
    return ExtractedPatch(data=patch, center_mm=center, support=support)


__all__ = [
    "EXTRACTION_SCHEMA",
    "EXTRACTION_SCHEMA_VERSION",
    "V0_MODALITIES",
    "CanonicalVolume",
    "ExtractedPatch",
    "ExtractionError",
    "ExtractionSpec",
    "LoadedNifti",
    "NormalizationStats",
    "PatchInterpolationSupport",
    "assert_interpolation_supports_disjoint",
    "assert_pairwise_interpolation_support_disjoint",
    "canonical_voxel_digest",
    "extract_patch",
    "intersect_modality_foreground_support_masks",
    "load_extraction_spec",
    "load_nifti_ras",
    "normalize_canonical_volume",
    "patch_interpolation_support",
    "prepare_canonical_volume",
    "resample_to_canonical_grid",
    "save_extraction_spec",
    "valid_patch_centers_mm",
]
