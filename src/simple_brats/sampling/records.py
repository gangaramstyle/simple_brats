"""Canonical, content-addressed records for materialized patch plans.

The sampler is stochastic, but objective comparisons should not be.  A
``MaterializedPatchPlan`` freezes one sampled case/bag independently of the
objective arm that will consume it.  The serialized record carries both the
physical extraction contract and every source/query/teacher patch identity,
so replay fails closed if any of those contracts drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from simple_brats.atomic_io import atomic_create_bytes, atomic_replace_bytes
from simple_brats.data.manifest import canonicalize_case_identity

from .geometry import SlabGeometry
from .modality_completion import (
    ModalityCompletionBatchPlan,
    PatchMetadata,
    registered_ordering_prism_extent,
)

PATCH_PLAN_SCHEMA = "simple-brats.materialized-patch-plan"
PATCH_PLAN_SCHEMA_VERSION = 2

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SEED = 2**64 - 1


class PatchPlanError(ValueError):
    """Raised when a materialized plan is ambiguous, altered, or non-canonical."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise PatchPlanError(f"{field} must be a string")
    if not value or value != value.strip():
        raise PatchPlanError(f"{field} must be non-empty and have no surrounding whitespace")
    if any(ord(character) < 32 for character in value):
        raise PatchPlanError(f"{field} must not contain control characters")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _required_text(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise PatchPlanError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PatchPlanError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise PatchPlanError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise PatchPlanError(f"{field} must be at most {maximum}")
    return value


def _coordinate3(
    value: object,
    field: str,
    *,
    positive: bool = False,
) -> tuple[float, float, float]:
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise PatchPlanError(f"{field} must contain three numeric coordinates") from error
    if len(raw) != 3 or any(
        isinstance(component, bool) or not isinstance(component, (int, float)) for component in raw
    ):
        raise PatchPlanError(f"{field} must contain three numeric coordinates")
    result = tuple(float(component) for component in raw)
    if not all(isfinite(component) for component in result):
        raise PatchPlanError(f"{field} must contain three finite coordinates")
    if positive and any(component <= 0 for component in result):
        raise PatchPlanError(f"{field} must contain three positive extents")
    return tuple(0.0 if component == 0.0 else component for component in result)  # type: ignore[return-value]


def _exact_keys(value: Mapping[str, object], expected: set[str], description: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {sorted(missing)}")
    if extra:
        details.append(f"unexpected {sorted(extra)}")
    raise PatchPlanError(f"invalid {description}: " + "; ".join(details))


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PatchPlanError(f"{description} must be a JSON object with string keys")
    return value


def _array(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise PatchPlanError(f"{description} must be a JSON array")
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return the sole JSON byte representation accepted by plan loading."""

    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PatchPlanError(f"value is not canonical-JSON serializable: {error}") from error
    return serialized.encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash a JSON extraction specification or other canonical metadata."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _decode_json(payload: str | bytes | bytearray) -> object:
    raw: str | bytes
    if isinstance(payload, bytearray):
        raw = bytes(payload)
    elif isinstance(payload, (str, bytes)):
        raw = payload
    else:
        raise PatchPlanError("patch-plan JSON must be str, bytes, or bytearray")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PatchPlanError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> object:
        raise PatchPlanError(f"non-finite JSON number {token!r} is forbidden")

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except PatchPlanError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PatchPlanError(f"invalid patch-plan JSON: {error}") from error


@dataclass(frozen=True, slots=True)
class PlanCaseIdentity:
    """Release-qualified case, subject, and visit identity for one plan."""

    source: str
    release: str
    case_id: str
    subject_id: str
    visit_id: str

    def __post_init__(self) -> None:
        source = _required_text(self.source, "case.source")
        release = _required_text(self.release, "case.release")
        try:
            identity = canonicalize_case_identity(
                self.case_id,
                subject_id=self.subject_id,
                visit_id=self.visit_id,
            )
        except ValueError as error:
            raise PatchPlanError(f"invalid case identity: {error}") from error
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "release", release)
        object.__setattr__(self, "case_id", identity.case_id)
        object.__setattr__(self, "subject_id", identity.subject_id)
        object.__setattr__(self, "visit_id", identity.visit_id)

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "release": self.release,
            "case_id": self.case_id,
            "subject_id": self.subject_id,
            "visit_id": self.visit_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PlanCaseIdentity:
        _exact_keys(
            value,
            {"source", "release", "case_id", "subject_id", "visit_id"},
            "plan case identity",
        )
        return cls(
            source=value["source"],  # type: ignore[arg-type]
            release=value["release"],  # type: ignore[arg-type]
            case_id=value["case_id"],  # type: ignore[arg-type]
            subject_id=value["subject_id"],  # type: ignore[arg-type]
            visit_id=value["visit_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GeometryRecord:
    """Canonical physical and model-visible geometry used for extraction."""

    in_plane_axes: tuple[int, int]
    thin_axis: int
    in_plane_footprint_mm: float
    thin_extent_mm: float
    model_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        try:
            axes = tuple(self.in_plane_axes)
            model_shape = tuple(self.model_shape)
        except TypeError as error:
            raise PatchPlanError("geometry axes and model_shape must be integer arrays") from error
        if len(axes) != 2 or any(
            isinstance(axis, bool) or not isinstance(axis, int) for axis in axes
        ):
            raise PatchPlanError("geometry.in_plane_axes must contain two integers")
        if isinstance(self.thin_axis, bool) or not isinstance(self.thin_axis, int):
            raise PatchPlanError("geometry.thin_axis must be an integer")
        if len(model_shape) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) for size in model_shape
        ):
            raise PatchPlanError("geometry.model_shape must contain three integers")
        for field, value in (
            ("in_plane_footprint_mm", self.in_plane_footprint_mm),
            ("thin_extent_mm", self.thin_extent_mm),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PatchPlanError(f"geometry.{field} must be numeric")
        try:
            geometry = SlabGeometry(
                in_plane_axes=axes,  # type: ignore[arg-type]
                thin_axis=self.thin_axis,
                in_plane_footprint_mm=float(self.in_plane_footprint_mm),
                thin_extent_mm=float(self.thin_extent_mm),
                model_shape=model_shape,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise PatchPlanError(f"invalid patch geometry: {error}") from error
        object.__setattr__(self, "in_plane_axes", geometry.in_plane_axes)
        object.__setattr__(self, "thin_axis", geometry.thin_axis)
        object.__setattr__(self, "in_plane_footprint_mm", geometry.in_plane_footprint_mm)
        object.__setattr__(self, "thin_extent_mm", geometry.thin_extent_mm)
        object.__setattr__(self, "model_shape", geometry.model_shape)

    @classmethod
    def from_geometry(cls, geometry: SlabGeometry) -> GeometryRecord:
        if not isinstance(geometry, SlabGeometry):
            raise TypeError("geometry must be a SlabGeometry")
        return cls(
            in_plane_axes=geometry.in_plane_axes,
            thin_axis=geometry.thin_axis,
            in_plane_footprint_mm=geometry.in_plane_footprint_mm,
            thin_extent_mm=geometry.thin_extent_mm,
            model_shape=geometry.model_shape,
        )

    def to_geometry(self) -> SlabGeometry:
        return SlabGeometry(
            in_plane_axes=self.in_plane_axes,
            thin_axis=self.thin_axis,
            in_plane_footprint_mm=self.in_plane_footprint_mm,
            thin_extent_mm=self.thin_extent_mm,
            model_shape=self.model_shape,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "in_plane_axes": list(self.in_plane_axes),
            "thin_axis": self.thin_axis,
            "in_plane_footprint_mm": self.in_plane_footprint_mm,
            "thin_extent_mm": self.thin_extent_mm,
            "model_shape": list(self.model_shape),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GeometryRecord:
        _exact_keys(
            value,
            {
                "in_plane_axes",
                "thin_axis",
                "in_plane_footprint_mm",
                "thin_extent_mm",
                "model_shape",
            },
            "patch geometry",
        )
        in_plane_axes = _array(value["in_plane_axes"], "geometry.in_plane_axes")
        model_shape = _array(value["model_shape"], "geometry.model_shape")
        return cls(
            in_plane_axes=tuple(in_plane_axes),  # type: ignore[arg-type]
            thin_axis=value["thin_axis"],  # type: ignore[arg-type]
            in_plane_footprint_mm=value["in_plane_footprint_mm"],  # type: ignore[arg-type]
            thin_extent_mm=value["thin_extent_mm"],  # type: ignore[arg-type]
            model_shape=tuple(model_shape),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class PatchIdentity:
    """Order-independent identity and physical center for one patch tensor."""

    position_id: int
    modality_id: int
    modality: str
    center_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        position_id = _integer(self.position_id, "patch.position_id")
        modality_id = _integer(self.modality_id, "patch.modality_id", minimum=0)
        modality = _required_text(self.modality, "patch.modality")
        try:
            raw_center = tuple(self.center_mm)
        except TypeError as error:
            raise PatchPlanError("patch.center_mm must contain numeric coordinates") from error
        if any(
            isinstance(component, bool) or not isinstance(component, (int, float))
            for component in raw_center
        ):
            raise PatchPlanError("patch.center_mm must contain numeric coordinates")
        center = tuple(float(component) for component in raw_center)
        if len(center) != 3 or not all(isfinite(component) for component in center):
            raise PatchPlanError("patch.center_mm must contain three finite coordinates")
        # Collapse negative zero so physically identical plans have one byte form.
        center = tuple(0.0 if component == 0.0 else component for component in center)
        object.__setattr__(self, "position_id", position_id)
        object.__setattr__(self, "modality_id", modality_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "center_mm", center)

    @property
    def key(self) -> tuple[int, int]:
        return (self.position_id, self.modality_id)

    @property
    def sort_key(self) -> tuple[int, int, str, tuple[float, float, float]]:
        return (*self.key, self.modality, self.center_mm)

    def to_dict(self) -> dict[str, object]:
        return {
            "position_id": self.position_id,
            "modality_id": self.modality_id,
            "modality": self.modality,
            "center_mm": list(self.center_mm),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PatchIdentity:
        _exact_keys(
            value,
            {"position_id", "modality_id", "modality", "center_mm"},
            "patch identity",
        )
        center = _array(value["center_mm"], "patch.center_mm")
        return cls(
            position_id=value["position_id"],  # type: ignore[arg-type]
            modality_id=value["modality_id"],  # type: ignore[arg-type]
            modality=value["modality"],  # type: ignore[arg-type]
            center_mm=tuple(center),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MaterializedPatchPlan:
    """One exactly replayable single-modality ordering bag."""

    data_manifest_sha256: str
    case: PlanCaseIdentity
    epoch: int
    bag_index: int
    seed: int
    modality_names: tuple[str, ...]
    geometry: GeometryRecord
    geometry_sha256: str
    extraction_spec_sha256: str
    prism_anchor_mm: tuple[float, float, float]
    prism_extent_mm: tuple[float, float, float]
    target_modality_id: int
    sources: tuple[PatchIdentity, ...]
    queries: tuple[PatchIdentity, ...]
    targets: tuple[PatchIdentity, ...]
    schema: str = PATCH_PLAN_SCHEMA
    schema_version: int = PATCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != PATCH_PLAN_SCHEMA:
            raise PatchPlanError(
                f"unsupported patch-plan schema {self.schema!r}; expected {PATCH_PLAN_SCHEMA!r}"
            )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PATCH_PLAN_SCHEMA_VERSION
        ):
            raise PatchPlanError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {PATCH_PLAN_SCHEMA_VERSION}"
            )

        data_manifest_sha256 = _sha256(self.data_manifest_sha256, "data_manifest_sha256")
        geometry_sha256 = _sha256(self.geometry_sha256, "geometry_sha256")
        extraction_spec_sha256 = _sha256(self.extraction_spec_sha256, "extraction_spec_sha256")
        epoch = _integer(self.epoch, "epoch", minimum=0)
        bag_index = _integer(self.bag_index, "bag_index", minimum=0)
        seed = _integer(self.seed, "seed", minimum=0, maximum=_MAX_SEED)
        if not isinstance(self.case, PlanCaseIdentity):
            raise PatchPlanError("case must be a PlanCaseIdentity")
        if not isinstance(self.geometry, GeometryRecord):
            raise PatchPlanError("geometry must be a GeometryRecord")
        if geometry_sha256 != self.geometry.sha256:
            raise PatchPlanError(
                f"geometry SHA mismatch: expected {geometry_sha256}, got {self.geometry.sha256}"
            )

        if isinstance(self.modality_names, (str, bytes)):
            raise PatchPlanError("modality_names must be an array of modality names")
        try:
            modality_names = tuple(
                _required_text(name, f"modality_names[{index}]")
                for index, name in enumerate(self.modality_names)
            )
        except TypeError as error:
            raise PatchPlanError("modality_names must be an array of modality names") from error
        if not modality_names or len(set(modality_names)) != len(modality_names):
            raise PatchPlanError("modality_names must contain distinct modality names")

        prism_anchor_mm = _coordinate3(self.prism_anchor_mm, "prism_anchor_mm")
        prism_extent_mm = _coordinate3(
            self.prism_extent_mm,
            "prism_extent_mm",
            positive=True,
        )
        try:
            registered_extent = registered_ordering_prism_extent(
                self.geometry.to_geometry(),
                prism_extent_mm,
            )
        except (TypeError, ValueError) as error:
            raise PatchPlanError(f"invalid registered ordering scale: {error}") from error
        if prism_extent_mm != registered_extent:
            raise PatchPlanError("prism_extent_mm does not match the registered ordering scale")
        target_modality_id = _integer(
            self.target_modality_id,
            "target_modality_id",
            minimum=0,
            maximum=len(modality_names) - 1,
        )

        sources = self._patch_tuple(self.sources, "sources")
        queries = self._patch_tuple(self.queries, "queries")
        targets = self._patch_tuple(self.targets, "targets")
        object.__setattr__(self, "data_manifest_sha256", data_manifest_sha256)
        object.__setattr__(self, "geometry_sha256", geometry_sha256)
        object.__setattr__(self, "extraction_spec_sha256", extraction_spec_sha256)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "bag_index", bag_index)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "modality_names", modality_names)
        object.__setattr__(self, "prism_anchor_mm", prism_anchor_mm)
        object.__setattr__(self, "prism_extent_mm", prism_extent_mm)
        object.__setattr__(self, "target_modality_id", target_modality_id)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "targets", targets)
        self.validate()

    @staticmethod
    def _patch_tuple(value: object, field: str) -> tuple[PatchIdentity, ...]:
        try:
            patches = tuple(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise PatchPlanError(f"{field} must contain PatchIdentity records") from error
        if not all(isinstance(patch, PatchIdentity) for patch in patches):
            raise PatchPlanError(f"{field} must contain PatchIdentity records")
        return tuple(sorted(patches, key=lambda patch: patch.sort_key))

    def validate(self) -> None:
        """Recheck all no-leakage and identity invariants."""

        for field, patches in (
            ("sources", self.sources),
            ("queries", self.queries),
            ("targets", self.targets),
        ):
            keys = [patch.key for patch in patches]
            if len(keys) != len(set(keys)):
                raise PatchPlanError(f"{field} contains duplicate patch identities")
            for patch in patches:
                if patch.modality_id >= len(self.modality_names):
                    raise PatchPlanError(
                        f"{field} patch modality_id {patch.modality_id} is out of range"
                    )
                expected_name = self.modality_names[patch.modality_id]
                if patch.modality != expected_name:
                    raise PatchPlanError(
                        f"{field} patch modality mapping mismatch: id {patch.modality_id} "
                        f"must be {expected_name!r}, got {patch.modality!r}"
                    )

        if len(self.targets) != 32 or len(self.queries) != 32:
            raise PatchPlanError("ordering plans require exactly 32 queries and targets")
        if self.queries != self.targets:
            raise PatchPlanError(
                "queries and targets must request identical position/modality/coordinate identities"
            )
        if any(target.modality_id != self.target_modality_id for target in self.targets):
            raise PatchPlanError("every query and target must use target_modality_id")
        target_positions = [target.position_id for target in self.targets]
        if len(set(target_positions)) != 32:
            raise PatchPlanError("ordering targets must use 32 distinct positions")

        if len(self.sources) != 96:
            raise PatchPlanError("ordering plans require exactly 96 sources")
        expected_source_counts = Counter(
            {
                modality_id: (6 if modality_id == self.target_modality_id else 30)
                for modality_id in range(len(self.modality_names))
            }
        )
        if Counter(source.modality_id for source in self.sources) != expected_source_counts:
            raise PatchPlanError(
                "source modality counts must be 6 for the target modality and "
                "30 for every other modality"
            )

        source_keys = {source.key for source in self.sources}
        target_keys = {target.key for target in self.targets}
        leaked = source_keys & target_keys
        if leaked:
            raise PatchPlanError(f"hidden target identities appear among sources: {sorted(leaked)}")

        coordinates_by_position: dict[int, tuple[float, float, float]] = {}
        for field, patches in (
            ("sources", self.sources),
            ("queries", self.queries),
            ("targets", self.targets),
        ):
            for patch in patches:
                known = coordinates_by_position.setdefault(patch.position_id, patch.center_mm)
                if known != patch.center_mm:
                    raise PatchPlanError(
                        f"position_id {patch.position_id} maps to inconsistent "
                        f"coordinates in {field}"
                    )

        physical_geometry = self.geometry.to_geometry()
        prism_lower = tuple(
            anchor - extent / 2.0
            for anchor, extent in zip(
                self.prism_anchor_mm,
                self.prism_extent_mm,
                strict=True,
            )
        )
        prism_upper = tuple(
            anchor + extent / 2.0
            for anchor, extent in zip(
                self.prism_anchor_mm,
                self.prism_extent_mm,
                strict=True,
            )
        )
        for patch in (*self.sources, *self.targets):
            patch_lower, patch_upper = physical_geometry.patch(patch.center_mm).bounds_mm
            if any(
                lower < outer_lower or upper > outer_upper
                for lower, upper, outer_lower, outer_upper in zip(
                    patch_lower,
                    patch_upper,
                    prism_lower,
                    prism_upper,
                    strict=True,
                )
            ):
                raise PatchPlanError(
                    f"patch {patch.key} is not fully contained by the stored prism"
                )

        slabs = tuple(physical_geometry.slab(target.center_mm) for target in self.targets)
        for index, first in enumerate(slabs):
            for second in slabs[index + 1 :]:
                if first.intersects(second):
                    raise PatchPlanError("query/target slabs must be pairwise non-overlapping")
        for source in self.sources:
            if source.modality_id != self.target_modality_id:
                continue
            source_slab = physical_geometry.slab(source.center_mm)
            if any(source_slab.intersects(target_slab) for target_slab in slabs):
                raise PatchPlanError(
                    "target-modality source footprints must not intersect target footprints"
                )

    def to_payload_dict(self) -> dict[str, object]:
        """Return every hashed field, excluding only the embedded payload digest."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "data_manifest_sha256": self.data_manifest_sha256,
            "case": self.case.to_dict(),
            "epoch": self.epoch,
            "bag_index": self.bag_index,
            "seed": self.seed,
            "modality_names": list(self.modality_names),
            "geometry": self.geometry.to_dict(),
            "geometry_sha256": self.geometry_sha256,
            "extraction_spec_sha256": self.extraction_spec_sha256,
            "prism_anchor_mm": list(self.prism_anchor_mm),
            "prism_extent_mm": list(self.prism_extent_mm),
            "target_modality_id": self.target_modality_id,
            "sources": [patch.to_dict() for patch in self.sources],
            "queries": [patch.to_dict() for patch in self.queries],
            "targets": [patch.to_dict() for patch in self.targets],
        }

    @property
    def sha256(self) -> str:
        """Digest of the full plan payload, used to pair objective arms."""

        return canonical_sha256(self.to_payload_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self.to_payload_dict(), "payload_sha256": self.sha256}

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MaterializedPatchPlan:
        expected = {
            "schema",
            "schema_version",
            "data_manifest_sha256",
            "case",
            "epoch",
            "bag_index",
            "seed",
            "modality_names",
            "geometry",
            "geometry_sha256",
            "extraction_spec_sha256",
            "prism_anchor_mm",
            "prism_extent_mm",
            "target_modality_id",
            "sources",
            "queries",
            "targets",
            "payload_sha256",
        }
        _exact_keys(value, expected, "materialized patch plan")
        case = _mapping(value["case"], "case")
        geometry = _mapping(value["geometry"], "geometry")
        modality_names = _array(value["modality_names"], "modality_names")
        prism_anchor_mm = _array(value["prism_anchor_mm"], "prism_anchor_mm")
        prism_extent_mm = _array(value["prism_extent_mm"], "prism_extent_mm")

        def patches(field: str) -> tuple[PatchIdentity, ...]:
            records = _array(value[field], field)
            return tuple(
                PatchIdentity.from_dict(_mapping(item, f"{field}[{index}]"))
                for index, item in enumerate(records)
            )

        plan = cls(
            schema=value["schema"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            data_manifest_sha256=value["data_manifest_sha256"],  # type: ignore[arg-type]
            case=PlanCaseIdentity.from_dict(case),
            epoch=value["epoch"],  # type: ignore[arg-type]
            bag_index=value["bag_index"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            modality_names=tuple(modality_names),  # type: ignore[arg-type]
            geometry=GeometryRecord.from_dict(geometry),
            geometry_sha256=value["geometry_sha256"],  # type: ignore[arg-type]
            extraction_spec_sha256=value["extraction_spec_sha256"],  # type: ignore[arg-type]
            prism_anchor_mm=tuple(prism_anchor_mm),  # type: ignore[arg-type]
            prism_extent_mm=tuple(prism_extent_mm),  # type: ignore[arg-type]
            target_modality_id=value["target_modality_id"],  # type: ignore[arg-type]
            sources=patches("sources"),
            queries=patches("queries"),
            targets=patches("targets"),
        )
        embedded_sha256 = _sha256(value["payload_sha256"], "payload_sha256")
        if embedded_sha256 != plan.sha256:
            raise PatchPlanError(
                f"patch-plan payload SHA mismatch: expected {embedded_sha256}, got {plan.sha256}"
            )
        return plan

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
        *,
        require_canonical: bool = True,
    ) -> MaterializedPatchPlan:
        value = _decode_json(payload)
        plan = cls.from_dict(_mapping(value, "materialized patch plan"))
        if require_canonical:
            raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
            canonical = canonical_json_bytes(plan.to_dict())
            if raw != canonical:
                raise PatchPlanError(
                    "patch-plan JSON is valid but not in the required canonical byte form"
                )
        return plan

    @classmethod
    def from_ordering_batch_plan(
        cls,
        batch_plan: object,
        *,
        data_manifest_sha256: str,
        source: str,
        release: str,
        case_id: str,
        subject_id: str,
        visit_id: str,
        epoch: int,
        bag_index: int,
        seed: int,
        extraction_spec_sha256: str,
    ) -> MaterializedPatchPlan:
        """Freeze a validated single-modality ordering plan into one replay record."""

        from .modality_completion import SingleModalityOrderingBatchPlan

        if not isinstance(batch_plan, SingleModalityOrderingBatchPlan):
            raise TypeError("batch_plan must be a SingleModalityOrderingBatchPlan")
        batch_plan.validate()
        modality_names = tuple(batch_plan.modality_names)

        def identity(patch: PatchMetadata) -> PatchIdentity:
            return PatchIdentity(
                position_id=patch.position_id,
                modality_id=patch.modality_id,
                modality=modality_names[patch.modality_id],
                center_mm=patch.center_mm,
            )

        geometry = GeometryRecord.from_geometry(batch_plan.geometry)
        targets = tuple(identity(target) for target in batch_plan.targets)
        return cls(
            data_manifest_sha256=data_manifest_sha256,
            case=PlanCaseIdentity(
                source=source,
                release=release,
                case_id=case_id,
                subject_id=subject_id,
                visit_id=visit_id,
            ),
            epoch=epoch,
            bag_index=bag_index,
            seed=seed,
            modality_names=modality_names,
            geometry=geometry,
            geometry_sha256=geometry.sha256,
            extraction_spec_sha256=extraction_spec_sha256,
            prism_anchor_mm=batch_plan.prism_anchor_mm,
            prism_extent_mm=batch_plan.prism_extent_mm,
            target_modality_id=batch_plan.target_modality_id,
            sources=tuple(identity(source_patch) for source_patch in batch_plan.sources),
            queries=targets,
            targets=targets,
        )

    @classmethod
    def from_batch_plan(
        cls,
        batch_plan: ModalityCompletionBatchPlan,
        **_kwargs: object,
    ) -> MaterializedPatchPlan:
        """Reject the superseded balanced leave-one-modality-out record contract."""

        if not isinstance(batch_plan, ModalityCompletionBatchPlan):
            raise TypeError("batch_plan must be a ModalityCompletionBatchPlan")
        raise PatchPlanError(
            "schema v2 accepts only SingleModalityOrderingBatchPlan via "
            "from_ordering_batch_plan"
        )


def save_patch_plan(
    plan: MaterializedPatchPlan,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> str:
    """Save canonical bytes, verify the write, and return the plan payload SHA."""

    if not isinstance(plan, MaterializedPatchPlan):
        raise TypeError("plan must be a MaterializedPatchPlan")
    destination = Path(path)
    payload = canonical_json_bytes(plan.to_dict())
    if overwrite:
        atomic_replace_bytes(destination, payload)
    else:
        atomic_create_bytes(destination, payload)
    load_patch_plan(destination, expected_sha256=plan.sha256)
    return plan.sha256


def load_patch_plan(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> MaterializedPatchPlan:
    """Strictly load canonical bytes and optionally enforce a pinned plan SHA."""

    plan = MaterializedPatchPlan.from_json(Path(path).read_bytes(), require_canonical=True)
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, "expected_sha256")
        if expected != plan.sha256:
            raise PatchPlanError(f"patch-plan SHA mismatch: expected {expected}, got {plan.sha256}")
    return plan


# ``PatchPlanRecord`` is the concise public name; the long name emphasizes that
# stochastic choices have already been materialized rather than being replayed.
PatchPlanRecord = MaterializedPatchPlan


__all__ = [
    "PATCH_PLAN_SCHEMA",
    "PATCH_PLAN_SCHEMA_VERSION",
    "GeometryRecord",
    "MaterializedPatchPlan",
    "PatchIdentity",
    "PatchPlanError",
    "PatchPlanRecord",
    "PlanCaseIdentity",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_patch_plan",
    "save_patch_plan",
]
