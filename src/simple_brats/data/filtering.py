"""Content-addressed, auditable subject quarantine before dataset splitting.

A filter specification is deliberately separate from the input and output
dataset manifests.  It records why a canonical subject was removed, cites the
exact file digest(s) used as evidence, and is bound to one immutable input
manifest SHA.  Applying a stale or fabricated specification therefore fails
closed before a split can be created.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .manifest import DatasetManifest, canonical_json_bytes

MANIFEST_FILTER_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestFilterError(ValueError):
    """Raised when quarantine provenance is incomplete or incompatible."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ManifestFilterError(f"{field} must be a string")
    if not value or value != value.strip():
        raise ManifestFilterError(
            f"{field} must be non-empty and contain no surrounding whitespace"
        )
    if any(ord(character) < 32 for character in value):
        raise ManifestFilterError(f"{field} must not contain control characters")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _required_text(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ManifestFilterError(f"{field} must be a lowercase SHA-256 digest")
    return digest


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
    raise ManifestFilterError(f"invalid {description}: " + "; ".join(details))


def _decode_json(payload: str | bytes | bytearray) -> object:
    if not isinstance(payload, (str, bytes, bytearray)):
        raise ManifestFilterError("manifest-filter JSON must be str, bytes, or bytearray")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestFilterError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> object:
        raise ManifestFilterError(f"non-finite JSON number {token!r} is forbidden")

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except ManifestFilterError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ManifestFilterError(f"invalid manifest-filter JSON: {error}") from error


@dataclass(frozen=True, order=True, slots=True)
class SubjectExclusion:
    """One canonical subject removal and the manifest-attached evidence for it."""

    subject_id: str
    reason: str
    evidence_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        subject_id = _required_text(self.subject_id, "exclusion.subject_id")
        reason = _required_text(self.reason, "exclusion.reason")
        try:
            evidence = tuple(self.evidence_sha256)
        except TypeError as error:
            raise ManifestFilterError(
                "exclusion.evidence_sha256 must be an array of SHA-256 digests"
            ) from error
        if not evidence:
            raise ManifestFilterError("an exclusion must cite at least one evidence digest")
        digests = tuple(
            _sha256(item, f"exclusion.evidence_sha256[{index}]")
            for index, item in enumerate(evidence)
        )
        if len(set(digests)) != len(digests):
            raise ManifestFilterError("evidence digests must be unique within an exclusion")

        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "evidence_sha256", tuple(sorted(digests)))

    @property
    def evidence_digests(self) -> tuple[str, ...]:
        """Descriptive alias for the canonical ``evidence_sha256`` field."""

        return self.evidence_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "reason": self.reason,
            "evidence_sha256": list(self.evidence_sha256),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SubjectExclusion:
        _exact_keys(
            value,
            {"subject_id", "reason", "evidence_sha256"},
            "subject exclusion",
        )
        raw_evidence = value["evidence_sha256"]
        if not isinstance(raw_evidence, list):
            raise ManifestFilterError("exclusion.evidence_sha256 must be a JSON array")
        return cls(
            subject_id=value["subject_id"],  # type: ignore[arg-type]
            reason=value["reason"],  # type: ignore[arg-type]
            evidence_sha256=tuple(raw_evidence),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ManifestFilterSpec:
    """Canonical subject quarantine bound to one exact input manifest."""

    input_manifest_sha256: str
    exclusions: tuple[SubjectExclusion, ...]
    schema_version: int = MANIFEST_FILTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise ManifestFilterError("schema_version must be an integer")
        if self.schema_version != MANIFEST_FILTER_SCHEMA_VERSION:
            raise ManifestFilterError(
                f"unsupported manifest-filter schema_version {self.schema_version!r}; "
                f"expected {MANIFEST_FILTER_SCHEMA_VERSION}"
            )
        input_manifest_sha256 = _sha256(self.input_manifest_sha256, "input_manifest_sha256")
        try:
            exclusions = tuple(self.exclusions)
        except TypeError as error:
            raise ManifestFilterError("exclusions must be an array") from error
        if not all(isinstance(item, SubjectExclusion) for item in exclusions):
            raise ManifestFilterError("exclusions must contain SubjectExclusion instances")
        subjects = [item.subject_id for item in exclusions]
        if len(set(subjects)) != len(subjects):
            raise ManifestFilterError("each subject may be excluded at most once")

        object.__setattr__(self, "input_manifest_sha256", input_manifest_sha256)
        object.__setattr__(
            self,
            "exclusions",
            tuple(sorted(exclusions, key=lambda item: item.subject_id)),
        )

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical specification JSON."""

        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @property
    def excluded_subject_ids(self) -> tuple[str, ...]:
        return tuple(item.subject_id for item in self.exclusions)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_manifest_sha256": self.input_manifest_sha256,
            "exclusions": [item.to_dict() for item in self.exclusions],
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ManifestFilterSpec:
        _exact_keys(
            value,
            {"schema_version", "input_manifest_sha256", "exclusions"},
            "manifest-filter specification",
        )
        raw_exclusions = value["exclusions"]
        if not isinstance(raw_exclusions, list):
            raise ManifestFilterError("exclusions must be a JSON array")
        if not all(isinstance(item, Mapping) for item in raw_exclusions):
            raise ManifestFilterError("each exclusion must be a JSON object")
        schema_version = value["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestFilterError("schema_version must be an integer")
        return cls(
            schema_version=schema_version,
            input_manifest_sha256=value["input_manifest_sha256"],  # type: ignore[arg-type]
            exclusions=tuple(SubjectExclusion.from_dict(item) for item in raw_exclusions),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ManifestFilterSpec:
        value = _decode_json(payload)
        if not isinstance(value, Mapping):
            raise ManifestFilterError("manifest-filter specification must be a JSON object")
        return cls.from_dict(value)


def apply_manifest_filter(
    manifest: DatasetManifest,
    spec: ManifestFilterSpec,
) -> DatasetManifest:
    """Remove every visit of explicitly excluded subjects, failing closed.

    Evidence is checked against the file SHA-256 identities attached to any
    case/timepoint belonging to the excluded canonical subject.
    """

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if not isinstance(spec, ManifestFilterSpec):
        raise TypeError("spec must be a ManifestFilterSpec")
    if manifest.sha256 != spec.input_manifest_sha256:
        raise ManifestFilterError(
            "input manifest SHA mismatch: "
            f"spec expects {spec.input_manifest_sha256}, got {manifest.sha256}"
        )

    evidence_by_subject: dict[str, set[str]] = {}
    for case in manifest.cases:
        evidence_by_subject.setdefault(case.subject_id, set()).update(case.file_digests)

    unknown_subjects = sorted(set(spec.excluded_subject_ids) - set(evidence_by_subject))
    if unknown_subjects:
        raise ManifestFilterError(
            f"exclusions reference subjects absent from the input manifest: {unknown_subjects}"
        )

    for exclusion in spec.exclusions:
        unattached = sorted(
            set(exclusion.evidence_sha256) - evidence_by_subject[exclusion.subject_id]
        )
        if unattached:
            raise ManifestFilterError(
                f"evidence digests are not attached to subject {exclusion.subject_id!r}: "
                f"{unattached}"
            )

    excluded = set(spec.excluded_subject_ids)
    remaining = tuple(case for case in manifest.cases if case.subject_id not in excluded)
    if not remaining:
        raise ManifestFilterError("manifest filtering would remove every case")
    return DatasetManifest(cases=remaining, schema_version=manifest.schema_version)


def save_manifest_filter_spec(
    spec: ManifestFilterSpec,
    path: str | os.PathLike[str],
) -> None:
    """Write exactly the canonical bytes hashed by ``spec.sha256``."""

    if not isinstance(spec, ManifestFilterSpec):
        raise TypeError("spec must be a ManifestFilterSpec")
    Path(path).write_bytes(canonical_json_bytes(spec.to_dict()))


def load_manifest_filter_spec(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> ManifestFilterSpec:
    """Load a filter specification and optionally verify its canonical SHA."""

    spec = ManifestFilterSpec.from_json(Path(path).read_bytes())
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, "expected_sha256")
        if spec.sha256 != expected:
            raise ManifestFilterError(
                f"manifest-filter SHA mismatch: expected {expected}, got {spec.sha256}"
            )
    return spec


# Concise aliases for callers that already carry the manifest-filter context.
apply_filter = apply_manifest_filter
save_filter_spec = save_manifest_filter_spec
load_filter_spec = load_manifest_filter_spec


__all__ = [
    "MANIFEST_FILTER_SCHEMA_VERSION",
    "ManifestFilterError",
    "ManifestFilterSpec",
    "SubjectExclusion",
    "apply_filter",
    "apply_manifest_filter",
    "load_filter_spec",
    "load_manifest_filter_spec",
    "save_filter_spec",
    "save_manifest_filter_spec",
]
