"""Deterministic, leakage-safe subject-level dataset splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .manifest import (
    CaseRecord,
    DatasetManifest,
    canonical_json_bytes,
)

SPLIT_SCHEMA_VERSION = 1
SPLIT_STRATEGY = "sha256-canonical-subject-v1"
_HASH_SPACE_SIZE = 1 << 256


class SplitError(ValueError):
    """Raised when split provenance is incomplete or incompatible."""


class SplitLeakageError(SplitError):
    """Raised when subjects or file contents cross a protected boundary."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SplitError(f"{field} must be a non-empty string without outer whitespace")
    return value


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool):
        raise SplitError(f"{field} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise SplitError(f"{field} must be a finite decimal") from error
    if not result.is_finite():
        raise SplitError(f"{field} must be finite")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True, order=True)
class SplitFraction:
    name: str
    fraction: str

    def __post_init__(self) -> None:
        name = _required_text(self.name, "split name")
        if any(character in name for character in ("/", "\\", "\0")):
            raise SplitError("split name must not contain path separators or NUL")
        fraction = _decimal(self.fraction, f"fraction for {name!r}")
        if fraction <= 0 or fraction >= 1:
            raise SplitError("each split fraction must be greater than 0 and less than 1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fraction", _decimal_text(fraction))

    @property
    def decimal(self) -> Decimal:
        return Decimal(self.fraction)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "fraction": self.fraction}


@dataclass(frozen=True, order=True)
class SubjectAssignment:
    subject_id: str
    split: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _required_text(self.subject_id, "subject_id"))
        object.__setattr__(self, "split", _required_text(self.split, "split"))

    def to_dict(self) -> dict[str, str]:
        return {"subject_id": self.subject_id, "split": self.split}


@dataclass(frozen=True)
class SplitManifest:
    """Subject assignments tied to one exact dataset manifest."""

    manifest_sha256: str
    seed: int
    fractions: tuple[SplitFraction, ...]
    assignments: tuple[SubjectAssignment, ...]
    strategy: str = SPLIT_STRATEGY
    schema_version: int = SPLIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_SCHEMA_VERSION:
            raise SplitError(
                f"unsupported split schema_version {self.schema_version!r}; "
                f"expected {SPLIT_SCHEMA_VERSION}"
            )
        if self.strategy != SPLIT_STRATEGY:
            raise SplitError(
                f"unsupported split strategy {self.strategy!r}; expected {SPLIT_STRATEGY!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SplitError("split seed must be an integer")
        manifest_sha256 = _required_text(self.manifest_sha256, "manifest_sha256").lower()
        if len(manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_sha256
        ):
            raise SplitError("manifest_sha256 must be a SHA-256 digest")

        fractions = tuple(self.fractions)
        if len(fractions) < 2:
            raise SplitError("at least two split fractions are required")
        if not all(isinstance(item, SplitFraction) for item in fractions):
            raise SplitError("fractions must contain SplitFraction instances")
        names = [item.name for item in fractions]
        if len(set(names)) != len(names):
            raise SplitError("split names must be unique")
        if sum((item.decimal for item in fractions), Decimal(0)) != Decimal(1):
            raise SplitError("split fractions must sum exactly to 1")

        assignments = tuple(self.assignments)
        if not assignments:
            raise SplitError("a split manifest must contain subject assignments")
        if not all(isinstance(item, SubjectAssignment) for item in assignments):
            raise SplitError("assignments must contain SubjectAssignment instances")
        subjects = [item.subject_id for item in assignments]
        if len(set(subjects)) != len(subjects):
            raise SplitError("each subject must have exactly one assignment")
        unknown = sorted({item.split for item in assignments} - set(names))
        if unknown:
            raise SplitError(f"assignments reference unknown splits: {unknown}")

        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "fractions", fractions)
        object.__setattr__(
            self, "assignments", tuple(sorted(assignments, key=lambda item: item.subject_id))
        )

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.fractions)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def split_of(self, subject_id: str) -> str:
        subject_id = _required_text(subject_id, "subject_id")
        for assignment in self.assignments:
            if assignment.subject_id == subject_id:
                return assignment.split
        raise SplitError(f"subject {subject_id!r} is absent from the split manifest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "manifest_sha256": self.manifest_sha256,
            "seed": self.seed,
            "fractions": [item.to_dict() for item in self.fractions],
            "assignments": [item.to_dict() for item in self.assignments],
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SplitManifest:
        expected = {
            "schema_version",
            "strategy",
            "manifest_sha256",
            "seed",
            "fractions",
            "assignments",
        }
        actual = set(value)
        if actual != expected:
            raise SplitError(
                f"invalid split manifest keys: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        raw_fractions = value["fractions"]
        raw_assignments = value["assignments"]
        if not isinstance(raw_fractions, list) or not isinstance(raw_assignments, list):
            raise SplitError("fractions and assignments must be JSON arrays")
        for item in raw_fractions:
            if not isinstance(item, Mapping) or set(item) != {"name", "fraction"}:
                raise SplitError("each fraction must be an object with exactly name and fraction")
        for item in raw_assignments:
            if not isinstance(item, Mapping) or set(item) != {"subject_id", "split"}:
                raise SplitError(
                    "each assignment must be an object with exactly subject_id and split"
                )
        try:
            fractions = tuple(
                SplitFraction(name=item["name"], fraction=item["fraction"])
                for item in raw_fractions
            )
            assignments = tuple(
                SubjectAssignment(subject_id=item["subject_id"], split=item["split"])
                for item in raw_assignments
            )
        except KeyError as error:
            raise SplitError(f"invalid split entry: missing {error.args[0]!r}") from error
        schema_version = value["schema_version"]
        seed = value["seed"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise SplitError("schema_version must be an integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise SplitError("seed must be an integer")
        return cls(
            schema_version=schema_version,
            strategy=value["strategy"],  # type: ignore[arg-type]
            manifest_sha256=value["manifest_sha256"],  # type: ignore[arg-type]
            seed=seed,
            fractions=fractions,
            assignments=assignments,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> SplitManifest:
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise SplitError(f"invalid split JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise SplitError("split manifest must be a JSON object")
        return cls.from_dict(value)


def _normalize_fractions(
    fractions: Mapping[str, object] | Sequence[tuple[str, object]],
) -> tuple[SplitFraction, ...]:
    items = fractions.items() if isinstance(fractions, Mapping) else fractions
    result = tuple(SplitFraction(name=name, fraction=str(value)) for name, value in items)
    # SplitManifest performs the definitive validation; validate here to fail
    # before hashing any subjects and to preserve the caller's split order.
    if len(result) < 2:
        raise SplitError("at least two split fractions are required")
    if len({item.name for item in result}) != len(result):
        raise SplitError("split names must be unique")
    if sum((item.decimal for item in result), Decimal(0)) != Decimal(1):
        raise SplitError("split fractions must sum exactly to 1")
    return result


def _integer_weights(fractions: tuple[SplitFraction, ...]) -> tuple[list[int], int]:
    decimal_places = max(max(0, -item.decimal.as_tuple().exponent) for item in fractions)
    denominator = 10**decimal_places
    weights = [int(item.decimal * denominator) for item in fractions]
    if sum(weights) != denominator:
        raise SplitError("split fractions could not be represented exactly")
    return weights, denominator


def _assign_subject(
    subject_id: str,
    *,
    seed: int,
    fractions: tuple[SplitFraction, ...],
) -> str:
    payload = f"simple_brats:{SPLIT_STRATEGY}\0{seed}\0{subject_id}".encode()
    point = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    weights, denominator = _integer_weights(fractions)
    bucket = (point * denominator) // _HASH_SPACE_SIZE
    cumulative = 0
    for item, weight in zip(fractions, weights, strict=True):
        cumulative += weight
        if bucket < cumulative:
            return item.name
    raise AssertionError("hash bucket was outside the normalized fraction range")


def create_subject_split(
    manifest: DatasetManifest,
    *,
    seed: int = 0,
    fractions: Mapping[str, object] | Sequence[tuple[str, object]] = (
        ("train", "0.8"),
        ("validation", "0.1"),
        ("test", "0.1"),
    ),
) -> SplitManifest:
    """Hash canonical subject IDs so all visits/releases stay together."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SplitError("split seed must be an integer")
    normalized = _normalize_fractions(fractions)
    assignments = tuple(
        SubjectAssignment(
            subject_id=subject,
            split=_assign_subject(subject, seed=seed, fractions=normalized),
        )
        for subject in manifest.subjects
    )
    split = SplitManifest(
        manifest_sha256=manifest.sha256,
        seed=seed,
        fractions=normalized,
        assignments=assignments,
    )
    validate_split(manifest, split)
    return split


def partition_cases(
    manifest: DatasetManifest, split: SplitManifest
) -> dict[str, tuple[CaseRecord, ...]]:
    """Partition cases after verifying exact manifest and subject coverage."""

    if manifest.sha256 != split.manifest_sha256:
        raise SplitError(
            "split manifest was created for a different dataset manifest: "
            f"expected {manifest.sha256}, got {split.manifest_sha256}"
        )
    manifest_subjects = set(manifest.subjects)
    assigned_subjects = {item.subject_id for item in split.assignments}
    if manifest_subjects != assigned_subjects:
        raise SplitError(
            "split subject coverage mismatch: "
            f"missing={sorted(manifest_subjects - assigned_subjects)}, "
            f"unexpected={sorted(assigned_subjects - manifest_subjects)}"
        )
    assignment_map = {item.subject_id: item.split for item in split.assignments}
    result: dict[str, list[CaseRecord]] = {name: [] for name in split.split_names}
    for case in manifest.cases:
        result[assignment_map[case.subject_id]].append(case)
    return {name: tuple(cases) for name, cases in result.items()}


def assert_subject_disjointness(
    partitions: Mapping[str, Iterable[CaseRecord]],
) -> None:
    """Fail if a canonical subject occurs in more than one partition."""

    owners: dict[str, str] = {}
    collisions: dict[str, set[str]] = {}
    for partition, cases in partitions.items():
        partition = _required_text(partition, "partition name")
        for case in cases:
            previous = owners.setdefault(case.subject_id, partition)
            if previous != partition:
                collisions.setdefault(case.subject_id, {previous}).add(partition)
    if collisions:
        details = ", ".join(
            f"{subject} in {sorted(names)}" for subject, names in sorted(collisions.items())
        )
        raise SplitLeakageError(f"subject leakage across partitions: {details}")


def assert_digest_disjointness(
    partitions: Mapping[str, Iterable[CaseRecord]],
) -> None:
    """Fail if identical file bytes occur in more than one partition."""

    owners: dict[str, str] = {}
    collisions: dict[str, set[str]] = {}
    for partition, cases in partitions.items():
        partition = _required_text(partition, "partition name")
        for case in cases:
            for file in case.files:
                previous = owners.setdefault(file.sha256, partition)
                if previous != partition:
                    collisions.setdefault(file.sha256, {previous}).add(partition)
    if collisions:
        details = ", ".join(
            f"{digest} in {sorted(names)}" for digest, names in sorted(collisions.items())
        )
        raise SplitLeakageError(f"file-digest leakage across partitions: {details}")


def validate_split(manifest: DatasetManifest, split: SplitManifest) -> None:
    """Assert manifest binding, full coverage, and both leakage boundaries."""

    partitions = partition_cases(manifest, split)
    empty = sorted(name for name, cases in partitions.items() if not cases)
    if empty:
        raise SplitError(f"declared splits contain no cases: {empty}")
    assert_subject_disjointness(partitions)
    assert_digest_disjointness(partitions)


def cases_for_splits(
    manifest: DatasetManifest,
    split: SplitManifest,
    names: Iterable[str],
) -> tuple[CaseRecord, ...]:
    """Select cases from named partitions, failing on misspelled/empty names."""

    partitions = partition_cases(manifest, split)
    requested = tuple(_required_text(name, "split name") for name in names)
    if not requested:
        raise SplitError("at least one split name must be selected")
    if len(set(requested)) != len(requested):
        raise SplitError("selected split names must be unique")
    unknown = sorted(set(requested) - set(partitions))
    if unknown:
        raise SplitError(f"unknown split names: {unknown}")
    return tuple(case for name in requested for case in partitions[name])


@dataclass(frozen=True)
class CompatibilityReport:
    """Subject and byte-level overlap between training and evaluation data."""

    overlapping_subjects: tuple[str, ...]
    overlapping_digests: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.overlapping_subjects and not self.overlapping_digests

    def assert_compatible(self, *, context: str = "training/evaluation") -> None:
        if self.compatible:
            return
        parts: list[str] = []
        if self.overlapping_subjects:
            parts.append(f"subjects={list(self.overlapping_subjects)}")
        if self.overlapping_digests:
            parts.append(f"file_digests={list(self.overlapping_digests)}")
        raise SplitLeakageError(f"{context} data overlap: " + "; ".join(parts))


def compatibility_report(
    training_cases: Iterable[CaseRecord],
    evaluation_cases: Iterable[CaseRecord],
) -> CompatibilityReport:
    """Return exact overlap evidence without mutating either collection."""

    training = tuple(training_cases)
    evaluation = tuple(evaluation_cases)
    training_subjects = {case.subject_id for case in training}
    evaluation_subjects = {case.subject_id for case in evaluation}
    training_digests = {file.sha256 for case in training for file in case.files}
    evaluation_digests = {file.sha256 for case in evaluation for file in case.files}
    return CompatibilityReport(
        overlapping_subjects=tuple(sorted(training_subjects & evaluation_subjects)),
        overlapping_digests=tuple(sorted(training_digests & evaluation_digests)),
    )


def assert_evaluation_compatible(
    training_cases: Iterable[CaseRecord],
    evaluation_cases: Iterable[CaseRecord],
    *,
    context: str = "training/evaluation",
) -> None:
    """Fail if any subject or file content was seen by both sides."""

    compatibility_report(training_cases, evaluation_cases).assert_compatible(context=context)


def assert_warm_start_compatible(
    warm_start_manifest: DatasetManifest,
    evaluation_manifest: DatasetManifest,
    *,
    warm_start_split: SplitManifest | None = None,
    warm_start_splits: Iterable[str] = ("train",),
    evaluation_split: SplitManifest | None = None,
    evaluation_splits: Iterable[str] = ("validation", "test"),
) -> None:
    """Ensure a warm-start checkpoint could not have consumed evaluation data.

    Each manifest must describe data that was available to the corresponding
    run.  When a split manifest is supplied, only the named partitions are
    considered consumed/evaluated; without one, all cases are conservatively
    selected.  This conservative default prevents an omitted split provenance
    record from silently approving a contaminated checkpoint.
    """

    if warm_start_split is None:
        warm_cases = warm_start_manifest.cases
    else:
        validate_split(warm_start_manifest, warm_start_split)
        warm_cases = cases_for_splits(warm_start_manifest, warm_start_split, warm_start_splits)

    if evaluation_split is None:
        eval_cases = evaluation_manifest.cases
    else:
        validate_split(evaluation_manifest, evaluation_split)
        eval_cases = cases_for_splits(evaluation_manifest, evaluation_split, evaluation_splits)

    assert_evaluation_compatible(warm_cases, eval_cases, context="warm-start/evaluation")


def save_split(split: SplitManifest, path: str | Path) -> None:
    Path(path).write_bytes(canonical_json_bytes(split.to_dict()))


def load_split(path: str | Path, *, expected_sha256: str | None = None) -> SplitManifest:
    split = SplitManifest.from_json(Path(path).read_bytes())
    if expected_sha256 is not None and split.sha256 != expected_sha256.lower():
        raise SplitError(
            f"split SHA mismatch: expected {expected_sha256.lower()}, got {split.sha256}"
        )
    return split
