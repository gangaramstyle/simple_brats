"""Canonical dataset manifests with fail-closed BraTS MET identity handling."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1

# MET is deliberately the only inferred identity format.  Adding a new cohort
# requires adding an explicit, reviewed parser rather than guessing where a
# subject identifier ends and a visit identifier begins.
_MET_CASE_RE = re.compile(r"^(?P<subject>BraTS-MET-[0-9]{5})-(?P<visit>[0-9]{3})$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest is incomplete, inconsistent, or non-canonical."""


class IdentityError(ManifestError):
    """Raised when a case cannot be mapped unambiguously to subject and visit."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ManifestError(f"{field} must not be empty")
    if normalized != value:
        raise ManifestError(f"{field} must not contain surrounding whitespace")
    if any(ord(character) < 32 for character in normalized):
        raise ManifestError(f"{field} must not contain control characters")
    return normalized


@dataclass(frozen=True, order=True)
class CaseIdentity:
    """Unambiguous longitudinal identity for one imaging visit."""

    case_id: str
    subject_id: str
    visit_id: str

    def __post_init__(self) -> None:
        _required_text(self.case_id, "case_id")
        _required_text(self.subject_id, "subject_id")
        _required_text(self.visit_id, "visit_id")


def canonicalize_case_identity(
    case_id: str,
    *,
    subject_id: str | None = None,
    visit_id: str | None = None,
) -> CaseIdentity:
    """Return canonical subject/visit identity for a BraTS MET case.

    ``BraTS-MET-00730-001`` maps to subject ``BraTS-MET-00730`` and visit
    ``001``.  Unsupported case formats are never heuristically split.  They
    require both ``subject_id`` and ``visit_id`` as explicit metadata.  A
    partial override is rejected because it leaves the other field's
    provenance ambiguous.
    """

    case_id = _required_text(case_id, "case_id")
    if (subject_id is None) != (visit_id is None):
        raise IdentityError(
            "subject_id and visit_id must either both be explicit or both be inferred"
        )

    match = _MET_CASE_RE.fullmatch(case_id)
    if match is not None:
        inferred_subject = match.group("subject")
        inferred_visit = match.group("visit")
        if subject_id is not None and visit_id is not None:
            explicit_subject = _required_text(subject_id, "subject_id")
            explicit_visit = _required_text(visit_id, "visit_id")
            if (explicit_subject, explicit_visit) != (inferred_subject, inferred_visit):
                raise IdentityError(
                    f"explicit identity for canonical MET case {case_id!r} conflicts "
                    "with its required subject/visit mapping"
                )
        return CaseIdentity(
            case_id=case_id,
            subject_id=inferred_subject,
            visit_id=inferred_visit,
        )

    if subject_id is None or visit_id is None:
        raise IdentityError(
            f"unsupported or ambiguous case_id {case_id!r}; provide explicit "
            "subject_id and visit_id metadata"
        )
    return CaseIdentity(
        case_id=case_id,
        subject_id=_required_text(subject_id, "subject_id"),
        visit_id=_required_text(visit_id, "visit_id"),
    )


@dataclass(frozen=True, order=True)
class FileRecord:
    """One modality file and its content-addressed identity."""

    modality: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        modality = _required_text(self.modality, "modality")
        path = _required_text(self.path, "path")
        digest = _required_text(self.sha256, "sha256").lower()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ManifestError("sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_path(
        cls,
        modality: str,
        path: str | os.PathLike[str],
        *,
        recorded_path: str | None = None,
    ) -> FileRecord:
        """Hash ``path`` and construct a record.

        ``recorded_path`` can be a stable dataset-relative path while ``path``
        points to the local mount used to calculate the digest.
        """

        filesystem_path = Path(path)
        return cls(
            modality=modality,
            path=recorded_path if recorded_path is not None else os.fspath(path),
            sha256=sha256_file(filesystem_path),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "modality": self.modality,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FileRecord:
        _require_exact_keys(value, {"modality", "path", "sha256"}, "file record")
        return cls(
            modality=value["modality"],  # type: ignore[arg-type]
            path=value["path"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CaseRecord:
    """One case/timepoint in a dataset release."""

    source: str
    release: str
    case_id: str
    subject_id: str
    visit_id: str
    modalities: tuple[str, ...]
    files: tuple[FileRecord, ...]

    def __post_init__(self) -> None:
        source = _required_text(self.source, "source")
        release = _required_text(self.release, "release")
        identity = canonicalize_case_identity(
            self.case_id,
            subject_id=self.subject_id,
            visit_id=self.visit_id,
        )

        modalities = tuple(_required_text(item, "modality") for item in self.modalities)
        if not modalities:
            raise ManifestError("a case must contain at least one modality")
        if len(set(modalities)) != len(modalities):
            raise ManifestError("modalities must be unique within a case")

        files = tuple(self.files)
        if not files:
            raise ManifestError("a case must contain at least one file")
        if not all(isinstance(item, FileRecord) for item in files):
            raise ManifestError("files must contain FileRecord instances")
        file_modalities = [item.modality for item in files]
        if len(set(file_modalities)) != len(file_modalities):
            raise ManifestError("each modality must have exactly one file record")
        if set(modalities) != set(file_modalities):
            raise ManifestError("modalities must exactly match the modalities represented by files")
        paths = [item.path for item in files]
        if len(set(paths)) != len(paths):
            raise ManifestError("file paths must be unique within a case")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "release", release)
        object.__setattr__(self, "case_id", identity.case_id)
        object.__setattr__(self, "subject_id", identity.subject_id)
        object.__setattr__(self, "visit_id", identity.visit_id)
        object.__setattr__(self, "modalities", tuple(sorted(modalities)))
        object.__setattr__(
            self, "files", tuple(sorted(files, key=lambda item: (item.modality, item.path)))
        )

    @classmethod
    def create(
        cls,
        *,
        source: str,
        release: str,
        case_id: str,
        files: Iterable[FileRecord],
        subject_id: str | None = None,
        visit_id: str | None = None,
    ) -> CaseRecord:
        """Construct a case while inferring only the reviewed MET ID format."""

        identity = canonicalize_case_identity(case_id, subject_id=subject_id, visit_id=visit_id)
        file_records = tuple(files)
        return cls(
            source=source,
            release=release,
            case_id=identity.case_id,
            subject_id=identity.subject_id,
            visit_id=identity.visit_id,
            modalities=tuple(item.modality for item in file_records),
            files=file_records,
        )

    @property
    def key(self) -> tuple[str, str, str]:
        """Release-qualified case key."""

        return (self.source, self.release, self.case_id)

    @property
    def file_digests(self) -> frozenset[str]:
        return frozenset(item.sha256 for item in self.files)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "release": self.release,
            "case_id": self.case_id,
            "subject_id": self.subject_id,
            "visit_id": self.visit_id,
            "modalities": list(self.modalities),
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CaseRecord:
        expected = {
            "source",
            "release",
            "case_id",
            "subject_id",
            "visit_id",
            "modalities",
            "files",
        }
        _require_exact_keys(value, expected, "case record")
        modalities = value["modalities"]
        files = value["files"]
        if not isinstance(modalities, list) or not isinstance(files, list):
            raise ManifestError("case modalities and files must be JSON arrays")
        if not all(isinstance(item, Mapping) for item in files):
            raise ManifestError("each case file must be a JSON object")
        return cls(
            source=value["source"],  # type: ignore[arg-type]
            release=value["release"],  # type: ignore[arg-type]
            case_id=value["case_id"],  # type: ignore[arg-type]
            subject_id=value["subject_id"],  # type: ignore[arg-type]
            visit_id=value["visit_id"],  # type: ignore[arg-type]
            modalities=tuple(modalities),  # type: ignore[arg-type]
            files=tuple(FileRecord.from_dict(item) for item in files),
        )


@dataclass(frozen=True)
class DatasetManifest:
    """Canonical collection of cases consumed by an experiment."""

    cases: tuple[CaseRecord, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported manifest schema_version {self.schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        cases = tuple(self.cases)
        if not cases:
            raise ManifestError("a dataset manifest must contain at least one case")
        if not all(isinstance(item, CaseRecord) for item in cases):
            raise ManifestError("cases must contain CaseRecord instances")

        keys = [item.key for item in cases]
        if len(set(keys)) != len(keys):
            raise ManifestError("duplicate source/release/case_id entries are not allowed")

        # A full case ID must not silently change longitudinal identity between
        # releases.  Explicit aliases should be canonicalized before manifest
        # construction rather than allowing release-dependent mappings.
        identities: dict[str, tuple[str, str]] = {}
        for case in cases:
            identity = (case.subject_id, case.visit_id)
            previous = identities.setdefault(case.case_id, identity)
            if previous != identity:
                raise ManifestError(
                    f"case_id {case.case_id!r} maps to multiple subject/visit identities"
                )

        object.__setattr__(
            self,
            "cases",
            tuple(
                sorted(
                    cases,
                    key=lambda item: (
                        item.subject_id,
                        item.visit_id,
                        item.source,
                        item.release,
                        item.case_id,
                    ),
                )
            ),
        )

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({case.subject_id for case in self.cases}))

    @property
    def file_digests(self) -> frozenset[str]:
        return frozenset(file.sha256 for case in self.cases for file in case.files)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON payload (without a self-hash field)."""

        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetManifest:
        _require_exact_keys(value, {"schema_version", "cases"}, "dataset manifest")
        cases = value["cases"]
        if not isinstance(cases, list):
            raise ManifestError("manifest cases must be a JSON array")
        if not all(isinstance(item, Mapping) for item in cases):
            raise ManifestError("each manifest case must be a JSON object")
        schema_version = value["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestError("schema_version must be an integer")
        return cls(
            schema_version=schema_version,
            cases=tuple(CaseRecord.from_dict(item) for item in cases),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> DatasetManifest:
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ManifestError(f"invalid manifest JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise ManifestError("dataset manifest must be a JSON object")
        return cls.from_dict(value)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically for hashing and provenance records."""

    if is_dataclass(value):
        value = asdict(value)
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(f"value is not canonical-JSON serializable: {error}") from error
    return payload.encode("utf-8")


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading an image volume into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(manifest: DatasetManifest, path: str | os.PathLike[str]) -> None:
    """Write exactly the canonical bytes used by ``manifest.sha256``."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    Path(path).write_bytes(canonical_json_bytes(manifest.to_dict()))


def load_manifest(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> DatasetManifest:
    """Load a manifest and optionally fail if its canonical SHA changed."""

    manifest = DatasetManifest.from_json(Path(path).read_bytes())
    if expected_sha256 is not None:
        expected_sha256 = _required_text(expected_sha256, "expected_sha256").lower()
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ManifestError("expected_sha256 is not a SHA-256 digest")
        if manifest.sha256 != expected_sha256:
            raise ManifestError(
                f"manifest SHA mismatch: expected {expected_sha256}, got {manifest.sha256}"
            )
    return manifest


def _require_exact_keys(value: Mapping[str, object], expected: set[str], description: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ManifestError(f"invalid {description}: " + "; ".join(details))


# Short alias for callers that prefer ``Manifest``; the descriptive name stays
# canonical in serialized provenance and documentation.
Manifest = DatasetManifest
