"""Strict filesystem discovery for BraTS MET release images."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

from .manifest import CaseRecord, DatasetManifest, FileRecord, ManifestError

REQUIRED_MET_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
OPTIONAL_MET_MODALITIES = ("seg",)

_MET_CASE_DIR_RE = re.compile(r"^BraTS-MET-[0-9]{5}-[0-9]{3}$")
_NIFTI_SUFFIXES = (".nii", ".nii.gz")


class DiscoveryError(ManifestError):
    """Raised when a release tree does not match the strict MET layout."""


def discover_met_release(
    root: str | os.PathLike[str],
    *,
    source: str,
    release: str,
) -> DatasetManifest:
    """Discover canonical MET cases recursively beneath ``root``.

    A supplied root may contain case directories directly or beneath benign
    release-level nesting.  Every canonical case directory must contain the
    four required images using the exact
    ``<case-id>-<modality>.nii.gz`` filename.  ``seg`` is optional.  Release
    identity is always taken from the explicit ``source`` and ``release``
    arguments, never inferred from a directory name.

    Recorded file paths are POSIX paths relative to the resolved supplied
    root, so the manifest is stable across mount points.  Symlinks are rejected
    rather than followed, and every resolved case and image path is checked to
    remain beneath that root.
    """

    root_path = Path(root).expanduser()
    if root_path.is_symlink():
        raise DiscoveryError(f"dataset root must not be a symlink: {root_path}")
    try:
        resolved_root = root_path.resolve(strict=True)
    except OSError as error:
        raise DiscoveryError(f"dataset root is unavailable: {root_path}") from error
    if not resolved_root.is_dir():
        raise DiscoveryError(f"dataset root is not a directory: {root_path}")

    case_directories: dict[str, Path] = {}
    for directory in _walk_directories_without_symlinks(resolved_root):
        case_id = directory.name
        if _MET_CASE_DIR_RE.fullmatch(case_id) is None:
            continue
        _assert_within_root(directory, resolved_root, description="case directory")
        previous = case_directories.get(case_id)
        if previous is not None:
            raise DiscoveryError(
                f"duplicate case directory for {case_id!r}: "
                f"{_display_path(previous, resolved_root)} and "
                f"{_display_path(directory, resolved_root)}"
            )
        case_directories[case_id] = directory

    if not case_directories:
        raise DiscoveryError(f"no canonical BraTS MET case directories found under {root_path}")

    cases = tuple(
        _case_record(
            case_directories[case_id],
            root=resolved_root,
            source=source,
            release=release,
        )
        for case_id in sorted(case_directories)
    )
    return DatasetManifest(cases=cases)


def _walk_directories_without_symlinks(root: Path) -> Iterator[Path]:
    """Yield descendants while rejecting links anywhere in the scanned tree."""

    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        _assert_within_root(current_path, root, description="directory")

        for name in sorted(directory_names):
            path = current_path / name
            if path.is_symlink():
                raise DiscoveryError(
                    f"symlinked directory is not allowed: {_display_path(path, root)}"
                )
            _assert_within_root(path, root, description="directory")
            yield path

        for name in sorted(file_names):
            path = current_path / name
            if path.is_symlink():
                raise DiscoveryError(f"symlinked file is not allowed: {_display_path(path, root)}")
            _assert_within_root(path, root, description="file")


def _case_record(case_directory: Path, *, root: Path, source: str, release: str) -> CaseRecord:
    case_id = case_directory.name
    expected_names = {
        modality: f"{case_id}-{modality}.nii.gz"
        for modality in (*REQUIRED_MET_MODALITIES, *OPTIONAL_MET_MODALITIES)
    }
    expected_by_name = {filename: modality for modality, filename in expected_names.items()}
    discovered: dict[str, Path] = {}

    for entry in sorted(case_directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            raise DiscoveryError(
                f"symlink inside case {case_id!r} is not allowed: {_display_path(entry, root)}"
            )
        _assert_within_root(entry, root, description=f"entry in case {case_id!r}")
        if entry.is_dir():
            raise DiscoveryError(
                f"nested directory inside case {case_id!r} is not allowed: "
                f"{_display_path(entry, root)}"
            )

        modality = expected_by_name.get(entry.name)
        if modality is not None:
            if not entry.is_file():
                raise DiscoveryError(
                    f"expected image is not a regular file: {_display_path(entry, root)}"
                )
            discovered[modality] = entry
            continue

        if entry.name.endswith(_NIFTI_SUFFIXES):
            raise DiscoveryError(
                f"unknown NIfTI in case {case_id!r}; expected exact case-modality names, got "
                f"{entry.name!r}"
            )

    missing = sorted(set(REQUIRED_MET_MODALITIES) - set(discovered))
    if missing:
        raise DiscoveryError(f"case {case_id!r} is missing required modalities: {missing}")

    files = tuple(
        FileRecord.from_path(
            modality,
            discovered[modality],
            recorded_path=discovered[modality].relative_to(root).as_posix(),
        )
        for modality in sorted(discovered)
    )
    return CaseRecord.create(
        source=source,
        release=release,
        case_id=case_id,
        files=files,
    )


def _assert_within_root(path: Path, root: Path, *, description: str) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DiscoveryError(f"unable to resolve {description}: {path}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DiscoveryError(f"{description} escapes dataset root: {path}") from error


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return os.fspath(path)


# ``scan`` is a concise synonym for callers building manifests at the command
# line or in experiment setup code.
scan_met_release = discover_met_release


__all__ = [
    "OPTIONAL_MET_MODALITIES",
    "REQUIRED_MET_MODALITIES",
    "DiscoveryError",
    "discover_met_release",
    "scan_met_release",
]
