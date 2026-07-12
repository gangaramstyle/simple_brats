"""Durable atomic publication helpers for immutable experiment artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _payload_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    return bytes(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file_and_parent(path: str | os.PathLike[str]) -> None:
    """Durably flush one published regular file and its directory entry."""

    published = Path(path)
    descriptor = os.open(published, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(published.parent)


def _write_temporary(destination: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def atomic_create_bytes(
    path: str | os.PathLike[str],
    payload: bytes | bytearray | memoryview,
) -> None:
    """Atomically create ``path`` without ever replacing an existing entry.

    Data is fsynced in a same-directory temporary file, published with an
    atomic hard-link operation, the temporary name is removed, and the parent
    directory is fsynced.  A racing destination—regular file or symlink—wins;
    this function raises ``FileExistsError`` and never alters it.
    """

    destination = Path(path)
    data = _payload_bytes(payload)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"destination parent is not a directory: {destination.parent}")
    temporary = _write_temporary(destination, data)
    published = False
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing path: {destination}") from error
        published = True
    finally:
        temporary.unlink(missing_ok=True)
        if published:
            fsync_file_and_parent(destination)


def atomic_replace_bytes(
    path: str | os.PathLike[str],
    payload: bytes | bytearray | memoryview,
) -> None:
    """Atomically replace ``path`` after durably writing a temporary file."""

    destination = Path(path)
    data = _payload_bytes(payload)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"destination parent is not a directory: {destination.parent}")
    temporary = _write_temporary(destination, data)
    replaced = False
    try:
        os.replace(temporary, destination)
        replaced = True
    finally:
        temporary.unlink(missing_ok=True)
        if replaced:
            fsync_file_and_parent(destination)


__all__ = ["atomic_create_bytes", "atomic_replace_bytes", "fsync_file_and_parent"]
