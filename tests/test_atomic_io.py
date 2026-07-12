from __future__ import annotations

import os
from pathlib import Path

import pytest

from simple_brats.atomic_io import atomic_create_bytes, atomic_replace_bytes
from simple_brats.short_run import _write_new_canonical


def _temporaries(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.tmp-*"))


def test_atomic_create_never_overwrites_and_cleans_temporary_files(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    atomic_create_bytes(destination, b"first")

    assert destination.read_bytes() == b"first"
    assert _temporaries(destination) == []
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_create_bytes(destination, b"second")
    assert destination.read_bytes() == b"first"
    assert _temporaries(destination) == []


def test_atomic_create_preserves_racing_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "race.json"

    def racing_link(_source: object, target: object) -> None:
        Path(target).write_bytes(b"winner")
        raise FileExistsError("simulated racing publisher")

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_create_bytes(destination, b"loser")

    assert destination.read_bytes() == b"winner"
    assert _temporaries(destination) == []


def test_atomic_replace_publishes_complete_payload_and_cleans_temp(tmp_path: Path) -> None:
    destination = tmp_path / "replace.bin"
    destination.write_bytes(b"old")
    atomic_replace_bytes(destination, b"new-complete-payload")

    assert destination.read_bytes() == b"new-complete-payload"
    assert _temporaries(destination) == []


def test_canonical_artifact_writer_uses_atomic_no_overwrite_publication(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "canonical.json"
    digest = _write_new_canonical(destination, {"b": 2, "a": 1})

    assert len(digest) == 64
    assert destination.read_bytes() == b'{"a":1,"b":2}'
    with pytest.raises(FileExistsError):
        _write_new_canonical(destination, {"a": 1, "b": 2})
    assert _temporaries(destination) == []
