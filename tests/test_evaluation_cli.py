from pathlib import Path

import pytest

from simple_brats.evaluation.cli import _write_new_canonical


def test_final_checkpoint_report_is_atomically_created_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkpoint-evaluation.json"
    _write_new_canonical(destination, {"schema": "test", "value": 1})
    original = destination.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_new_canonical(destination, {"schema": "test", "value": 2})

    assert destination.read_bytes() == original
    assert not tuple(tmp_path.glob(f".{destination.name}.tmp-*"))
