from pathlib import Path

import pytest
import torch

from simple_brats.training import CheckpointManager, CheckpointPolicy


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, int, dict[str, object]]] = []

    def log_checkpoint(self, path, *, step, metadata) -> None:
        self.calls.append((path, step, dict(metadata)))


def test_checkpoint_and_artifact_cadence(tmp_path) -> None:
    sink = RecordingSink()
    manager = CheckpointManager(
        tmp_path,
        policy=CheckpointPolicy(),
        artifact_sink=sink,
    )
    assert manager.maybe_save(step=999, state={}, metadata={}) is None
    first = manager.maybe_save(
        step=1_000,
        state={"weight": torch.tensor([1.0])},
        metadata={"git_sha": "abc"},
    )
    assert first is not None and first.exists()
    assert not sink.calls

    artifact_checkpoint = manager.maybe_save(
        step=5_000,
        state={"weight": torch.tensor([2.0])},
        metadata={"git_sha": "abc"},
    )
    assert artifact_checkpoint is not None
    assert sink.calls == [(artifact_checkpoint, 5_000, {"git_sha": "abc"})]
    payload = torch.load(artifact_checkpoint, weights_only=False)
    assert payload["step"] == 5_000


def test_artifact_step_fails_without_sink(tmp_path) -> None:
    manager = CheckpointManager(
        tmp_path,
        policy=CheckpointPolicy(),
        artifact_sink=None,
    )
    with pytest.raises(RuntimeError, match="without a configured artifact sink"):
        manager.maybe_save(step=5_000, state={}, metadata={})
