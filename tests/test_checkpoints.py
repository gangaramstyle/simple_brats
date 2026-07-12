import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from simple_brats.training import CheckpointManager, CheckpointPolicy, WandbArtifactSink


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
    receipt = manager.artifact_receipt_path(artifact_checkpoint)
    assert receipt.is_file() and not receipt.is_symlink()
    receipt_payload = json.loads(receipt.read_bytes())
    assert receipt_payload["step"] == 5_000
    assert receipt_payload["checkpoint"]["name"] == artifact_checkpoint.name
    assert receipt_payload["upload"]["completion"] == ("log_checkpoint_returned_without_error")
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


def test_wandb_sink_uses_one_stable_collection_with_step_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []
    logged: list[tuple[object, list[str]]] = []
    waited: list[int] = []

    class Artifact:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.files: list[tuple[str, str]] = []
            self.id = "artifact-id"
            self.version = "v3"
            self.qualified_name = "entity/project/science-provenance-checkpoints:v3"
            self.digest = "artifact-digest"
            self.draft = True
            created.append(self)

        def add_file(self, path: str, *, name: str) -> None:
            self.files.append((path, name))

        def wait(self, *, timeout: int):
            waited.append(timeout)
            self.draft = False
            return self

        def is_draft(self) -> bool:
            return self.draft

    def log_artifact(artifact, *, aliases):
        logged.append((artifact, aliases))
        return artifact

    run = SimpleNamespace(
        id="run-123",
        log_artifact=log_artifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    checkpoint = tmp_path / "step-000005000.pt"
    checkpoint.write_bytes(b"checkpoint")
    sink = WandbArtifactSink(run, collection_name="science-provenance-checkpoints")

    proof = sink.log_checkpoint(
        checkpoint,
        step=5_000,
        metadata={"manifest_sha256": "a" * 64},
    )

    artifact = created[0]
    assert artifact.kwargs["name"] == "science-provenance-checkpoints"
    assert artifact.kwargs["metadata"]["checkpoint_step"] == 5_000
    assert artifact.files == [(str(checkpoint), "checkpoint.pt")]
    assert logged == [(artifact, ["step-000005000", "latest"])]
    assert waited == [900]
    assert proof == {
        "backend": "wandb",
        "completion": "wait_returned_committed_artifact",
        "producing_run_id": "run-123",
        "collection_name": "science-provenance-checkpoints",
        "upload_timeout_seconds": 900,
        "artifact": {
            "id": "artifact-id",
            "version": "v3",
            "qualified_name": "entity/project/science-provenance-checkpoints:v3",
            "digest": "artifact-digest",
        },
    }


def test_wandb_sink_propagates_upload_wait_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Artifact:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def add_file(self, _path: str, *, name: str) -> None:
            assert name == "checkpoint.pt"

        def wait(self, *, timeout: int):
            assert timeout == 17
            raise RuntimeError("transient upload failure")

    artifact = Artifact()
    run = SimpleNamespace(id="run-123", log_artifact=lambda *_args, **_kwargs: artifact)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=lambda **_kwargs: artifact))
    checkpoint = tmp_path / "step-000005000.pt"
    checkpoint.write_bytes(b"checkpoint")

    sink = WandbArtifactSink(run, upload_timeout_seconds=17)
    with pytest.raises(RuntimeError, match="transient upload failure"):
        sink.log_checkpoint(checkpoint, step=5_000, metadata={})


def test_failed_artifact_upload_leaves_checkpoint_and_resume_publishes_receipt(
    tmp_path: Path,
) -> None:
    class FailingSink:
        def log_checkpoint(self, *_args, **_kwargs):
            raise RuntimeError("transport unavailable")

    metadata = {"git_sha": "a" * 40}
    policy = CheckpointPolicy(checkpoint_every_steps=1_000, artifact_every_steps=5_000)
    failed = CheckpointManager(tmp_path, policy=policy, artifact_sink=FailingSink())

    with pytest.raises(RuntimeError, match="transport unavailable"):
        failed.maybe_save(step=5_000, state={"weight": torch.tensor([3.0])}, metadata=metadata)

    checkpoint = tmp_path / "step-000005000.pt"
    receipt = failed.artifact_receipt_path(checkpoint)
    assert checkpoint.is_file()
    assert not receipt.exists()

    recovered_sink = RecordingSink()
    recovered = CheckpointManager(tmp_path, policy=policy, artifact_sink=recovered_sink)
    assert recovered.ensure_artifact_logged(checkpoint, step=5_000, metadata=metadata) == receipt
    assert receipt.is_file()
    assert recovered_sink.calls == [(checkpoint, 5_000, metadata)]

    # A durable matching receipt makes subsequent retries network-free.
    recovered.ensure_artifact_logged(checkpoint, step=5_000, metadata=metadata)
    assert recovered_sink.calls == [(checkpoint, 5_000, metadata)]
