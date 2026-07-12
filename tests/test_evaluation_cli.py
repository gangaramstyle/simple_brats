import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from simple_brats.evaluation.cli import _log_wandb, _write_new_canonical
from simple_brats.tracking import OnlineWandbConfig


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


def test_evaluation_wandb_is_online_deterministic_and_records_url_without_rng_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation.json"
    report = {
        "schema": "test-evaluation",
        "metric": 0.75,
        "provenance": {
            "evaluation_patch_manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "checkpoint_provenance_sha256": "c" * 64,
        },
    }
    _write_new_canonical(output, report)
    init_calls: list[dict[str, object]] = []
    logged: list[tuple[object, int]] = []
    artifact_calls: list[tuple[object, list[str]]] = []

    class FakeArtifact:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.files: list[tuple[str, str]] = []

        def add_file(self, path: str, *, name: str) -> None:
            self.files.append((path, name))

    class FakeRun:
        id = "server-id"
        entity = "research-team"
        url = "https://wandb.example/simple-brats/runs/server-id"

        def log(self, values: object, *, step: int) -> None:
            logged.append((values, step))

        def log_artifact(self, artifact: object, *, aliases: list[str]) -> None:
            artifact_calls.append((artifact, aliases))

        def finish(self) -> None:
            pass

    def init(**kwargs: object) -> FakeRun:
        init_calls.append(kwargs)
        random.random()
        np.random.random()
        torch.rand(3)
        return FakeRun()

    wandb = SimpleNamespace(init=init, Artifact=FakeArtifact)
    tracking = OnlineWandbConfig(
        project="simple-brats",
        entity="research-team",
        base_url="https://wandb.example",
    )
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    record = _log_wandb(
        report=report,
        output=output,
        checkpoint_step=5_000,
        run_name="heldout-5000",
        wandb_module=wandb,
        tracking=tracking,
    )

    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert init_calls[0]["mode"] == "online"
    assert init_calls[0]["force"] is True
    assert init_calls[0]["resume"] == "allow"
    assert init_calls[0]["group"] == f"heldout-{'c' * 12}-{'a' * 12}"
    assert logged[0][1] == 5_000
    assert artifact_calls[0][1] == ["step-000005000", "latest"]
    assert record["run_url"] == FakeRun.url
    tracking_record = json.loads((tmp_path / "evaluation.json.wandb.json").read_text())
    assert tracking_record == record
