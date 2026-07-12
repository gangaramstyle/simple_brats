import json
import sys
from pathlib import Path
from types import SimpleNamespace

from simple_brats.provenance import current_git_sha
from simple_brats.wandb_smoke import run_wandb_connectivity_smoke


def test_wandb_connectivity_smoke_records_visible_compute_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    launch_sha = current_git_sha(repo)
    login_calls: list[dict[str, object]] = []
    init_calls: list[dict[str, object]] = []
    log_calls: list[tuple[object, int]] = []

    class Run:
        id = "smoke-run"
        entity = "research-team"
        url = "https://wandb.example/simple-brats/runs/smoke-run"

        def log(self, values: object, *, step: int) -> None:
            log_calls.append((values, step))

        def finish(self) -> None:
            pass

    wandb = SimpleNamespace(
        login=lambda **kwargs: login_calls.append(kwargs) or True,
        init=lambda **kwargs: init_calls.append(kwargs) or Run(),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "simple-brats")
    monkeypatch.setenv("WANDB_ENTITY", "research-team")

    output = tmp_path / "wandb-smoke.json"
    report = run_wandb_connectivity_smoke(
        output=output,
        expected_git_sha=launch_sha,
        repo_root=repo,
    )

    assert login_calls == [{"verify": True, "force": True}]
    assert init_calls[0]["mode"] == "online"
    assert init_calls[0]["resume"] == "allow"
    assert log_calls == [({"connectivity/compute_node_online": 1}, 0)]
    assert report["run_url"] == Run.url
    assert json.loads(output.read_text()) == report
