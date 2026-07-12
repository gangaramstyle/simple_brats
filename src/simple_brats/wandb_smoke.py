"""Tiny scheduled proof that compute nodes can stream to the configured W&B server."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.data.manifest import canonical_json_bytes
from simple_brats.provenance import verify_git_sha
from simple_brats.tracking import (
    OnlineWandbConfig,
    online_run_url,
    require_verified_online_login,
)
from simple_brats.training import preserve_runner_rng_state


def run_wandb_connectivity_smoke(
    *,
    output: str | Path,
    expected_git_sha: str,
    repo_root: str | Path = ".",
) -> dict[str, object]:
    """Create one deterministic online run, stream one scalar, and record its URL."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    launch_sha = verify_git_sha(expected_git_sha, repo)
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite W&B smoke output: {destination}")
    destination.parent.resolve(strict=True)
    tracking = OnlineWandbConfig.from_environment()
    try:
        import wandb
    except Exception as error:
        raise RuntimeError("pinned W&B tracking extra is unavailable") from error
    with preserve_runner_rng_state():
        require_verified_online_login(wandb)
    identity_payload = canonical_json_bytes(
        {
            "schema": "simple-brats.wandb-connectivity-smoke-identity",
            "schema_version": 1,
            "launch_sha": launch_sha,
            "tracking": tracking.to_dict(),
        }
    )
    run_id = hashlib.sha256(identity_payload).hexdigest()[:24]
    group = f"connectivity-{launch_sha[:12]}"
    with preserve_runner_rng_state():
        run = wandb.init(
            **tracking.init_kwargs(),
            id=run_id,
            group=group,
            name=f"simple-brats-connectivity-{launch_sha[:12]}",
            job_type="connectivity-smoke",
            dir=str(destination.parent),
            config={"launch_sha": launch_sha, "tracking": tracking.to_dict()},
            reinit=True,
        )
    if run is None:
        raise RuntimeError("online W&B connectivity smoke returned no run")
    try:
        url = online_run_url(run)
        run.log({"connectivity/compute_node_online": 1}, step=0)
    finally:
        run.finish()
    report: dict[str, object] = {
        "schema": "simple-brats.wandb-connectivity-smoke",
        "schema_version": 1,
        "status": "ok",
        "launch_sha": launch_sha,
        "tracking": tracking.to_dict(),
        "actual_entity": getattr(run, "entity", None) or tracking.entity,
        "group": group,
        "run_id": run_id,
        "run_url": url,
    }
    atomic_create_bytes(destination, canonical_json_bytes(report))
    print(f"W&B compute connectivity verified: {url}", flush=True)
    print(canonical_json_bytes(report).decode(), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify online W&B from a scheduled compute node")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_wandb_connectivity_smoke(
        output=args.output,
        expected_git_sha=args.expected_git_sha,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
