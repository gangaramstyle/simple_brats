"""Runtime provenance checks shared by local and Slurm entrypoints."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import load_manifest, load_split, validate_split

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(root: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip().lower()
    if _FULL_GIT_SHA.fullmatch(sha) is None:
        raise RuntimeError(f"git returned a non-canonical commit SHA: {sha!r}")
    return sha


def verify_git_sha(expected: str, root: str | Path = ".") -> str:
    expected = expected.strip().lower()
    if _FULL_GIT_SHA.fullmatch(expected) is None:
        raise ValueError("expected git SHA must be the full 40-character lowercase commit ID")
    actual = current_git_sha(root)
    if actual != expected:
        raise RuntimeError(f"git provenance mismatch: expected {expected}, got {actual}")
    return actual


@dataclass(frozen=True)
class RunProvenance:
    git_sha: str
    config_sha256: str
    data_manifest_sha256: str | None
    split_manifest_sha256: str | None
    synthetic_dataset_id: str | None
    execution_sha256: str
    lock_sha256: str
    torch_version: str
    seed: int

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def collect_provenance(
    config: ExperimentConfig,
    *,
    execution: dict[str, object],
    data_manifest_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
    synthetic_dataset_id: str | None = None,
    root: str | Path = ".",
    expected_git_sha: str | None = None,
) -> RunProvenance:
    using_real_manifests = data_manifest_sha256 is not None or split_manifest_sha256 is not None
    if using_real_manifests:
        if synthetic_dataset_id is not None:
            raise ValueError(
                "real manifest digests and a synthetic dataset ID are mutually exclusive"
            )
        if data_manifest_sha256 is None or split_manifest_sha256 is None:
            raise ValueError("data and split manifest SHA-256 values must be supplied together")
        data_manifest_sha256 = data_manifest_sha256.lower()
        split_manifest_sha256 = split_manifest_sha256.lower()
        if _SHA256.fullmatch(data_manifest_sha256) is None:
            raise ValueError("data_manifest_sha256 must be a canonical SHA-256 digest")
        if _SHA256.fullmatch(split_manifest_sha256) is None:
            raise ValueError("split_manifest_sha256 must be a canonical SHA-256 digest")
    elif not synthetic_dataset_id or synthetic_dataset_id != synthetic_dataset_id.strip():
        raise ValueError("a non-empty synthetic_dataset_id is required without real manifests")

    root_path = Path(root)
    git_sha = (
        verify_git_sha(expected_git_sha, root_path)
        if expected_git_sha is not None
        else current_git_sha(root_path)
    )
    execution_payload = json.dumps(
        execution,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return RunProvenance(
        git_sha=git_sha,
        config_sha256=config.sha256,
        data_manifest_sha256=data_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        synthetic_dataset_id=synthetic_dataset_id,
        execution_sha256=hashlib.sha256(execution_payload).hexdigest(),
        lock_sha256=sha256_file(root_path / "uv.lock"),
        torch_version=torch.__version__,
        seed=config.seed,
    )


def collect_locked_provenance(
    config: ExperimentConfig,
    *,
    execution: dict[str, object],
    data_manifest_path: str | Path,
    split_manifest_path: str | Path,
    expected_data_manifest_sha256: str,
    expected_split_manifest_sha256: str,
    root: str | Path = ".",
    expected_git_sha: str | None = None,
) -> RunProvenance:
    """Load the exact locked manifests before recording real-data provenance."""

    manifest = load_manifest(
        data_manifest_path,
        expected_sha256=expected_data_manifest_sha256,
    )
    split = load_split(
        split_manifest_path,
        expected_sha256=expected_split_manifest_sha256,
    )
    validate_split(manifest, split)
    return collect_provenance(
        config,
        execution=execution,
        data_manifest_sha256=manifest.sha256,
        split_manifest_sha256=split.sha256,
        root=root,
        expected_git_sha=expected_git_sha,
    )
