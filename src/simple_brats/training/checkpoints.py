"""Atomic checkpoint cadence with mandatory periodic W&B artifacts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from simple_brats.atomic_io import fsync_file_and_parent


class ArtifactSink(Protocol):
    def log_checkpoint(
        self,
        path: Path,
        *,
        step: int,
        metadata: Mapping[str, Any],
    ) -> None: ...


class WandbArtifactSink:
    """Thin adapter that works with online or `WANDB_MODE=offline` runs."""

    def __init__(self, run: Any) -> None:
        if run is None or not getattr(run, "id", None):
            raise ValueError("a live W&B run with an ID is required")
        self.run = run

    def log_checkpoint(
        self,
        path: Path,
        *,
        step: int,
        metadata: Mapping[str, Any],
    ) -> None:
        try:
            import wandb
        except ImportError as error:  # pragma: no cover - depends on tracking extra
            raise RuntimeError("install the 'tracking' extra to log W&B artifacts") from error
        artifact = wandb.Artifact(
            name=f"{self.run.id}-checkpoint-{step:09d}",
            type="model",
            metadata=dict(metadata),
        )
        artifact.add_file(str(path), name="checkpoint.pt")
        self.run.log_artifact(artifact)


@dataclass(frozen=True)
class CheckpointPolicy:
    checkpoint_every_steps: int = 1_000
    artifact_every_steps: int = 5_000

    def __post_init__(self) -> None:
        if self.checkpoint_every_steps <= 0 or self.artifact_every_steps <= 0:
            raise ValueError("checkpoint and artifact cadences must be positive")
        if self.artifact_every_steps % self.checkpoint_every_steps:
            raise ValueError("every artifact step must also be a checkpoint step")

    def is_checkpoint_step(self, step: int) -> bool:
        return step > 0 and step % self.checkpoint_every_steps == 0

    def is_artifact_step(self, step: int) -> bool:
        return step > 0 and step % self.artifact_every_steps == 0


class CheckpointManager:
    """Write checkpoints atomically and fail if a required artifact cannot be logged."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: CheckpointPolicy,
        artifact_sink: ArtifactSink | None,
    ) -> None:
        self.root = Path(root)
        self.policy = policy
        self.artifact_sink = artifact_sink

    def maybe_save(
        self,
        *,
        step: int,
        state: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Path | None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if not self.policy.is_checkpoint_step(step):
            return None
        if self.policy.is_artifact_step(step) and self.artifact_sink is None:
            raise RuntimeError("artifact cadence reached without a configured artifact sink")

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"step-{step:09d}.pt"
        temporary = self.root / f".{destination.name}.tmp-{os.getpid()}"
        payload = {
            "schema_version": 1,
            "step": step,
            "metadata": dict(metadata),
            "state": dict(state),
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
            fsync_file_and_parent(destination)
        finally:
            temporary.unlink(missing_ok=True)

        if self.policy.is_artifact_step(step):
            assert self.artifact_sink is not None
            self.artifact_sink.log_checkpoint(
                destination,
                step=step,
                metadata=metadata,
            )
        return destination
