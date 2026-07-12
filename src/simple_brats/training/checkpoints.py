"""Atomic checkpoint cadence with mandatory periodic W&B artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from simple_brats.atomic_io import atomic_create_bytes, fsync_file_and_parent

_ARTIFACT_RECEIPT_SCHEMA = "simple-brats.checkpoint-artifact-receipt"
_ARTIFACT_RECEIPT_SCHEMA_VERSION = 1
_DEFAULT_WANDB_UPLOAD_TIMEOUT_SECONDS = 900


class ArtifactSink(Protocol):
    def log_checkpoint(
        self,
        path: Path,
        *,
        step: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any] | None: ...


class WandbArtifactSink:
    """Log one checkpoint and return only after W&B confirms its artifact version."""

    def __init__(
        self,
        run: Any,
        *,
        collection_name: str | None = None,
        upload_timeout_seconds: int = _DEFAULT_WANDB_UPLOAD_TIMEOUT_SECONDS,
    ) -> None:
        if run is None or not getattr(run, "id", None):
            raise ValueError("a live W&B run with an ID is required")
        selected_name = collection_name or f"{run.id}-checkpoints"
        if not isinstance(selected_name, str) or not selected_name:
            raise ValueError("W&B artifact collection name must be non-empty")
        if (
            isinstance(upload_timeout_seconds, bool)
            or not isinstance(upload_timeout_seconds, int)
            or upload_timeout_seconds <= 0
        ):
            raise ValueError("W&B artifact upload timeout must be a positive integer")
        self.run = run
        self.collection_name = selected_name
        self.upload_timeout_seconds = upload_timeout_seconds

    def log_checkpoint(
        self,
        path: Path,
        *,
        step: int,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            import wandb
        except ImportError as error:  # pragma: no cover - depends on tracking extra
            raise RuntimeError("install the 'tracking' extra to log W&B artifacts") from error
        artifact = wandb.Artifact(
            name=self.collection_name,
            type="model",
            metadata={
                "checkpoint_step": step,
                "producing_run_id": self.run.id,
                "provenance": dict(metadata),
            },
        )
        artifact.add_file(str(path), name="checkpoint.pt")
        logged_artifact = self.run.log_artifact(
            artifact,
            aliases=[f"step-{step:09d}", "latest"],
        )
        wait = getattr(logged_artifact, "wait", None)
        if not callable(wait):
            raise RuntimeError("W&B did not return an awaitable artifact upload handle")
        completed_artifact = wait(timeout=self.upload_timeout_seconds)
        if completed_artifact is None:
            completed_artifact = logged_artifact
        is_draft = getattr(completed_artifact, "is_draft", None)
        if not callable(is_draft) or bool(is_draft()):
            raise RuntimeError("W&B artifact upload did not reach a committed version")

        proof: dict[str, str] = {}
        for field in ("id", "version", "qualified_name", "digest"):
            value = getattr(completed_artifact, field, None)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"committed W&B artifact is missing {field}")
            proof[field] = value
        return {
            "backend": "wandb",
            "completion": "wait_returned_committed_artifact",
            "producing_run_id": self.run.id,
            "collection_name": self.collection_name,
            "upload_timeout_seconds": self.upload_timeout_seconds,
            "artifact": proof,
        }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError("artifact receipt data must be canonical JSON") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Atomically checkpoint, synchronously upload, and durably receipt artifacts."""

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

    @staticmethod
    def artifact_receipt_path(checkpoint: str | Path) -> Path:
        path = Path(checkpoint)
        return path.parent / "artifact-receipts" / f"{path.stem}.artifact.json"

    @staticmethod
    def _checkpoint_binding(
        checkpoint: Path,
        *,
        step: int,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise RuntimeError(
                f"checkpoint artifact input must be a regular non-symlink file: {checkpoint}"
            )
        expected_name = f"step-{step:09d}.pt"
        if checkpoint.name != expected_name:
            raise RuntimeError(
                f"checkpoint artifact filename must be {expected_name}: {checkpoint}"
            )
        metadata_bytes = _canonical_json_bytes(dict(metadata))
        return {
            "step": step,
            "checkpoint": {
                "name": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256_file(checkpoint),
            },
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        }

    @staticmethod
    def _validate_receipt(path: Path, expected_binding: Mapping[str, Any]) -> None:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"artifact receipt must be a regular non-symlink file: {path}")
        payload_bytes = path.read_bytes()
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"artifact receipt is not valid JSON: {path}") from error
        if not isinstance(payload, dict) or payload_bytes != _canonical_json_bytes(payload):
            raise RuntimeError(f"artifact receipt is not canonical JSON: {path}")
        expected_keys = {
            "schema",
            "schema_version",
            "step",
            "checkpoint",
            "metadata_sha256",
            "upload",
        }
        if set(payload) != expected_keys:
            raise RuntimeError(f"artifact receipt has an unsupported schema: {path}")
        if (
            payload["schema"] != _ARTIFACT_RECEIPT_SCHEMA
            or payload["schema_version"] != _ARTIFACT_RECEIPT_SCHEMA_VERSION
        ):
            raise RuntimeError(f"artifact receipt has an unsupported version: {path}")
        for field in ("step", "checkpoint", "metadata_sha256"):
            if payload[field] != expected_binding[field]:
                raise RuntimeError(f"artifact receipt does not match {field}: {path}")
        if not isinstance(payload["upload"], dict):
            raise RuntimeError(f"artifact receipt upload proof must be a mapping: {path}")

    def ensure_artifact_logged(
        self,
        checkpoint: str | Path,
        *,
        step: int,
        metadata: Mapping[str, Any],
    ) -> Path:
        """Return a valid durable receipt, uploading first when it is absent.

        The checkpoint is already durable before this method is called.  Therefore an
        interrupted upload leaves a resumable checkpoint and no success receipt; the
        next invocation can safely retry this method before advancing the optimizer.
        """

        if not self.policy.is_artifact_step(step):
            raise RuntimeError("artifact receipt requested off the configured artifact cadence")
        path = Path(checkpoint)
        binding = self._checkpoint_binding(path, step=step, metadata=metadata)
        receipt = self.artifact_receipt_path(path)
        if receipt.exists() or receipt.is_symlink():
            self._validate_receipt(receipt, binding)
            return receipt
        if self.artifact_sink is None:
            raise RuntimeError("artifact cadence reached without a configured artifact sink")

        receipt_root = receipt.parent
        receipt_root.mkdir(parents=True, exist_ok=True)
        if receipt_root.is_symlink() or not receipt_root.is_dir():
            raise RuntimeError(
                f"artifact receipt root must be a non-symlink directory: {receipt_root}"
            )
        upload = self.artifact_sink.log_checkpoint(
            path,
            step=step,
            metadata=metadata,
        )
        if upload is None:
            upload_record: dict[str, Any] = {
                "backend": "opaque_artifact_sink",
                "completion": "log_checkpoint_returned_without_error",
            }
        elif isinstance(upload, Mapping):
            upload_record = dict(upload)
        else:
            raise RuntimeError("artifact sink upload proof must be a mapping or None")

        # Re-hash after transport completion so a concurrently changed checkpoint
        # can never receive a success receipt for the bytes that were first read.
        if self._checkpoint_binding(path, step=step, metadata=metadata) != binding:
            raise RuntimeError("checkpoint changed while its artifact was being uploaded")
        payload = {
            "schema": _ARTIFACT_RECEIPT_SCHEMA,
            "schema_version": _ARTIFACT_RECEIPT_SCHEMA_VERSION,
            **binding,
            "upload": upload_record,
        }
        payload_bytes = _canonical_json_bytes(payload)
        try:
            atomic_create_bytes(receipt, payload_bytes)
        except FileExistsError:
            # A racing retry may have published an equally valid receipt.  Never
            # replace it; validate its checkpoint/provenance binding instead.
            self._validate_receipt(receipt, binding)
            return receipt
        if receipt.read_bytes() != payload_bytes:
            raise RuntimeError(f"artifact receipt changed after publication: {receipt}")
        return receipt

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
            self.ensure_artifact_logged(
                destination,
                step=step,
                metadata=metadata,
            )
        return destination
