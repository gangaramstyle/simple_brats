"""One-step, provenance-locked real-data I/O and CUDA training pilot.

This entrypoint is deliberately smaller than a training runner.  It proves
that one exact filtered-manifest case can pass through canonical NIfTI
preparation, materialized planning, leakage-checked batch assembly, the small
matching model, backward, AdamW, and exactly one EMA update.  It never retries
with relaxed sampling rules and never overwrites an output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch

from simple_brats.config import ExperimentConfig, load_experiment_config
from simple_brats.data.case_grids import (
    CaseGridManifest,
    CaseGridRecord,
    load_case_grid_manifest,
)
from simple_brats.data.extraction import ExtractionSpec
from simple_brats.data.manifest import (
    CaseRecord,
    DatasetManifest,
    canonical_json_bytes,
    load_manifest,
    sha256_file,
)
from simple_brats.data.pipeline import (
    CachedNiftiPatchExtractor,
    prepare_case_matching_plan_record,
)
from simple_brats.data.real_batches import assemble_matching_batch
from simple_brats.data.splits import cases_for_splits, load_split, validate_split
from simple_brats.provenance import verify_git_sha
from simple_brats.sampling import save_patch_plan
from simple_brats.training.matching import build_matching_system, optimizer_parameter_groups

_T = TypeVar("_T")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RealPilotError(RuntimeError):
    """Raised when the one-step pilot cannot honor its locked contract."""


def _require_canonical_file(path: Path, expected_payload: bytes, description: str) -> None:
    try:
        actual_payload = path.read_bytes()
    except OSError as error:
        raise RealPilotError(f"unable to read {description}: {path}") from error
    if actual_payload != expected_payload:
        raise RealPilotError(f"{description} is not canonical on disk: {path}")


def _resolve_regular_file(path: str | os.PathLike[str], description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RealPilotError(f"{description} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RealPilotError(f"{description} is unavailable: {path}") from error
    if not resolved.is_file():
        raise RealPilotError(f"{description} must be a regular file")
    return resolved


@dataclass(frozen=True, slots=True)
class _CaseGridExtractionBinding:
    """Selected case's exact physical grid and derived extraction contract."""

    path: Path
    catalog: CaseGridManifest
    record: CaseGridRecord
    spec: ExtractionSpec

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_sha256: str,
        manifest: DatasetManifest,
        case: CaseRecord,
    ) -> _CaseGridExtractionBinding:
        resolved = _resolve_regular_file(path, "case-grid manifest")
        catalog = load_case_grid_manifest(resolved, expected_sha256=expected_sha256)
        catalog.validate_manifest(manifest)
        record = catalog.record_for_case(case)
        spec = catalog.extraction_spec_for_case(case)
        if record.extraction_spec_sha256 != spec.sha256:
            raise RealPilotError(
                "selected case-grid record does not match its derived extraction spec"
            )
        return cls(path=resolved, catalog=catalog, record=record, spec=spec)

    @property
    def extraction_spec_sha256(self) -> str:
        return self.spec.sha256

    def build_extractor(
        self,
        *,
        data_root: str | os.PathLike[str],
        manifest: DatasetManifest,
    ) -> CachedNiftiPatchExtractor:
        return CachedNiftiPatchExtractor(
            data_root=data_root,
            manifest=manifest,
            data_manifest_sha256=manifest.sha256,
            extraction_spec=self.spec,
            max_cached_volumes=4,
        )


def _new_output_path(path: str | os.PathLike[str]) -> Path:
    requested = Path(path).expanduser()
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise RealPilotError("output directory parent must already exist") from error
    if not parent.is_dir():
        raise RealPilotError("output directory parent must be a directory")
    destination = parent / requested.name
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite output directory {destination}")
    return destination


def _write_new_canonical(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != payload:
        raise RealPilotError(f"canonical artifact verification failed after write: {path}")
    return hashlib.sha256(payload).hexdigest()


def _select_train_case(
    train_cases: Sequence[CaseRecord],
    *,
    case_id: str | None,
) -> CaseRecord:
    ordered = tuple(
        sorted(
            train_cases,
            key=lambda case: (
                case.subject_id,
                case.visit_id,
                case.source,
                case.release,
                case.case_id,
            ),
        )
    )
    if not ordered:
        raise RealPilotError("the pinned split contains no training cases")
    if case_id is None:
        return ordered[0]
    matches = tuple(case for case in ordered if case.case_id == case_id)
    if len(matches) != 1:
        raise RealPilotError(
            f"requested case_id {case_id!r} matched {len(matches)} training cases; expected one"
        )
    return matches[0]


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation: Callable[[], _T]) -> tuple[_T, float]:
    _cuda_synchronize(device)
    start = time.perf_counter()
    result = operation()
    _cuda_synchronize(device)
    return result, time.perf_counter() - start


def _finite_float(value: object, description: str) -> float:
    result = float(value)  # type: ignore[arg-type]
    if not torch.isfinite(torch.tensor(result)):
        raise RealPilotError(f"{description} is not finite")
    return result


def run_real_io_pilot(
    *,
    data_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    expected_manifest_sha256: str,
    split_path: str | os.PathLike[str],
    expected_split_sha256: str,
    case_grid_manifest_path: str | os.PathLike[str],
    expected_case_grid_manifest_sha256: str,
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    expected_git_sha: str,
    repo_root: str | os.PathLike[str] = ".",
    device: torch.device | str = "cuda",
    case_id: str | None = None,
    epoch: int = 0,
    bag_index: int = 0,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    max_gradient_norm: float = 10.0,
    candidate_pool_size: int = 512,
    max_plan_attempts: int = 8,
) -> dict[str, object]:
    """Run and persist one exact real-data optimizer/EMA step.

    The output directory is an exclusive, immutable bundle.  A materialized
    plan and its prepared-volume/candidate audit are written before batch
    assembly and model execution, so an I/O or CUDA failure still leaves the
    exact attempted sampling record for inspection.
    """

    start_total = time.perf_counter()
    if _FULL_GIT_SHA.fullmatch(expected_git_sha) is None:
        raise ValueError("expected_git_sha must be a full lowercase 40-character commit ID")
    for value, description in (
        (expected_manifest_sha256, "expected_manifest_sha256"),
        (expected_split_sha256, "expected_split_sha256"),
        (
            expected_case_grid_manifest_sha256,
            "expected_case_grid_manifest_sha256",
        ),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    if epoch < 0 or bag_index < 0:
        raise ValueError("epoch and bag_index must be non-negative")
    if learning_rate <= 0 or weight_decay < 0 or max_gradient_norm <= 0:
        raise ValueError("optimizer values must satisfy lr>0, weight_decay>=0, grad_clip>0")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RealPilotError("CUDA pilot requested but torch.cuda.is_available() is false")

    repo_root_path = Path(repo_root).expanduser().resolve(strict=True)
    launch_sha = verify_git_sha(expected_git_sha, repo_root_path)
    manifest_file = _resolve_regular_file(manifest_path, "filtered manifest")
    split_file = _resolve_regular_file(split_path, "subject split")
    config_file = _resolve_regular_file(config_path, "experiment config")
    destination = _new_output_path(output_dir)

    artifact_start = time.perf_counter()
    manifest = load_manifest(manifest_file, expected_sha256=expected_manifest_sha256)
    split = load_split(split_file, expected_sha256=expected_split_sha256)
    config: ExperimentConfig = load_experiment_config(config_file)
    validate_split(manifest, split)
    _require_canonical_file(
        manifest_file,
        canonical_json_bytes(manifest.to_dict()),
        "filtered manifest",
    )
    _require_canonical_file(split_file, canonical_json_bytes(split.to_dict()), "subject split")
    train_cases = cases_for_splits(manifest, split, ("train",))
    selected_case = _select_train_case(train_cases, case_id=case_id)
    extraction_binding = _CaseGridExtractionBinding.load(
        case_grid_manifest_path,
        expected_sha256=expected_case_grid_manifest_sha256,
        manifest=manifest,
        case=selected_case,
    )
    artifact_seconds = time.perf_counter() - artifact_start

    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    extractor = extraction_binding.build_extractor(
        data_root=data_root,
        manifest=manifest,
    )
    plan_start = time.perf_counter()
    prepared = prepare_case_matching_plan_record(
        extractor,
        selected_case,
        epoch=epoch,
        bag_index=bag_index,
        experiment_seed=config.seed,
        target_count=config.task.positions_per_bag,
        candidate_pool_size=candidate_pool_size,
        max_attempts=max_plan_attempts,
    )
    plan_seconds = time.perf_counter() - plan_start

    destination.mkdir(mode=0o700)
    plan_path = destination / "materialized-patch-plan.json"
    audit_path = destination / "prepared-plan-audit.json"
    save_patch_plan(prepared.plan, plan_path, overwrite=False)
    audit_sha256 = _write_new_canonical(audit_path, prepared.to_dict())
    if audit_sha256 != prepared.sha256:
        raise RealPilotError(
            f"prepared plan audit SHA mismatch: expected {prepared.sha256}, got {audit_sha256}"
        )

    batch_start = time.perf_counter()
    cpu_batch = assemble_matching_batch(
        selected_case,
        prepared.plan,
        extractor,
        data_manifest_sha256=manifest.sha256,
        plan_sha256=prepared.plan.sha256,
        extraction_spec_sha256=extraction_binding.extraction_spec_sha256,
    )
    batch_seconds = time.perf_counter() - batch_start

    batch, transfer_seconds = _timed(
        resolved_device,
        lambda: cpu_batch.to(resolved_device),
    )

    def build_model_and_optimizer() -> tuple[Any, torch.optim.AdamW]:
        system = build_matching_system(config).to(resolved_device).train()
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(system, weight_decay=weight_decay),
            lr=learning_rate,
        )
        return system, optimizer

    (system, optimizer), model_seconds = _timed(resolved_device, build_model_and_optimizer)
    if int(system.target_teacher.num_updates) != 0:
        raise RealPilotError("fresh EMA teacher did not begin with zero updates")
    ema_parameter_before = next(system.target_teacher.parameters()).detach().clone()

    output, forward_seconds = _timed(resolved_device, lambda: system(batch))
    loss = _finite_float(output.loss.detach(), "matching loss")

    optimizer.zero_grad(set_to_none=True)

    def backward_and_clip() -> torch.Tensor:
        output.loss.backward()
        return torch.nn.utils.clip_grad_norm_(
            system.parameters(),
            max_norm=max_gradient_norm,
        )

    gradient_norm_tensor, backward_seconds = _timed(resolved_device, backward_and_clip)
    gradient_norm = _finite_float(gradient_norm_tensor, "gradient norm")
    if gradient_norm <= 0:
        raise RealPilotError("real-data pilot produced no training gradient")

    _, optimizer_seconds = _timed(resolved_device, optimizer.step)
    _, ema_seconds = _timed(resolved_device, system.update_teacher)
    if int(system.target_teacher.num_updates) != 1:
        raise RealPilotError("one pilot step must perform exactly one EMA teacher update")
    ema_parameter_after = next(system.target_teacher.parameters()).detach()
    ema_update_norm = _finite_float(
        (ema_parameter_after - ema_parameter_before).float().norm(),
        "EMA update norm",
    )
    if ema_update_norm <= 0:
        raise RealPilotError("EMA teacher did not change after the optimizer step")

    hashes = {
        "filtered_manifest_sha256": manifest.sha256,
        "subject_split_sha256": split.sha256,
        "case_grid_manifest_sha256": extraction_binding.catalog.sha256,
        "extraction_policy_sha256": extraction_binding.catalog.policy.sha256,
        "case_grid_record_sha256": extraction_binding.record.sha256,
        "extraction_spec_sha256": extraction_binding.extraction_spec_sha256,
        "experiment_config_sha256": config.sha256,
        "experiment_config_file_sha256": sha256_file(config_file),
        "materialized_patch_plan_sha256": prepared.plan.sha256,
        "prepared_plan_audit_sha256": prepared.sha256,
        "candidate_centers_sha256": prepared.candidate_centers_sha256,
        "uv_lock_sha256": sha256_file(repo_root_path / "uv.lock"),
    }
    metrics = {
        "loss": loss,
        "accuracy": _finite_float(output.matching.accuracy, "matching accuracy"),
        "chance": _finite_float(output.matching.chance, "matching chance"),
        "gradient_norm_before_clip": gradient_norm,
        "gradient_clip_max_norm": max_gradient_norm,
        "ema_update_norm": ema_update_norm,
        "teacher_updates": int(system.target_teacher.num_updates),
        "trainable_parameters": sum(
            parameter.numel() for parameter in system.parameters() if parameter.requires_grad
        ),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.type == "cuda"
            else 0
        ),
    }
    report: dict[str, object] = {
        "schema": "simple-brats.real-io-pilot",
        "schema_version": 1,
        "status": "ok",
        "case": selected_case.to_dict(),
        "sampling": {
            "epoch": epoch,
            "bag_index": bag_index,
            "experiment_seed": config.seed,
            "candidate_count": prepared.candidate_count,
            "candidate_pool_size": candidate_pool_size,
            "max_plan_attempts": max_plan_attempts,
            "source_patches": len(prepared.plan.sources),
            "query_patches": len(prepared.plan.queries),
            "target_patches": len(prepared.plan.targets),
        },
        "hashes": hashes,
        "case_grid": {
            "native_grid": extraction_binding.record.native_grid.to_dict(),
            "prepared_grid": extraction_binding.record.prepared_grid.to_dict(),
        },
        "volume_digests": [item.to_dict() for item in prepared.volume_digests],
        "shapes": {
            "source_patches": list(batch.source_patches.shape),
            "target_patches": list(batch.target_patches.shape),
            "source_tokens": list(output.source_tokens.shape),
            "predictions": list(output.predictions.shape),
            "teacher_targets": list(output.targets.shape),
        },
        "metrics": metrics,
        "timing_seconds": {
            "load_and_validate_artifacts": artifact_seconds,
            "prepare_volumes_candidates_and_plan": plan_seconds,
            "assemble_replay_batch": batch_seconds,
            "host_to_device": transfer_seconds,
            "build_model_and_optimizer": model_seconds,
            "forward": forward_seconds,
            "backward_and_gradient_clip": backward_seconds,
            "optimizer_step": optimizer_seconds,
            "ema_update": ema_seconds,
            "total": time.perf_counter() - start_total,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "provenance": {
            "launch_sha": launch_sha,
            "device": str(resolved_device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
            ),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "wandb_mode": os.environ.get("WANDB_MODE"),
            "paths": {
                "data_root": str(extractor.data_root),
                "filtered_manifest": str(manifest_file),
                "subject_split": str(split_file),
                "case_grid_manifest": str(extraction_binding.path),
                "experiment_config": str(config_file),
            },
        },
        "artifacts": {
            "materialized_patch_plan": plan_path.name,
            "prepared_plan_audit": audit_path.name,
            "pilot_report": "pilot-report.json",
        },
    }
    report_path = destination / "pilot-report.json"
    _write_new_canonical(report_path, report)
    print(canonical_json_bytes(report).decode("utf-8"), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one pinned real-data I/O/forward/backward/EMA pilot step."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", default="configs/v0_cross_matching_small.toml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case-id")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--bag-index", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-gradient-norm", type=float, default=10.0)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--max-plan-attempts", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_real_io_pilot(
        data_root=args.data_root,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        split_path=args.split,
        expected_split_sha256=args.expected_split_sha256,
        case_grid_manifest_path=args.case_grid_manifest,
        expected_case_grid_manifest_sha256=args.expected_case_grid_manifest_sha256,
        config_path=args.config,
        output_dir=args.output_dir,
        expected_git_sha=args.expected_git_sha,
        repo_root=args.repo_root,
        device=args.device,
        case_id=args.case_id,
        epoch=args.epoch,
        bag_index=args.bag_index,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_gradient_norm=args.max_gradient_norm,
        candidate_pool_size=args.candidate_pool_size,
        max_plan_attempts=args.max_plan_attempts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
