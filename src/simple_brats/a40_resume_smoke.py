"""A40 gate for bit-exact continuation of the real 4 mm matching system.

The coordinator materializes one pinned real training bag, then launches three
fresh Python processes: an uninterrupted two-step run, a one-step run, and a
resume of that one-step checkpoint.  The gate is intentionally tiny; it tests
the numerical/runtime contract required by the long run rather than model
quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.config import ExperimentConfig, load_experiment_config
from simple_brats.data.case_grids import load_case_grid_manifest
from simple_brats.data.manifest import canonical_json_bytes, load_manifest, sha256_file
from simple_brats.data.splits import load_split, validate_split
from simple_brats.long_run import configure_exact_resume_runtime
from simple_brats.provenance import verify_git_sha
from simple_brats.short_run import DeterministicRealBatchFactory, _ordered_train_cases
from simple_brats.training import (
    CheckpointManager,
    CheckpointPolicy,
    CollapseThresholds,
    FixedTargetPatchProbe,
    MatchingBatch,
    StepMetrics,
    TrainingRuntimePolicy,
    apply_model_runtime,
    build_adamw_optimizer,
    build_matching_system,
    configure_training_runtime,
    run_matching_training,
    stats_by_modality,
)

_CONFIG_PATH = "configs/v0_cross_matching_small.toml"
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_LEARNING_RATE = 1e-4
_WEIGHT_DECAY = 0.05
_GRADIENT_CLIP_NORM = 10.0
_COLLAPSE_THRESHOLDS = CollapseThresholds(
    minimum_variance_ratio=0.10,
    minimum_effective_rank_ratio=0.25,
    maximum_off_diagonal_cosine=0.95,
)


class A40ResumeSmokeError(RuntimeError):
    """The exact-resume gate could not establish its registered contract."""


def _write_new_canonical(path: Path, value: Mapping[str, object]) -> str:
    payload = canonical_json_bytes(value)
    atomic_create_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _resolve_regular_file(path: str | os.PathLike[str], description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise A40ResumeSmokeError(f"{description} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise A40ResumeSmokeError(f"{description} is unavailable: {path}") from error
    if not resolved.is_file():
        raise A40ResumeSmokeError(f"{description} must be a regular file")
    return resolved


def _assert_registered_config(config: ExperimentConfig) -> None:
    arm = config.registered_single_d_arm
    if arm is None:
        raise A40ResumeSmokeError("resume smoke requires an exact registered single-D scale arm")
    expected = {
        "prism_extent_mm": config.task.prism_extent_mm,
        "footprint_mm": config.patch.footprint_mm,
        "thin_mm": config.patch.footprint_mm,
        "tensor_shape": (16, 16, 16),
        "width": 256,
        "depth": 8,
        "heads": 4,
        "target_patches_per_bag": 32,
        "context_patches_per_nontarget_modality": 30,
        "context_patches_target_modality": 6,
        "source_patches_per_bag": 96,
        "modalities": ("t1n", "t1c", "t2w", "t2f"),
    }
    observed = {
        "prism_extent_mm": config.task.prism_extent_mm,
        "footprint_mm": config.patch.footprint_mm,
        "thin_mm": config.patch.thin_mm,
        "tensor_shape": config.patch.tensor_shape,
        "width": config.model.width,
        "depth": config.model.depth,
        "heads": config.model.heads,
        "target_patches_per_bag": config.task.target_patches_per_bag,
        "context_patches_per_nontarget_modality": (
            config.task.context_patches_per_nontarget_modality
        ),
        "context_patches_target_modality": config.task.context_patches_target_modality,
        "source_patches_per_bag": config.task.source_patches_per_bag,
        "modalities": config.task.modalities,
    }
    if observed != expected:
        raise A40ResumeSmokeError(f"resume smoke scientific contract drifted for {arm}")
    if (
        config.task.objective != "match"
        or not config.task.allow_target_modality_elsewhere
        or config.task.allow_target_modality_at_target
        or config.task.pass_scan_statistics_to_teacher
    ):
        raise A40ResumeSmokeError("resume smoke requires the registered leakage contract")


def _load_inputs(args: argparse.Namespace) -> tuple[Any, Any, Any, ExperimentConfig, Path]:
    repo = Path(args.repo_root).expanduser().resolve(strict=True)
    verify_git_sha(args.expected_git_sha, repo)
    manifest_path = _resolve_regular_file(args.manifest, "filtered manifest")
    split_path = _resolve_regular_file(args.split, "subject split")
    grids_path = _resolve_regular_file(args.case_grid_manifest, "case-grid manifest")
    config_path = _resolve_regular_file(args.config, "experiment config")
    manifest = load_manifest(manifest_path, expected_sha256=args.expected_manifest_sha256)
    split = load_split(split_path, expected_sha256=args.expected_split_sha256)
    case_grids = load_case_grid_manifest(
        grids_path,
        expected_sha256=args.expected_case_grid_manifest_sha256,
    )
    config = load_experiment_config(config_path)
    _assert_registered_config(config)
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    return manifest, split, case_grids, config, config_path


def _real_batch(
    args: argparse.Namespace,
    *,
    plans_dir: Path,
) -> tuple[MatchingBatch, Mapping[str, object], ExperimentConfig, Path]:
    manifest, split, case_grids, config, config_path = _load_inputs(args)
    case = _ordered_train_cases(manifest, split, seed=config.seed, max_cases=1)[0]
    plans_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    factory = DeterministicRealBatchFactory(
        data_root=args.data_root,
        manifest=manifest,
        case_grids=case_grids,
        cases=(case,),
        config=config,
        plans_dir=plans_dir,
        bags_per_case=1,
        candidate_pool_size=512,
        max_plan_attempts=8,
        replay_existing=True,
    )
    batch = factory(0)
    if factory.last_record is None:
        raise A40ResumeSmokeError("real batch is missing its materialized plan record")
    return batch, dict(factory.last_record), config, config_path


def _update_digest(digest: Any, value: Any) -> None:
    """Hash a Torch checkpoint tree by values, independent of torch.save bytes."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")


def semantic_digest(value: Any) -> str:
    """Return a stable digest for nested checkpoint state."""

    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _batch_digest(batch: MatchingBatch) -> str:
    return semantic_digest(
        {field.name: getattr(batch, field.name) for field in fields(MatchingBatch)}
    )


def _metrics_record(metrics: StepMetrics) -> dict[str, object]:
    return {
        "step": metrics.step,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "chance": metrics.chance,
        "ema_update_count": metrics.ema_update_count,
        "diagnostics_by_stream": {
            stream: {
                str(modality): statistics.to_dict()
                for modality, statistics in sorted(by_modality.items())
            }
            for stream, by_modality in sorted(metrics.diagnostics_by_stream.items())
        },
    }


def _calibration_record(
    system: Any,
    batch: MatchingBatch,
    probe: FixedTargetPatchProbe,
    *,
    batch_sha256: str,
    runtime_policy: TrainingRuntimePolicy,
) -> tuple[dict[str, object], Mapping[int, Any]]:
    with torch.no_grad(), runtime_policy.autocast(batch.target_patches.device):
        output = system(batch)
        teacher_targets = system.target_teacher(
            probe.target_patches.to(batch.target_patches.device)
        )
    references = stats_by_modality(
        teacher_targets,
        probe.target_modality_ids.to(batch.target_modality_ids.device),
    )
    record: dict[str, object] = {
        "schema": "simple-brats.a40-resume-smoke-calibration",
        "schema_version": 1,
        "timing": "initialized_small_4mm_model_before_optimizer_and_training",
        "real_batch_sha256": batch_sha256,
        "fixed_probe_sha256": probe.sha256,
        "teacher_reference_by_modality": {
            str(key): value.to_dict() for key, value in sorted(references.items())
        },
        "training_teacher_by_modality": {
            str(key): value.to_dict()
            for key, value in sorted(
                stats_by_modality(output.targets, batch.target_modality_ids).items()
            )
        },
        "prediction_by_modality": {
            str(key): value.to_dict()
            for key, value in sorted(
                stats_by_modality(output.predictions, batch.query_modality_ids).items()
            )
        },
    }
    return record, references


def _stage(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    exact_runtime = configure_exact_resume_runtime(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise A40ResumeSmokeError("this gate must run on one CUDA A40 allocation")
    training_runtime = configure_training_runtime(device)
    stage_output = Path(args.stage_output).resolve()
    stage_output.mkdir(mode=0o700, parents=True, exist_ok=False)
    batch, plan_record, config, config_path = _real_batch(args, plans_dir=Path(args.plans_dir))
    batch_sha256 = _batch_digest(batch)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    system = build_matching_system(config).to(device).train()
    apply_model_runtime(system, training_runtime)
    device_batch = batch.to(device)
    probe = FixedTargetPatchProbe(batch.target_patches, batch.target_modality_ids)
    calibration, references = _calibration_record(
        system,
        device_batch,
        probe,
        batch_sha256=batch_sha256,
        runtime_policy=training_runtime,
    )
    calibration_sha256 = _write_new_canonical(stage_output / "calibration.json", calibration)

    provenance: dict[str, object] = {
        "schema": "simple-brats.a40-exact-resume-smoke",
        "schema_version": 1,
        "launch_sha": args.expected_git_sha,
        "manifest_sha256": args.expected_manifest_sha256,
        "split_sha256": args.expected_split_sha256,
        "case_grid_manifest_sha256": args.expected_case_grid_manifest_sha256,
        "config_sha256": config.sha256,
        "config_file_sha256": sha256_file(config_path),
        "real_batch_sha256": batch_sha256,
        "real_batch_plan": plan_record,
        "runtime": {
            "exact_resume": exact_runtime,
            "training": training_runtime.to_dict(),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": _LEARNING_RATE,
            "weight_decay": _WEIGHT_DECAY,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "implementation": training_runtime.to_dict()["optimizer"],
        },
        "schedule": "same_single_pinned_real_train_batch_for_two_optimizer_steps",
    }
    optimizer = build_adamw_optimizer(
        system,
        learning_rate=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
        policy=training_runtime,
    )
    manager = CheckpointManager(
        args.checkpoint_root,
        policy=CheckpointPolicy(checkpoint_every_steps=1, artifact_every_steps=1_000_000),
        artifact_sink=None,
    )
    emitted: list[dict[str, object]] = []

    def on_step(metrics: StepMetrics) -> None:
        emitted.append(_metrics_record(metrics))

    result = run_matching_training(
        system,
        optimizer,
        lambda _absolute_index: batch,
        manager,
        provenance,
        total_steps=2,
        max_steps=args.max_steps,
        resume_from=args.resume_from,
        collapse_probe=probe,
        collapse_reference=references,
        collapse_thresholds=_COLLAPSE_THRESHOLDS,
        collapse_warmup_steps=2,
        gradient_clip_norm=_GRADIENT_CLIP_NORM,
        runtime_policy=training_runtime,
        on_step=on_step,
    )
    if result.latest_checkpoint is None:
        raise A40ResumeSmokeError("stage ended without its required checkpoint")
    report: dict[str, object] = {
        "schema": "simple-brats.a40-resume-smoke-stage",
        "schema_version": 1,
        "pid": os.getpid(),
        "stage": args.stage,
        "start_step": result.start_step,
        "end_step": result.end_step,
        "ema_update_count": result.ema_update_count,
        "runner_contract_sha256": result.runner_contract_sha256,
        "calibration_sha256": calibration_sha256,
        "real_batch_sha256": batch_sha256,
        "checkpoint": str(result.latest_checkpoint.resolve()),
        "metrics": emitted,
    }
    _write_new_canonical(stage_output / "report.json", report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return report


def _common_child_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--expected-manifest-sha256",
        args.expected_manifest_sha256,
        "--split",
        str(Path(args.split).resolve()),
        "--expected-split-sha256",
        args.expected_split_sha256,
        "--case-grid-manifest",
        str(Path(args.case_grid_manifest).resolve()),
        "--expected-case-grid-manifest-sha256",
        args.expected_case_grid_manifest_sha256,
        "--config",
        str(Path(args.config).resolve()),
        "--expected-git-sha",
        args.expected_git_sha,
        "--repo-root",
        str(Path(args.repo_root).resolve()),
        "--device",
        "cuda",
    ]


def _run_child(
    args: argparse.Namespace,
    *,
    stage: str,
    stage_output: Path,
    checkpoint_root: Path,
    max_steps: int,
    plans_dir: Path,
    resume_from: Path | None = None,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "simple_brats.a40_resume_smoke",
        "stage",
        *_common_child_args(args),
        "--stage",
        stage,
        "--stage-output",
        str(stage_output),
        "--checkpoint-root",
        str(checkpoint_root),
        "--plans-dir",
        str(plans_dir),
        "--max-steps",
        str(max_steps),
    ]
    if resume_from is not None:
        command.extend(("--resume-from", str(resume_from)))
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    completed = subprocess.run(
        command,
        cwd=Path(args.repo_root).resolve(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log = {
        "schema": "simple-brats.a40-resume-smoke-child-log",
        "schema_version": 1,
        "stage": stage,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    _write_new_canonical(Path(args.output_dir) / f"{stage}-process.json", log)
    if completed.returncode:
        raise A40ResumeSmokeError(
            f"fresh process {stage!r} failed with status {completed.returncode}; "
            f"see {stage}-process.json"
        )
    report_path = stage_output / "report.json"
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        raise A40ResumeSmokeError(f"stage {stage!r} emitted a malformed report")
    return report


def _checkpoint_state(path: str) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
        raise A40ResumeSmokeError(f"malformed checkpoint: {path}")
    return payload["state"]


def compare_smoke_outputs(
    continuous_step1: Mapping[str, Any],
    split_step1: Mapping[str, Any],
    continuous_step2: Mapping[str, Any],
    resumed_step2: Mapping[str, Any],
    *,
    continuous_report: Mapping[str, Any],
    first_report: Mapping[str, Any],
    resumed_report: Mapping[str, Any],
) -> dict[str, object]:
    """Fail closed unless continuous and fresh-process continuation are exact."""

    continuous_ema = {
        key: value
        for key, value in continuous_step2["model"].items()
        if key.startswith("target_teacher.")
    }
    resumed_ema = {
        key: value
        for key, value in resumed_step2["model"].items()
        if key.startswith("target_teacher.")
    }
    checks = {
        "step1_model": semantic_digest(continuous_step1["model"])
        == semantic_digest(split_step1["model"]),
        "step1_optimizer": semantic_digest(continuous_step1["optimizer"])
        == semantic_digest(split_step1["optimizer"]),
        "step1_ema_count": continuous_step1["ema_update_count"]
        == split_step1["ema_update_count"]
        == 1,
        "step1_runner_contract": continuous_step1["runner_contract_sha256"]
        == split_step1["runner_contract_sha256"],
        "final_model": semantic_digest(continuous_step2["model"])
        == semantic_digest(resumed_step2["model"]),
        "final_optimizer": semantic_digest(continuous_step2["optimizer"])
        == semantic_digest(resumed_step2["optimizer"]),
        "final_ema_count": continuous_step2["ema_update_count"]
        == resumed_step2["ema_update_count"]
        == 2,
        "final_ema_teacher": bool(continuous_ema)
        and semantic_digest(continuous_ema) == semantic_digest(resumed_ema),
        "final_runner_contract": semantic_digest(continuous_step2["runner_contract"])
        == semantic_digest(resumed_step2["runner_contract"]),
        "final_runner_contract_sha256": continuous_step2["runner_contract_sha256"]
        == resumed_step2["runner_contract_sha256"],
        "final_rng": semantic_digest(continuous_step2["rng"])
        == semantic_digest(resumed_step2["rng"]),
        "step1_metrics": continuous_report["metrics"][0] == first_report["metrics"][0],
        "next_step_metrics": continuous_report["metrics"][1] == resumed_report["metrics"][0],
        "calibration_canonical_digest": len(
            {
                continuous_report["calibration_sha256"],
                first_report["calibration_sha256"],
                resumed_report["calibration_sha256"],
            }
        )
        == 1,
        "real_batch_digest": len(
            {
                continuous_report["real_batch_sha256"],
                first_report["real_batch_sha256"],
                resumed_report["real_batch_sha256"],
            }
        )
        == 1,
        "fresh_python_processes": len(
            {
                continuous_report["pid"],
                first_report["pid"],
                resumed_report["pid"],
            }
        )
        == 3,
        "stage_boundaries": (
            continuous_report["start_step"],
            continuous_report["end_step"],
            first_report["start_step"],
            first_report["end_step"],
            resumed_report["start_step"],
            resumed_report["end_step"],
        )
        == (0, 2, 0, 1, 1, 2),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise A40ResumeSmokeError("bit-exact resume checks failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "final_model_sha256": semantic_digest(continuous_step2["model"]),
        "final_optimizer_sha256": semantic_digest(continuous_step2["optimizer"]),
        "final_ema_teacher_sha256": semantic_digest(continuous_ema),
        "final_runner_contract_sha256": continuous_step2["runner_contract_sha256"],
        "calibration_sha256": continuous_report["calibration_sha256"],
        "real_batch_sha256": continuous_report["real_batch_sha256"],
    }


def _coordinate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cuda")
    exact_runtime = configure_exact_resume_runtime(device)
    if not torch.cuda.is_available():
        raise A40ResumeSmokeError("CUDA is unavailable in the A40 smoke allocation")
    training_runtime = configure_training_runtime(device)
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" not in gpu_name.upper():
        raise A40ResumeSmokeError(f"expected an A40 allocation, observed {gpu_name!r}")
    output = Path(args.output_dir).expanduser()
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    output.mkdir(mode=0o700, exist_ok=False)
    args.output_dir = str(output)
    plans = output / "real-batch-plan"
    batch, plan_record, config, _ = _real_batch(args, plans_dir=plans)
    materialized_batch_sha256 = _batch_digest(batch)
    _write_new_canonical(
        output / "real-batch.json",
        {
            "schema": "simple-brats.a40-resume-smoke-real-batch",
            "schema_version": 1,
            "source_split": "train",
            "config_sha256": config.sha256,
            "real_batch_sha256": materialized_batch_sha256,
            "plan": dict(plan_record),
        },
    )

    continuous = _run_child(
        args,
        stage="continuous",
        stage_output=output / "continuous",
        checkpoint_root=output / "continuous-checkpoints",
        max_steps=2,
        plans_dir=plans,
    )
    first = _run_child(
        args,
        stage="split-first",
        stage_output=output / "split-first",
        checkpoint_root=output / "split-checkpoints",
        max_steps=1,
        plans_dir=plans,
    )
    resumed = _run_child(
        args,
        stage="split-resumed",
        stage_output=output / "split-resumed",
        checkpoint_root=output / "split-checkpoints",
        max_steps=1,
        plans_dir=plans,
        resume_from=Path(str(first["checkpoint"])),
    )
    continuous_step1 = _checkpoint_state(
        str(output / "continuous-checkpoints" / "step-000000001.pt")
    )
    split_step1 = _checkpoint_state(str(first["checkpoint"]))
    continuous_step2 = _checkpoint_state(str(continuous["checkpoint"]))
    resumed_step2 = _checkpoint_state(str(resumed["checkpoint"]))
    comparison = compare_smoke_outputs(
        continuous_step1,
        split_step1,
        continuous_step2,
        resumed_step2,
        continuous_report=continuous,
        first_report=first,
        resumed_report=resumed,
    )
    if comparison["real_batch_sha256"] != materialized_batch_sha256:
        raise A40ResumeSmokeError("child process did not replay the coordinator's pinned batch")
    report: dict[str, object] = {
        "schema": "simple-brats.a40-exact-resume-smoke-result",
        "schema_version": 1,
        "status": "passed",
        "launch_sha": args.expected_git_sha,
        "runtime": {
            "exact_resume": exact_runtime,
            "training": training_runtime.to_dict(),
        },
        "gpu_name": gpu_name,
        "process_ids": [continuous["pid"], first["pid"], resumed["pid"]],
        **comparison,
    }
    _write_new_canonical(output / "result.json", report)
    print(canonical_json_bytes(report).decode(), flush=True)
    return report


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", default=_CONFIG_PATH)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="cuda")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real A40 exact-resume gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    coordinate = subparsers.add_parser("coordinate")
    _add_common_arguments(coordinate)
    coordinate.add_argument("--output-dir", required=True)
    stage = subparsers.add_parser("stage")
    _add_common_arguments(stage)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--stage-output", required=True)
    stage.add_argument("--checkpoint-root", required=True)
    stage.add_argument("--plans-dir", required=True)
    stage.add_argument("--max-steps", required=True, type=int, choices=(1, 2))
    stage.add_argument("--resume-from")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "coordinate":
        _coordinate(args)
    else:
        _stage(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
