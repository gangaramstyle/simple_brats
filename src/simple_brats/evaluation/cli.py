"""Command-line entrypoints for materializing and evaluating held-out patches."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from simple_brats.atomic_io import atomic_create_bytes
from simple_brats.config import load_experiment_config
from simple_brats.data.case_grids import load_case_grid_manifest
from simple_brats.data.manifest import canonical_json_bytes, load_manifest
from simple_brats.data.splits import load_split
from simple_brats.tracking import (
    OnlineWandbConfig,
    TrackingError,
    online_run_url,
    require_verified_online_login,
)
from simple_brats.training import preserve_runner_rng_state

from .checkpoint import (
    build_random_online_encoder,
    evaluate_checkpoint_feature_tables,
    extract_evaluation_feature_tables,
    load_online_encoder_checkpoint,
)
from .patches import (
    BinaryPatchLabelRule,
    build_evaluation_patch_manifest,
    load_evaluation_patch_manifest,
    save_evaluation_patch_manifest,
)


def _new_output(path: str | os.PathLike[str]) -> Path:
    output = Path(path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _write_new_canonical(path: Path, value: Mapping[str, object]) -> None:
    atomic_create_bytes(path, canonical_json_bytes(value))


def _load_data_inputs(args: argparse.Namespace):
    manifest = load_manifest(args.manifest, expected_sha256=args.expected_manifest_sha256)
    split = load_split(args.split, expected_sha256=args.expected_split_sha256)
    grids = load_case_grid_manifest(
        args.case_grid_manifest,
        expected_sha256=args.expected_case_grid_manifest_sha256,
    )
    config = load_experiment_config(args.config)
    return manifest, split, grids, config


def _materialize(args: argparse.Namespace) -> int:
    manifest, split, grids, config = _load_data_inputs(args)
    output = _new_output(args.output)
    result = build_evaluation_patch_manifest(
        data_root=args.data_root,
        manifest=manifest,
        split=split,
        case_grids=grids,
        segmentation_label_audit_path=args.segmentation_label_audit,
        expected_segmentation_label_audit_sha256=(args.expected_segmentation_label_audit_sha256),
        patch_config=config.patch,
        probe_train_subject_count=args.probe_train_subject_count,
        maximum_patches_per_class_per_subject=args.maximum_per_class,
        minimum_patches_per_class_per_subject=args.minimum_per_class,
        seed=args.seed,
        label_rule=BinaryPatchLabelRule(
            positive_minimum_fraction=args.positive_minimum_fraction,
            negative_halo_mm=args.negative_halo_mm,
        ),
    )
    save_evaluation_patch_manifest(result, output)
    summary = {
        "schema": "simple-brats.evaluation-patch-materialization-result",
        "schema_version": 1,
        "evaluation_patch_manifest_sha256": result.sha256,
        "output": str(output.resolve()),
        "probe_train_subject_count": len(result.probe_train_subjects),
        "validation_subject_count": len(result.validation_subjects),
        "ineligible_probe_train_subjects": list(result.ineligible_probe_train_subjects),
        "patch_location_count": len(result.records),
        "locked_test_image_or_label_access": False,
    }
    print(canonical_json_bytes(summary).decode(), flush=True)
    return 0


def _flatten_scalars(value: object, *, prefix: str = "") -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_scalars(item, prefix=child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def _log_wandb(
    *,
    report: Mapping[str, object],
    output: Path,
    checkpoint_step: int,
    run_name: str,
    wandb_module: object,
    tracking: OnlineWandbConfig,
) -> dict[str, object]:
    report_sha256 = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("evaluation report must contain provenance before W&B logging")
    patch_sha = provenance.get("evaluation_patch_manifest_sha256")
    if not isinstance(patch_sha, str) or len(patch_sha) != 64:
        raise RuntimeError("evaluation report lacks its patch-manifest SHA")
    checkpoint_provenance_sha = provenance.get("checkpoint_provenance_sha256")
    if not isinstance(checkpoint_provenance_sha, str) or len(checkpoint_provenance_sha) != 64:
        raise RuntimeError("evaluation report lacks its checkpoint-provenance SHA")
    group = f"heldout-{checkpoint_provenance_sha[:12]}-{patch_sha[:12]}"
    run_id = hashlib.sha256(
        f"simple-brats-heldout-evaluation\0{report_sha256}".encode()
    ).hexdigest()[:24]
    with preserve_runner_rng_state():
        run = wandb_module.init(  # type: ignore[attr-defined]
            **tracking.init_kwargs(),
            name=run_name,
            id=run_id,
            group=group,
            job_type="held-out-representation-evaluation",
            dir=str(output.parent),
            config={
                "evaluation_report": str(output.name),
                "report_sha256": report_sha256,
                "provenance": dict(provenance),
            },
            reinit=True,
        )
    if run is None:
        raise RuntimeError("online W&B evaluation initialization returned no run")
    try:
        url = online_run_url(run)
    except TrackingError as error:
        with preserve_runner_rng_state():
            run.finish()
        raise RuntimeError(str(error)) from error
    tracking_record: dict[str, object] = {
        "schema": "simple-brats.online-wandb-evaluation-run",
        "schema_version": 1,
        **tracking.to_dict(),
        "actual_entity": getattr(run, "entity", None) or tracking.entity,
        "group": group,
        "run_id": run_id,
        "run_url": url,
        "report_sha256": report_sha256,
        "checkpoint_step": checkpoint_step,
        "artifact_collection": f"{group}-reports",
    }
    tracking_output = output.with_name(f"{output.name}.wandb.json")
    try:
        _write_new_canonical(tracking_output, tracking_record)
        print(f"W&B online evaluation run: {url}", flush=True)
        run.log(_flatten_scalars(report), step=checkpoint_step)
        artifact = wandb_module.Artifact(  # type: ignore[attr-defined]
            name=f"{group}-reports",
            type="evaluation",
            metadata={
                "checkpoint_step": checkpoint_step,
                "report_sha256": report_sha256,
                "producing_run_id": run_id,
            },
        )
        artifact.add_file(str(output), name="evaluation-report.json")
        run.log_artifact(
            artifact,
            aliases=[f"step-{checkpoint_step:09d}", "latest"],
        )
    finally:
        run.finish()
    return tracking_record


def _verified_online_wandb() -> tuple[object, OnlineWandbConfig]:
    try:
        import wandb
    except Exception as error:
        raise RuntimeError("W&B is required when --wandb is selected") from error
    try:
        tracking = OnlineWandbConfig.from_environment()
        with preserve_runner_rng_state():
            require_verified_online_login(wandb)
    except TrackingError as error:
        raise RuntimeError(str(error)) from error
    return wandb, tracking


def _checkpoint(args: argparse.Namespace) -> int:
    wandb_module: object | None = None
    wandb_tracking: OnlineWandbConfig | None = None
    if args.wandb:
        wandb_module, wandb_tracking = _verified_online_wandb()
    manifest, split, grids, config = _load_data_inputs(args)
    output = _new_output(args.output)
    patch_manifest = load_evaluation_patch_manifest(
        args.evaluation_patch_manifest,
        expected_sha256=args.expected_evaluation_patch_manifest_sha256,
    )
    loaded = load_online_encoder_checkpoint(
        args.checkpoint,
        config=config,
        manifest=manifest,
        split=split,
        case_grids=grids,
        evaluation_patches=patch_manifest,
        device=args.device,
        require_all_ssl_train_subjects=not args.allow_partial_ssl_train,
    )
    random_encoder = build_random_online_encoder(
        config, seed=args.random_encoder_seed, device=args.device
    )
    tables = extract_evaluation_feature_tables(
        data_root=args.data_root,
        manifest=manifest,
        case_grids=grids,
        evaluation_patches=patch_manifest,
        trained_encoder=loaded.encoder,
        random_encoder=random_encoder,
        device=args.device,
        batch_size=args.batch_size,
    )
    metrics = evaluate_checkpoint_feature_tables(
        tables,
        evaluation_patches=patch_manifest,
        subject_budgets=tuple(args.subject_budget),
        l2_penalty=args.l2_penalty,
        neighbors=tuple(args.neighbors),
    )
    report: dict[str, object] = {
        **metrics,
        "provenance": {
            "checkpoint_step": loaded.step,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "checkpoint_provenance_sha256": hashlib.sha256(
                canonical_json_bytes(loaded.provenance)
            ).hexdigest(),
            "evaluation_patch_manifest_sha256": patch_manifest.sha256,
            "segmentation_label_audit_sha256": (patch_manifest.segmentation_label_audit_sha256),
            "manifest_sha256": manifest.sha256,
            "split_sha256": split.sha256,
            "case_grid_manifest_sha256": grids.sha256,
            "config_sha256": config.sha256,
            "random_encoder_seed": args.random_encoder_seed,
            "consumed_ssl_train_subject_count": loaded.consumed_ssl_train_subject_count,
            "total_ssl_train_subject_count": loaded.total_ssl_train_subject_count,
            "complete_ssl_train_subject_coverage": (loaded.complete_ssl_train_subject_coverage),
            "evaluation_classification": (
                "scientific_held_out_representation_report"
                if loaded.complete_ssl_train_subject_coverage
                else "partial_ssl_coverage_labeled_mechanics_readout_only"
            ),
            "deterministic_runtime": dict(loaded.deterministic_runtime),
            "evaluation_launch_sha": os.environ.get("LAUNCH_SHA", "not_exported"),
        },
    }
    _write_new_canonical(output, report)
    if args.wandb:
        assert wandb_module is not None and wandb_tracking is not None
        _log_wandb(
            report=report,
            output=output,
            checkpoint_step=loaded.step,
            run_name=args.wandb_run_name or f"held-out-eval-step-{loaded.step:09d}",
            wandb_module=wandb_module,
            tracking=wandb_tracking,
        )
    print(canonical_json_bytes(report).decode(), flush=True)
    return 0


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Held-out 4mm frozen-token evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize", help="materialize labeled patch locations")
    _data_arguments(materialize)
    materialize.add_argument("--segmentation-label-audit", required=True)
    materialize.add_argument("--expected-segmentation-label-audit-sha256", required=True)
    materialize.add_argument("--probe-train-subject-count", type=int, default=128)
    materialize.add_argument("--maximum-per-class", type=int, default=32)
    materialize.add_argument("--minimum-per-class", type=int, default=4)
    materialize.add_argument("--positive-minimum-fraction", type=float, default=0.25)
    materialize.add_argument("--negative-halo-mm", type=float, default=4.0)
    materialize.add_argument("--seed", type=int, default=0)
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(handler=_materialize)

    checkpoint = commands.add_parser("checkpoint", help="evaluate one runner-v3 checkpoint")
    _data_arguments(checkpoint)
    checkpoint.add_argument("--evaluation-patch-manifest", required=True)
    checkpoint.add_argument("--expected-evaluation-patch-manifest-sha256", required=True)
    checkpoint.add_argument("--checkpoint", required=True)
    checkpoint.add_argument("--device", default="cuda")
    checkpoint.add_argument("--batch-size", type=int, default=256)
    checkpoint.add_argument("--random-encoder-seed", type=int, default=17)
    checkpoint.add_argument(
        "--allow-partial-ssl-train",
        action="store_true",
        help="permit a train-only subset checkpoint as a mechanics readout, not a result",
    )
    checkpoint.add_argument("--subject-budget", type=int, action="append", default=[])
    checkpoint.add_argument("--neighbors", type=int, action="append", default=[])
    checkpoint.add_argument("--l2-penalty", type=float, default=1.0)
    checkpoint.add_argument("--wandb", action="store_true")
    checkpoint.add_argument("--wandb-run-name")
    checkpoint.add_argument("--output", required=True)
    checkpoint.set_defaults(handler=_checkpoint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "checkpoint":
        args.subject_budget = args.subject_budget or [8, 32, 128]
        args.neighbors = args.neighbors or [1, 5, 20]
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
