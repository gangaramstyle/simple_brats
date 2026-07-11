"""Small command line surface with fail-fast config and launch checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import load_experiment_config
from .data import (
    create_subject_split,
    discover_met_release,
    load_manifest,
    save_manifest,
    save_split,
)
from .provenance import collect_provenance
from .training import run_synthetic_smoke


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_config(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    _print_json({"config": config.to_dict(), "config_sha256": config.sha256})
    return 0


def _smoke(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    root = Path(args.repo_root).resolve()
    positions = args.positions if args.positions is not None else config.task.positions_per_bag
    device = _resolve_device(args.device)
    execution = {
        "command": "smoke",
        "device": device,
        "batch_size": args.batch_size,
        "positions": positions,
        "tiny_model": args.tiny_model,
    }
    provenance = collect_provenance(
        config,
        execution=execution,
        synthetic_dataset_id=args.synthetic_dataset_id,
        root=root,
        expected_git_sha=args.expected_git_sha,
    )
    metrics = run_synthetic_smoke(
        config,
        device=device,
        batch_size=args.batch_size,
        positions=positions,
        tiny_model=args.tiny_model,
    )
    _print_json(
        {
            "execution": execution,
            "metrics": metrics,
            "provenance": provenance.to_dict(),
        }
    )
    return 0


def _write_new(path: Path, writer: Any, value: Any, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(value, path)


def _build_met_manifest(args: argparse.Namespace) -> int:
    manifest = discover_met_release(
        args.root,
        source=args.source,
        release=args.release,
    )
    output = Path(args.output)
    _write_new(output, save_manifest, manifest, force=args.force)
    _print_json(
        {
            "cases": len(manifest.cases),
            "manifest_sha256": manifest.sha256,
            "output": str(output.resolve()),
            "subjects": len(manifest.subjects),
        }
    )
    return 0


def _fractions(values: list[str] | None) -> list[tuple[str, str]]:
    values = values or ["train=0.8", "validation=0.1", "test=0.1"]
    result: list[tuple[str, str]] = []
    for value in values:
        name, separator, fraction = value.partition("=")
        if not separator or not name or not fraction:
            raise ValueError("split fractions must use NAME=FRACTION syntax")
        result.append((name, fraction))
    return result


def _build_split(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest, expected_sha256=args.expected_manifest_sha)
    split = create_subject_split(
        manifest,
        seed=args.seed,
        fractions=_fractions(args.fraction),
    )
    output = Path(args.output)
    _write_new(output, save_split, split, force=args.force)
    counts = {name: 0 for name in split.split_names}
    for assignment in split.assignments:
        counts[assignment.split] += 1
    _print_json(
        {
            "manifest_sha256": manifest.sha256,
            "output": str(output.resolve()),
            "split_sha256": split.sha256,
            "subjects_by_split": counts,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-brats")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="resolve and validate a TOML config")
    validate.add_argument("--config", default="configs/v0_cross_matching.toml")
    validate.set_defaults(handler=_validate_config)

    smoke = subparsers.add_parser("smoke", help="run one end-to-end synthetic training step")
    smoke.add_argument("--config", default="configs/v0_cross_matching.toml")
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--positions", type=int)
    smoke.add_argument("--tiny-model", action="store_true")
    smoke.add_argument("--synthetic-dataset-id", default="synthetic-smoke-v0")
    smoke.add_argument("--expected-git-sha")
    smoke.add_argument("--repo-root", default=".")
    smoke.set_defaults(handler=_smoke)

    manifest = subparsers.add_parser(
        "build-met-manifest",
        help="strictly discover and hash one complete four-sequence MET release",
    )
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--source", required=True)
    manifest.add_argument("--release", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--force", action="store_true")
    manifest.set_defaults(handler=_build_met_manifest)

    split = subparsers.add_parser(
        "build-split",
        help="create a deterministic canonical-subject split bound to one manifest",
    )
    split.add_argument("--manifest", required=True)
    split.add_argument("--expected-manifest-sha")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument(
        "--fraction",
        action="append",
        help="repeat NAME=FRACTION; default train=.8, validation=.1, test=.1",
    )
    split.add_argument("--force", action="store_true")
    split.set_defaults(handler=_build_split)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
