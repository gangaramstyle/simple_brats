#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-data-gate
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${RAW_MANIFEST_PATH:?RAW_MANIFEST_PATH was not exported}"
: "${EXPECTED_RAW_MANIFEST_SHA256:?EXPECTED_RAW_MANIFEST_SHA256 was not exported}"
: "${FILTER_SPEC_RELATIVE_PATH:?FILTER_SPEC_RELATIVE_PATH was not exported}"
: "${EXPECTED_FILTER_SPEC_SHA256:?EXPECTED_FILTER_SPEC_SHA256 was not exported}"
: "${SPLIT_SEED:?SPLIT_SEED was not exported}"
: "${OUTPUT_BUNDLE_PATH:?OUTPUT_BUNDLE_PATH was not exported}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full 40-character lowercase commit ID" >&2
  exit 2
fi
if [[ ! "${EXPECTED_RAW_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_RAW_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ ! "${EXPECTED_FILTER_SPEC_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_FILTER_SPEC_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ ! "${SPLIT_SEED}" =~ ^[0-9]+$ ]]; then
  echo "SPLIT_SEED must be a non-negative integer" >&2
  exit 2
fi

actual_sha="$(git rev-parse HEAD)"
if [[ "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout mismatch: expected ${LAUNCH_SHA}, got ${actual_sha}" >&2
  exit 2
fi
if git symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "Runtime checkout must be detached at the literal LAUNCH_SHA" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Runtime launch tree contains modified tracked files" >&2
  exit 2
fi
if ! git ls-files --error-unmatch -- "${FILTER_SPEC_RELATIVE_PATH}" >/dev/null 2>&1; then
  echo "Filter spec is not tracked by the runtime checkout" >&2
  exit 2
fi
filter_mode="$(git ls-tree "${LAUNCH_SHA}" -- "${FILTER_SPEC_RELATIVE_PATH}" | awk '{print $1}')"
if [[ "${filter_mode}" != "100644" && "${filter_mode}" != "100755" ]]; then
  echo "Filter spec must be a regular file committed at LAUNCH_SHA" >&2
  exit 2
fi

if [[ -L "${DATA_ROOT}" || ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT must be an existing directory, not a symlink" >&2
  exit 2
fi
if [[ -L "${RAW_MANIFEST_PATH}" || ! -f "${RAW_MANIFEST_PATH}" ]]; then
  echo "RAW_MANIFEST_PATH must be an existing regular file, not a symlink" >&2
  exit 2
fi
if [[ -e "${OUTPUT_BUNDLE_PATH}" || -L "${OUTPUT_BUNDLE_PATH}" ]]; then
  echo "Refusing to overwrite existing output bundle: ${OUTPUT_BUNDLE_PATH}" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python || ! -f .venv/.simple-brats-lock-sha256 ]]; then
  echo "Missing pinned uv environment" >&2
  exit 2
fi
lock_sha="$(sha256sum uv.lock | awk '{print $1}')"
installed_lock_sha="$(<.venv/.simple-brats-lock-sha256)"
if [[ "${installed_lock_sha}" != "${lock_sha}" ]]; then
  echo "Runtime uv environment does not match the pinned lock" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

srun .venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from simple_brats.data.extraction import ExtractionSpec
from simple_brats.data.filtering import (
    apply_manifest_filter,
    load_manifest_filter_spec,
)
from simple_brats.data.grid_audit import infer_common_extraction_spec
from simple_brats.data.manifest import canonical_json_bytes, load_manifest
from simple_brats.data.splits import (
    create_subject_split,
    partition_cases,
    validate_split,
)


def require_canonical_file(
    path: Path,
    canonical_payload: bytes,
    description: str,
    *,
    allow_final_newline: bool = False,
) -> None:
    accepted = {canonical_payload}
    if allow_final_newline:
        accepted.add(canonical_payload + b"\n")
    if path.read_bytes() not in accepted:
        raise RuntimeError(f"{description} is valid JSON but not canonical on disk: {path}")


repo_root = Path.cwd().resolve(strict=True)
data_root = Path(os.environ["DATA_ROOT"]).resolve(strict=True)
raw_manifest_path = Path(os.environ["RAW_MANIFEST_PATH"]).resolve(strict=True)
filter_relative = Path(os.environ["FILTER_SPEC_RELATIVE_PATH"])
filter_spec_path = (repo_root / filter_relative).resolve(strict=True)
try:
    filter_spec_path.relative_to(repo_root)
except ValueError as error:
    raise RuntimeError("filter spec escapes the pinned checkout") from error
if filter_spec_path.is_symlink() or not filter_spec_path.is_file():
    raise RuntimeError("filter spec must be a committed regular file")

raw_manifest = load_manifest(
    raw_manifest_path,
    expected_sha256=os.environ["EXPECTED_RAW_MANIFEST_SHA256"],
)
raw_manifest_payload = canonical_json_bytes(raw_manifest.to_dict())
require_canonical_file(raw_manifest_path, raw_manifest_payload, "raw manifest")

filter_spec = load_manifest_filter_spec(
    filter_spec_path,
    expected_sha256=os.environ["EXPECTED_FILTER_SPEC_SHA256"],
)
filter_spec_payload = canonical_json_bytes(filter_spec.to_dict())
require_canonical_file(
    filter_spec_path,
    filter_spec_payload,
    "manifest filter spec",
    allow_final_newline=True,
)
if not filter_spec.exclusions:
    raise RuntimeError("the real-data gate requires at least one explicit subject exclusion")

filtered_manifest = apply_manifest_filter(raw_manifest, filter_spec)
expected_remaining_subjects = set(raw_manifest.subjects) - set(
    filter_spec.excluded_subject_ids
)
if set(filtered_manifest.subjects) != expected_remaining_subjects:
    raise RuntimeError("filtered manifest subject set does not match the filter specification")

split_seed = int(os.environ["SPLIT_SEED"])
split = create_subject_split(filtered_manifest, seed=split_seed)
validate_split(filtered_manifest, split)
partitions = partition_cases(filtered_manifest, split)

extraction_spec = infer_common_extraction_spec(filtered_manifest, data_root)
if not isinstance(extraction_spec, ExtractionSpec):
    raise RuntimeError("grid audit did not return an ExtractionSpec")

subject_counts = {name: 0 for name in split.split_names}
for assignment in split.assignments:
    subject_counts[assignment.split] += 1
case_counts = {name: len(partitions[name]) for name in split.split_names}

report = {
    "schema": "simple_brats.filtered_data_gate",
    "schema_version": 1,
    "launch_sha": os.environ["LAUNCH_SHA"],
    "hashes": {
        "raw_manifest_sha256": raw_manifest.sha256,
        "manifest_filter_spec_sha256": filter_spec.sha256,
        "filtered_manifest_sha256": filtered_manifest.sha256,
        "subject_split_sha256": split.sha256,
        "extraction_spec_sha256": extraction_spec.sha256,
    },
    "counts": {
        "raw_cases": len(raw_manifest.cases),
        "raw_subjects": len(raw_manifest.subjects),
        "excluded_subjects": len(filter_spec.excluded_subject_ids),
        "filtered_cases": len(filtered_manifest.cases),
        "filtered_subjects": len(filtered_manifest.subjects),
        "subjects_by_split": subject_counts,
        "cases_by_split": case_counts,
        "grid_audited_mri_files": len(filtered_manifest.cases) * 4,
    },
    "excluded_subject_ids": list(filter_spec.excluded_subject_ids),
    "split_seed": split.seed,
    "split_fractions": [fraction.to_dict() for fraction in split.fractions],
    "artifacts": {
        "filtered_manifest": "filtered.manifest.json",
        "subject_split": "subject-split.json",
        "extraction_spec": "extraction-spec.json",
        "gate_report": "gate-report.json",
    },
}

payloads = {
    "filtered.manifest.json": canonical_json_bytes(filtered_manifest.to_dict()),
    "subject-split.json": canonical_json_bytes(split.to_dict()),
    "extraction-spec.json": canonical_json_bytes(extraction_spec.to_dict()),
    "gate-report.json": canonical_json_bytes(report),
}
report_sha256 = hashlib.sha256(payloads["gate-report.json"]).hexdigest()

bundle_path = Path(os.environ["OUTPUT_BUNDLE_PATH"])
if not bundle_path.is_absolute():
    raise RuntimeError("OUTPUT_BUNDLE_PATH must be absolute")
bundle_parent = bundle_path.parent.resolve(strict=True)
bundle_path = bundle_parent / bundle_path.name
if os.path.lexists(bundle_path):
    raise FileExistsError(f"refusing to overwrite output bundle {bundle_path}")

job_id = os.environ.get("SLURM_JOB_ID", "no-slurm-id")
staging_path = bundle_parent / f".{bundle_path.name}.tmp.{job_id}.{os.getpid()}"
publish_lock = bundle_parent / f".{bundle_path.name}.publish-lock"
if os.path.lexists(staging_path):
    raise FileExistsError(f"staging path already exists: {staging_path}")
staging_path.mkdir(mode=0o700)

lock_fd: int | None = None
published = False
try:
    for name, payload in payloads.items():
        artifact = staging_path / name
        with artifact.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    directory_fd = os.open(staging_path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    lock_fd = os.open(publish_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lock_fd, f"job_id={job_id}\n".encode())
    os.fsync(lock_fd)
    if os.path.lexists(bundle_path):
        raise FileExistsError(f"refusing to overwrite output bundle {bundle_path}")
    os.rename(staging_path, bundle_path)
    published = True
    parent_fd = os.open(bundle_parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
finally:
    if lock_fd is not None:
        os.close(lock_fd)
    try:
        publish_lock.unlink()
    except FileNotFoundError:
        pass
    if not published:
        shutil.rmtree(staging_path, ignore_errors=True)

print(
    json.dumps(
        {
            **report,
            "gate_report_sha256": report_sha256,
            "output_bundle": str(bundle_path),
            "status": "ok",
        },
        sort_keys=True,
        allow_nan=False,
    )
)
PY
