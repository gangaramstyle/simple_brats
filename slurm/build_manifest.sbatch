#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-manifest
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --propagate=NONE

set -euo pipefail

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${DATA_SOURCE:?DATA_SOURCE was not exported}"
: "${DATA_RELEASE:?DATA_RELEASE was not exported}"
: "${MANIFEST_PATH:?MANIFEST_PATH was not exported}"
: "${SPLIT_PATH:?SPLIT_PATH was not exported}"

actual_sha="$(git rev-parse HEAD)"
if [[ "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout mismatch: expected ${LAUNCH_SHA}, got ${actual_sha}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Runtime launch tree contains modified tracked files" >&2
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

srun .venv/bin/python -m simple_brats.cli build-met-manifest \
  --root "${DATA_ROOT}" \
  --source "${DATA_SOURCE}" \
  --release "${DATA_RELEASE}" \
  --output "${MANIFEST_PATH}"

manifest_sha="$(.venv/bin/python -c 'import sys; from simple_brats.data import load_manifest; print(load_manifest(sys.argv[1]).sha256)' "${MANIFEST_PATH}")"

srun .venv/bin/python -m simple_brats.cli build-split \
  --manifest "${MANIFEST_PATH}" \
  --expected-manifest-sha "${manifest_sha}" \
  --output "${SPLIT_PATH}" \
  --seed 0

.venv/bin/python - "${MANIFEST_PATH}" "${SPLIT_PATH}" <<'PY'
import json
import sys

from simple_brats.data import load_manifest, load_split, validate_split

manifest = load_manifest(sys.argv[1])
split = load_split(sys.argv[2])
validate_split(manifest, split)
print(json.dumps({
    "status": "ok",
    "manifest_sha256": manifest.sha256,
    "split_sha256": split.sha256,
    "cases": len(manifest.cases),
    "subjects": len(manifest.subjects),
}, sort_keys=True))
PY
