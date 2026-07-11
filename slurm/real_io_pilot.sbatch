#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-real-pilot
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${DATA_GATE_BUNDLE:?DATA_GATE_BUNDLE was not exported}"
: "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 was not exported}"
: "${EXPECTED_SPLIT_SHA256:?EXPECTED_SPLIT_SHA256 was not exported}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?EXPECTED_CASE_GRID_MANIFEST_SHA256 was not exported}"
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${OUTPUT_BUNDLE_PATH:?OUTPUT_BUNDLE_PATH was not exported}"
: "${PILOT_EPOCH:=0}"
: "${PILOT_BAG_INDEX:=0}"
: "${PILOT_CASE_ID:=}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase commit ID" >&2
  exit 2
fi
for value in \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${EXPECTED_SPLIT_SHA256}" \
  "${EXPECTED_CASE_GRID_MANIFEST_SHA256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every expected artifact SHA must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
if [[ ! "${PILOT_EPOCH}" =~ ^[0-9]+$ || ! "${PILOT_BAG_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "PILOT_EPOCH and PILOT_BAG_INDEX must be non-negative integers" >&2
  exit 2
fi
if [[ "${CONFIG_RELATIVE_PATH}" != "configs/v0_cross_matching_small.toml" ]]; then
  echo "The real I/O pilot is locked to the registered small-model config" >&2
  exit 2
fi

actual_sha="$(git rev-parse HEAD)"
if [[ "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout mismatch: expected ${LAUNCH_SHA}, got ${actual_sha}" >&2
  exit 2
fi
if git symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "Runtime checkout must be detached at literal LAUNCH_SHA" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Runtime launch tree contains modified tracked files" >&2
  exit 2
fi
if ! git ls-files --error-unmatch -- "${CONFIG_RELATIVE_PATH}" >/dev/null 2>&1; then
  echo "Pilot config is not tracked by the runtime checkout" >&2
  exit 2
fi
config_mode="$(git ls-tree "${LAUNCH_SHA}" -- "${CONFIG_RELATIVE_PATH}" | awk '{print $1}')"
if [[ "${config_mode}" != "100644" && "${config_mode}" != "100755" ]]; then
  echo "Pilot config must be a regular file committed at LAUNCH_SHA" >&2
  exit 2
fi

if [[ -L "${DATA_ROOT}" || ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT must be an existing directory, not a symlink" >&2
  exit 2
fi
if [[ -L "${DATA_GATE_BUNDLE}" || ! -d "${DATA_GATE_BUNDLE}" ]]; then
  echo "DATA_GATE_BUNDLE must be an existing directory, not a symlink" >&2
  exit 2
fi
for artifact in filtered.manifest.json subject-split.json case-grid-manifest.json; do
  artifact_path="${DATA_GATE_BUNDLE}/${artifact}"
  if [[ -L "${artifact_path}" || ! -f "${artifact_path}" ]]; then
    echo "Missing regular data-gate artifact: ${artifact_path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_BUNDLE_PATH}" || -L "${OUTPUT_BUNDLE_PATH}" ]]; then
  echo "Refusing to overwrite pilot output: ${OUTPUT_BUNDLE_PATH}" >&2
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

pilot_args=(
  --data-root "${DATA_ROOT}"
  --manifest "${DATA_GATE_BUNDLE}/filtered.manifest.json"
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}"
  --split "${DATA_GATE_BUNDLE}/subject-split.json"
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA256}"
  --case-grid-manifest "${DATA_GATE_BUNDLE}/case-grid-manifest.json"
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}"
  --config "${CONFIG_RELATIVE_PATH}"
  --output-dir "${OUTPUT_BUNDLE_PATH}"
  --expected-git-sha "${LAUNCH_SHA}"
  --repo-root .
  --device cuda
  --epoch "${PILOT_EPOCH}"
  --bag-index "${PILOT_BAG_INDEX}"
)
if [[ -n "${PILOT_CASE_ID}" ]]; then
  pilot_args+=(--case-id "${PILOT_CASE_ID}")
fi

srun .venv/bin/python -m simple_brats.real_pilot "${pilot_args[@]}"
