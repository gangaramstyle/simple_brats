#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-content-audit
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${FILTERED_MANIFEST_PATH:?FILTERED_MANIFEST_PATH was not exported}"
: "${SUBJECT_SPLIT_PATH:?SUBJECT_SPLIT_PATH was not exported}"
: "${CASE_GRID_MANIFEST_PATH:?CASE_GRID_MANIFEST_PATH was not exported}"
: "${EXPECTED_FILTERED_MANIFEST_SHA256:?Expected manifest SHA was not exported}"
: "${EXPECTED_SUBJECT_SPLIT_SHA256:?Expected split SHA was not exported}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Expected case-grid manifest SHA was not exported}"
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${CONTENT_AUDIT_OUTPUT_PATH:?CONTENT_AUDIT_OUTPUT_PATH was not exported}"
: "${CONTENT_AUDIT_STATE_DIR:?CONTENT_AUDIT_STATE_DIR was not exported}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full 40-character lowercase commit ID" >&2
  exit 2
fi
for variable_name in \
  EXPECTED_FILTERED_MANIFEST_SHA256 \
  EXPECTED_SUBJECT_SPLIT_SHA256 \
  EXPECTED_CASE_GRID_MANIFEST_SHA256; do
  value="${!variable_name}"
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${variable_name} must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
if [[ ! "${CONFIG_RELATIVE_PATH}" =~ ^configs/[A-Za-z0-9._-]+\.toml$ ]]; then
  echo "CONFIG_RELATIVE_PATH must name one TOML file directly under configs/" >&2
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

if [[ -L "${DATA_ROOT}" || ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT must be an existing directory, not a symlink" >&2
  exit 2
fi
for artifact in \
  "${FILTERED_MANIFEST_PATH}" \
  "${SUBJECT_SPLIT_PATH}" \
  "${CASE_GRID_MANIFEST_PATH}"; do
  if [[ -L "${artifact}" || ! -f "${artifact}" ]]; then
    echo "Pinned audit input must be a regular file, not a symlink: ${artifact}" >&2
    exit 2
  fi
done
if [[ -L "${CONFIG_RELATIVE_PATH}" || ! -f "${CONFIG_RELATIVE_PATH}" ]] || \
  ! git ls-files --error-unmatch -- "${CONFIG_RELATIVE_PATH}" >/dev/null 2>&1; then
  echo "Experiment config must be a committed regular file at LAUNCH_SHA" >&2
  exit 2
fi
if [[ -e "${CONTENT_AUDIT_OUTPUT_PATH}" || -L "${CONTENT_AUDIT_OUTPUT_PATH}" ]]; then
  echo "Refusing to overwrite existing content-audit output" >&2
  exit 2
fi
if [[ -L "${CONTENT_AUDIT_STATE_DIR}" ]]; then
  echo "CONTENT_AUDIT_STATE_DIR must not be a symlink" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

srun .venv/bin/python -m simple_brats.data.content_audit \
  --manifest "${FILTERED_MANIFEST_PATH}" \
  --expected-manifest-sha256 "${EXPECTED_FILTERED_MANIFEST_SHA256}" \
  --split "${SUBJECT_SPLIT_PATH}" \
  --expected-split-sha256 "${EXPECTED_SUBJECT_SPLIT_SHA256}" \
  --case-grid-manifest "${CASE_GRID_MANIFEST_PATH}" \
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  --config "${CONFIG_RELATIVE_PATH}" \
  --data-root "${DATA_ROOT}" \
  --expected-git-sha "${LAUNCH_SHA}" \
  --repo-root . \
  --state-dir "${CONTENT_AUDIT_STATE_DIR}" \
  --output "${CONTENT_AUDIT_OUTPUT_PATH}"
