#!/usr/bin/env bash
#SBATCH --job-name=brats-heldout-eval
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${DATA_GATE_BUNDLE:?DATA_GATE_BUNDLE was not exported}"
: "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 was not exported}"
: "${EXPECTED_SPLIT_SHA256:?EXPECTED_SPLIT_SHA256 was not exported}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Expected case-grid SHA was not exported}"
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${EVALUATION_PATCH_MANIFEST_PATH:?Patch manifest was not exported}"
: "${EXPECTED_EVALUATION_PATCH_MANIFEST_SHA256:?Patch manifest SHA was not exported}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH was not exported}"
: "${EVALUATION_REPORT_OUTPUT:?EVALUATION_REPORT_OUTPUT was not exported}"
: "${ALLOW_PARTIAL_SSL_TRAIN:=0}"

actual_sha="$(git rev-parse HEAD)"
if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ || "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout does not match the full literal LAUNCH_SHA" >&2
  exit 2
fi
if git symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "Runtime checkout must be detached" >&2
  exit 2
fi
for artifact in \
  "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  "${DATA_GATE_BUNDLE}/subject-split.json" \
  "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  "${EVALUATION_PATCH_MANIFEST_PATH}" \
  "${CHECKPOINT_PATH}"; do
  if [[ -L "${artifact}" || ! -f "${artifact}" ]]; then
    echo "Evaluation input must be a regular non-symlink file: ${artifact}" >&2
    exit 2
  fi
done
if [[ -e "${EVALUATION_REPORT_OUTPUT}" || -L "${EVALUATION_REPORT_OUTPUT}" ]]; then
  echo "Refusing to overwrite evaluation report" >&2
  exit 2
fi
if [[ ! "${ALLOW_PARTIAL_SSL_TRAIN}" =~ ^[01]$ ]]; then
  echo "ALLOW_PARTIAL_SSL_TRAIN must be 0 or 1" >&2
  exit 2
fi

partial_args=()
if [[ "${ALLOW_PARTIAL_SSL_TRAIN}" == 1 ]]; then
  partial_args+=(--allow-partial-ssl-train)
fi
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun .venv/bin/python -m simple_brats.evaluation.cli checkpoint \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --split "${DATA_GATE_BUNDLE}/subject-split.json" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA256}" \
  --case-grid-manifest "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  --config "${CONFIG_RELATIVE_PATH}" \
  --evaluation-patch-manifest "${EVALUATION_PATCH_MANIFEST_PATH}" \
  --expected-evaluation-patch-manifest-sha256 \
    "${EXPECTED_EVALUATION_PATCH_MANIFEST_SHA256}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device cuda \
  --batch-size 256 \
  --random-encoder-seed 17 \
  --wandb \
  --output "${EVALUATION_REPORT_OUTPUT}" \
  "${partial_args[@]}"
