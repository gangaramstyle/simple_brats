#!/usr/bin/env bash
#SBATCH --job-name=brats-eval-patches
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
: "${SEGMENTATION_LABEL_AUDIT_PATH:?Label-audit path was not exported}"
: "${EXPECTED_SEGMENTATION_LABEL_AUDIT_SHA256:?Label-audit SHA was not exported}"
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${EVALUATION_PATCH_MANIFEST_OUTPUT:?Evaluation output was not exported}"
: "${PROBE_TRAIN_SUBJECT_COUNT:=128}"

actual_sha="$(git rev-parse HEAD)"
if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ || "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout does not match the full literal LAUNCH_SHA" >&2
  exit 2
fi
if git symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "Runtime checkout must be detached" >&2
  exit 2
fi
for value in \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${EXPECTED_SPLIT_SHA256}" \
  "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  "${EXPECTED_SEGMENTATION_LABEL_AUDIT_SHA256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every expected artifact SHA must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
for artifact in \
  "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  "${DATA_GATE_BUNDLE}/subject-split.json" \
  "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  "${SEGMENTATION_LABEL_AUDIT_PATH}"; do
  if [[ -L "${artifact}" || ! -f "${artifact}" ]]; then
    echo "Pinned evaluation input must be a regular non-symlink file: ${artifact}" >&2
    exit 2
  fi
done
if [[ -e "${EVALUATION_PATCH_MANIFEST_OUTPUT}" || \
      -L "${EVALUATION_PATCH_MANIFEST_OUTPUT}" ]]; then
  echo "Refusing to overwrite evaluation patch manifest" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Pinned Python environment is missing" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun .venv/bin/python -m simple_brats.evaluation.cli materialize \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --split "${DATA_GATE_BUNDLE}/subject-split.json" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA256}" \
  --case-grid-manifest "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  --config "${CONFIG_RELATIVE_PATH}" \
  --segmentation-label-audit "${SEGMENTATION_LABEL_AUDIT_PATH}" \
  --expected-segmentation-label-audit-sha256 \
    "${EXPECTED_SEGMENTATION_LABEL_AUDIT_SHA256}" \
  --probe-train-subject-count "${PROBE_TRAIN_SUBJECT_COUNT}" \
  --maximum-per-class 32 \
  --minimum-per-class 4 \
  --positive-minimum-fraction 0.25 \
  --negative-halo-mm 4 \
  --seed 0 \
  --output "${EVALUATION_PATCH_MANIFEST_OUTPUT}"
