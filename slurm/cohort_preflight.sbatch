#!/usr/bin/env bash
#SBATCH --job-name=brats-cohort-preflight
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE
#SBATCH --signal=USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${DATA_GATE_BUNDLE:?DATA_GATE_BUNDLE was not exported}"
: "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 was not exported}"
: "${EXPECTED_SPLIT_SHA256:?EXPECTED_SPLIT_SHA256 was not exported}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?EXPECTED_CASE_GRID_MANIFEST_SHA256 was not exported}"
: "${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 was not exported}"
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${OUTPUT_BUNDLE_PATH:?OUTPUT_BUNDLE_PATH was not exported}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase commit ID" >&2
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
case "${CONFIG_RELATIVE_PATH}:${EXPECTED_CONFIG_SHA256}" in
  configs/v0_cross_matching_small.toml:1ee8f45f2938c1d005fa975f20f3dcbeb8e378aada19b01b7d0dcc9fb28d847c) ;;
  configs/v0_cross_matching_small_8mm.toml:fdc89047dd0739c0108d077a9f9b38b611af8b241774a2f1a6bfb9c3aca568eb) ;;
  *)
    echo "Cohort preflight requires one exact registered scale-matched config and digest" >&2
    exit 2
    ;;
esac
for artifact in filtered.manifest.json subject-split.json case-grid-manifest.json; do
  if [[ -L "${DATA_GATE_BUNDLE}/${artifact}" || ! -f "${DATA_GATE_BUNDLE}/${artifact}" ]]; then
    echo "Missing regular data-gate artifact: ${DATA_GATE_BUNDLE}/${artifact}" >&2
    exit 2
  fi
done
if [[ ! -x .venv/bin/python ]]; then
  echo "Pinned environment is missing Python" >&2
  exit 2
fi
lock_sha="$(sha256sum uv.lock | awk '{print $1}')"
if [[ ! -f .venv/.simple-brats-lock-sha256 ]] || \
  [[ "$(<.venv/.simple-brats-lock-sha256)" != "${lock_sha}" ]]; then
  echo "Runtime uv environment does not match the pinned lock" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
set +e
srun .venv/bin/python -m simple_brats.cohort_preflight \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --split "${DATA_GATE_BUNDLE}/subject-split.json" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA256}" \
  --case-grid-manifest "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  --config "${CONFIG_RELATIVE_PATH}" \
  --expected-config-sha256 "${EXPECTED_CONFIG_SHA256}" \
  --output-dir "${OUTPUT_BUNDLE_PATH}" \
  --expected-git-sha "${LAUNCH_SHA}" \
  --repo-root .
run_status=$?
set -e
if (( run_status == 75 )); then
  echo "Cold-path case result completed; requeueing Slurm job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${run_status}"
