#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-long
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
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
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${OUTPUT_BUNDLE_PATH:?OUTPUT_BUNDLE_PATH was not exported}"
: "${TOTAL_STEPS:=50000}"
: "${MAX_STEPS_PER_INVOCATION:=5000}"
: "${BAGS_PER_SUBJECT:=8}"
: "${EXPECTED_TRAIN_CASES:=1044}"
: "${EXPECTED_TRAIN_SUBJECTS:=643}"

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
if [[ "${CONFIG_RELATIVE_PATH}" != "configs/v0_cross_matching_small.toml" ]]; then
  echo "Long pretraining is locked to the validated 4 mm small-model config" >&2
  exit 2
fi
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

if [[ -L "${OUTPUT_BUNDLE_PATH}/result.json" ]]; then
  echo "Long-run result must not be a symlink" >&2
  exit 2
fi
if [[ -f "${OUTPUT_BUNDLE_PATH}/result.json" ]]; then
  echo "Long run is already complete: ${OUTPUT_BUNDLE_PATH}/result.json"
  exit 0
fi

# Keep this array non-empty: CUBIC's older Bash treats expansion of a declared
# but empty array as an unbound variable under ``set -u``.
execution_args=(--device cuda)
if [[ -e "${OUTPUT_BUNDLE_PATH}" || -L "${OUTPUT_BUNDLE_PATH}" ]]; then
  if [[ -L "${OUTPUT_BUNDLE_PATH}" || ! -d "${OUTPUT_BUNDLE_PATH}" ]]; then
    echo "Existing output bundle must be a non-symlink directory" >&2
    exit 2
  fi
  latest_checkpoint=""
  if [[ -d "${OUTPUT_BUNDLE_PATH}/checkpoints" && \
        ! -L "${OUTPUT_BUNDLE_PATH}/checkpoints" ]]; then
    latest_checkpoint="$(
      find "${OUTPUT_BUNDLE_PATH}/checkpoints" -maxdepth 1 -type f \
        -name 'step-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt' \
        -print | LC_ALL=C sort | tail -n 1
    )"
  fi
  if [[ -z "${latest_checkpoint}" ]]; then
    execution_args+=(--resume-existing-output)
  else
    execution_args+=(--resume-from "${latest_checkpoint}")
  fi
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE

set +e
srun .venv/bin/python -m simple_brats.long_run \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --split "${DATA_GATE_BUNDLE}/subject-split.json" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA256}" \
  --case-grid-manifest "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  --config "${CONFIG_RELATIVE_PATH}" \
  --output-dir "${OUTPUT_BUNDLE_PATH}" \
  --expected-git-sha "${LAUNCH_SHA}" \
  --repo-root . \
  "${execution_args[@]}" \
  --total-steps "${TOTAL_STEPS}" \
  --max-steps-per-invocation "${MAX_STEPS_PER_INVOCATION}" \
  --bags-per-subject "${BAGS_PER_SUBJECT}" \
  --expected-train-cases "${EXPECTED_TRAIN_CASES}" \
  --expected-train-subjects "${EXPECTED_TRAIN_SUBJECTS}"
run_status=$?
set -e
if (( run_status == 75 )); then
  echo "Walltime checkpoint completed; requeueing Slurm job ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${run_status}"
