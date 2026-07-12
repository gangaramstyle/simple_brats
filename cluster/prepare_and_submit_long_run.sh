#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed data-gate bundle}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_SPLIT_SHA256:?Set EXPECTED_SPLIT_SHA256}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set EXPECTED_CASE_GRID_MANIFEST_SHA256}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/long-runs}"
: "${OUTPUT_STEM:=brats-met-small-4mm-subject-balanced-50k-bf16-v1}"
: "${TOTAL_STEPS:=50000}"
: "${MAX_STEPS_PER_INVOCATION:=5000}"
: "${BAGS_PER_SUBJECT:=8}"
: "${EXPECTED_TRAIN_CASES:=1044}"
: "${EXPECTED_TRAIN_SUBJECTS:=643}"
: "${RESUME_EXISTING:=0}"
: "${WANDB_MODE:=online}"
: "${WANDB_PROJECT:=simple-brats}"
: "${WANDB_ENTITY:=}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase commit ID" >&2
  exit 2
fi
for value in \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${EXPECTED_SPLIT_SHA256}" \
  "${EXPECTED_CASE_GRID_MANIFEST_SHA256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every expected data artifact SHA must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
for value in \
  "${TOTAL_STEPS}" \
  "${MAX_STEPS_PER_INVOCATION}" \
  "${BAGS_PER_SUBJECT}" \
  "${EXPECTED_TRAIN_CASES}" \
  "${EXPECTED_TRAIN_SUBJECTS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Step, bag, case, and subject counts must be positive integers" >&2
    exit 2
  fi
done
if (( TOTAL_STEPS % 1000 != 0 )); then
  echo "TOTAL_STEPS must end on the 1,000-step checkpoint cadence" >&2
  exit 2
fi
if (( MAX_STEPS_PER_INVOCATION != 5000 )); then
  echo "Long-run Slurm invocations are registered at exactly 5,000 steps" >&2
  exit 2
fi
if (( BAGS_PER_SUBJECT != 8 )); then
  echo "The subject-balanced schedule is registered at eight bags per subject" >&2
  exit 2
fi
case "${CONFIG_RELATIVE_PATH}" in
  configs/v0_cross_matching_small.toml|configs/v0_cross_matching_small_8mm.toml) ;;
  *)
    echo "Long pretraining requires one of the two registered scale-matched small-model configs" >&2
    exit 2
    ;;
esac
if [[ "${RESUME_EXISTING}" != 0 && "${RESUME_EXISTING}" != 1 ]]; then
  echo "RESUME_EXISTING must be 0 or 1" >&2
  exit 2
fi
if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "Registered long pretraining requires WANDB_MODE=online" >&2
  exit 2
fi
if [[ ! "${WANDB_PROJECT}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "WANDB_PROJECT must be a non-empty safe project name" >&2
  exit 2
fi
if [[ -n "${WANDB_ENTITY}" && ! "${WANDB_ENTITY}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "WANDB_ENTITY contains unsupported characters" >&2
  exit 2
fi
if [[ ! "${OUTPUT_STEM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_STEM contains unsafe characters" >&2
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

data_root="$(cd "${DATA_ROOT}" && pwd -P)"
data_gate_bundle="$(cd "${DATA_GATE_BUNDLE}" && pwd -P)"
for artifact in filtered.manifest.json subject-split.json case-grid-manifest.json; do
  if [[ -L "${data_gate_bundle}/${artifact}" || ! -f "${data_gate_bundle}/${artifact}" ]]; then
    echo "Missing regular data-gate artifact: ${data_gate_bundle}/${artifact}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output_bundle_path="${output_dir}/${OUTPUT_STEM}"
start_step=0
if [[ -e "${output_bundle_path}" || -L "${output_bundle_path}" ]]; then
  if [[ "${RESUME_EXISTING}" != 1 ]]; then
    echo "Output already exists; set RESUME_EXISTING=1 to continue it" >&2
    exit 2
  fi
  if [[ -L "${output_bundle_path}" || ! -d "${output_bundle_path}" ]]; then
    echo "Resume output must be a non-symlink directory" >&2
    exit 2
  fi
  if [[ -L "${output_bundle_path}/result.json" ]]; then
    echo "Long-run result must not be a symlink" >&2
    exit 2
  fi
  if [[ -f "${output_bundle_path}/result.json" ]]; then
    echo "Long run is already complete: ${output_bundle_path}/result.json"
    exit 0
  fi
  latest_checkpoint=""
  if [[ -d "${output_bundle_path}/checkpoints" && \
        ! -L "${output_bundle_path}/checkpoints" ]]; then
    latest_checkpoint="$(
      find "${output_bundle_path}/checkpoints" -maxdepth 1 -type f \
        -name 'step-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt' \
        -print | LC_ALL=C sort | tail -n 1
    )"
  fi
  if [[ -n "${latest_checkpoint}" ]]; then
    checkpoint_name="${latest_checkpoint##*/}"
    start_step=$((10#${checkpoint_name:5:9}))
  fi
fi
if (( start_step > TOTAL_STEPS )); then
  echo "Latest checkpoint is beyond TOTAL_STEPS" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
if ! git -C "${launch_dir}" ls-files --error-unmatch -- \
  "${CONFIG_RELATIVE_PATH}" >/dev/null 2>&1; then
  echo "Validated config is not committed at LAUNCH_SHA" >&2
  exit 2
fi
(cd "${launch_dir}" && uv sync --frozen --extra tracking --no-build-package wandb) >&2
if ! "${launch_dir}/.venv/bin/python" -c \
  'import wandb; assert callable(wandb.init)' >/dev/null 2>&1; then
  echo "Long run requires functional pinned W&B support" >&2
  exit 2
fi
if ! WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  "${launch_dir}/.venv/bin/wandb" login --verify </dev/null >&2; then
  echo "Login node could not verify W&B credentials and server connectivity" >&2
  exit 2
fi

export LAUNCH_SHA
export DATA_ROOT="${data_root}"
export DATA_GATE_BUNDLE="${data_gate_bundle}"
export EXPECTED_MANIFEST_SHA256
export EXPECTED_SPLIT_SHA256
export EXPECTED_CASE_GRID_MANIFEST_SHA256
export CONFIG_RELATIVE_PATH
export OUTPUT_BUNDLE_PATH="${output_bundle_path}"
export TOTAL_STEPS
export MAX_STEPS_PER_INVOCATION
export BAGS_PER_SUBJECT
export EXPECTED_TRAIN_CASES
export EXPECTED_TRAIN_SUBJECTS
export WANDB_MODE WANDB_PROJECT
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
else
  unset WANDB_ENTITY
fi

remaining_steps=$((TOTAL_STEPS - start_step))
if (( remaining_steps == 0 )); then
  segment_count=1
else
  segment_count=$(((remaining_steps + MAX_STEPS_PER_INVOCATION - 1) / MAX_STEPS_PER_INVOCATION))
fi
dependency=""
if [[ -n "${AFTER_JOB_ID:-}" ]]; then
  if [[ ! "${AFTER_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "AFTER_JOB_ID must be a numeric Slurm job ID" >&2
    exit 2
  fi
  dependency="afterok:${AFTER_JOB_ID}"
fi

job_ids=()
for ((segment = 0; segment < segment_count; segment++)); do
  sbatch_args=(
    --parsable
    --chdir="${launch_dir}"
    --output="${output_dir}/${OUTPUT_STEM}-%j.out"
    --export=ALL
  )
  if [[ -n "${dependency}" ]]; then
    sbatch_args+=(--dependency="${dependency}")
  fi
  submitted="$(sbatch "${sbatch_args[@]}" "${launch_dir}/slurm/long_run.sbatch")"
  job_id="${submitted%%;*}"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Unexpected sbatch response: ${submitted}" >&2
    exit 2
  fi
  job_ids+=("${job_id}")
  dependency="afterok:${job_id}"
done

printf '%s\n' "${job_ids[@]}"
