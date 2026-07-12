#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact evaluator commit}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed data gate}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_SPLIT_SHA256:?Set EXPECTED_SPLIT_SHA256}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set expected case-grid SHA}"
: "${EVALUATION_PATCH_MANIFEST_PATH:?Set the materialized patch manifest path}"
: "${EXPECTED_EVALUATION_PATCH_MANIFEST_SHA256:?Set its canonical SHA}"
: "${CHECKPOINT_PATH:?Set the runner-v3 checkpoint path}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/heldout-evaluation/checkpoints}"
: "${ALLOW_PARTIAL_SSL_TRAIN:=0}"
: "${WANDB_MODE:=online}"
: "${WANDB_PROJECT:=simple-brats}"
: "${WANDB_ENTITY:=}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase Git SHA" >&2
  exit 2
fi
case "${CONFIG_RELATIVE_PATH}" in
  configs/v0_cross_matching_small.toml) evaluation_arm="32mm-4mm-tensor8" ;;
  configs/v0_cross_matching_small_8mm.toml) evaluation_arm="64mm-8mm-tensor8" ;;
  *)
    echo "Held-out evaluation requires one of the two registered scale configs" >&2
    exit 2
    ;;
esac
if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "Registered held-out evaluation requires WANDB_MODE=online" >&2
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
for value in \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${EXPECTED_SPLIT_SHA256}" \
  "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  "${EXPECTED_EVALUATION_PATCH_MANIFEST_SHA256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every expected artifact SHA must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
for directory in "${DATA_ROOT}" "${DATA_GATE_BUNDLE}"; do
  if [[ -L "${directory}" || ! -d "${directory}" ]]; then
    echo "Input directory must exist and not be a symlink: ${directory}" >&2
    exit 2
  fi
done
if [[ -L "${EVALUATION_PATCH_MANIFEST_PATH}" || \
      ! -f "${EVALUATION_PATCH_MANIFEST_PATH}" ]]; then
  echo "Evaluation patch manifest must be a regular file" >&2
  exit 2
fi
data_root="$(cd "${DATA_ROOT}" && pwd -P)"
data_gate_bundle="$(cd "${DATA_GATE_BUNDLE}" && pwd -P)"
evaluation_patch_manifest_path="$(
  cd "$(dirname "${EVALUATION_PATCH_MANIFEST_PATH}")" && pwd -P
)/$(basename "${EVALUATION_PATCH_MANIFEST_PATH}")"
if [[ ! "${CHECKPOINT_PATH}" = /* ]]; then
  echo "CHECKPOINT_PATH must be absolute" >&2
  exit 2
fi
if [[ -e "${CHECKPOINT_PATH}" ]]; then
  if [[ -L "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Existing checkpoint must be a regular non-symlink file" >&2
    exit 2
  fi
elif [[ -z "${AFTER_JOB_ID:-}" ]]; then
  echo "A future checkpoint requires AFTER_JOB_ID" >&2
  exit 2
fi
checkpoint_name="$(basename "${CHECKPOINT_PATH}" .pt)"
: "${OUTPUT_STEM:=${checkpoint_name}-${evaluation_arm}-heldout-evaluation-v1}"
mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output="${output_dir}/${OUTPUT_STEM}.json"
for candidate in "${output}" "${output}.wandb.json"; do
  if [[ -e "${candidate}" || -L "${candidate}" ]]; then
    echo "Refusing to overwrite ${candidate}" >&2
    exit 2
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
(cd "${launch_dir}" && uv sync --frozen --extra tracking --no-build-package wandb) >&2
if ! WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  "${launch_dir}/.venv/bin/wandb" login --verify </dev/null >&2; then
  echo "Login node could not verify W&B credentials and server connectivity" >&2
  exit 2
fi
export LAUNCH_SHA
export DATA_ROOT="${data_root}"
export DATA_GATE_BUNDLE="${data_gate_bundle}"
export EXPECTED_MANIFEST_SHA256 EXPECTED_SPLIT_SHA256
export EXPECTED_CASE_GRID_MANIFEST_SHA256 CONFIG_RELATIVE_PATH
export EVALUATION_PATCH_MANIFEST_PATH="${evaluation_patch_manifest_path}"
export EXPECTED_EVALUATION_PATCH_MANIFEST_SHA256
export CHECKPOINT_PATH ALLOW_PARTIAL_SSL_TRAIN
export EVALUATION_REPORT_OUTPUT="${output}"
export WANDB_MODE WANDB_PROJECT
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
else
  unset WANDB_ENTITY
fi

sbatch_args=(
  --parsable
  --chdir="${launch_dir}"
  --output="${output_dir}/${OUTPUT_STEM}-%j.out"
  --export=ALL
)
if [[ -n "${AFTER_JOB_ID:-}" ]]; then
  if [[ ! "${AFTER_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "AFTER_JOB_ID must be numeric" >&2
    exit 2
  fi
  sbatch_args+=(--dependency="afterok:${AFTER_JOB_ID}")
fi
sbatch "${sbatch_args[@]}" "${launch_dir}/slurm/evaluate_checkpoint.sbatch"
