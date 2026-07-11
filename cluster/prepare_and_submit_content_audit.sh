#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed filtered data-gate bundle}"
: "${EXPECTED_FILTERED_MANIFEST_SHA256:?Set the filtered manifest canonical SHA-256}"
: "${EXPECTED_SUBJECT_SPLIT_SHA256:?Set the subject split canonical SHA-256}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set the case-grid manifest canonical SHA-256}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/content-audits}"

if [[ ! "${CONFIG_RELATIVE_PATH}" =~ ^configs/[A-Za-z0-9._-]+\.toml$ ]]; then
  echo "CONFIG_RELATIVE_PATH must name one TOML file directly under configs/" >&2
  exit 2
fi
config_stem="$(basename "${CONFIG_RELATIVE_PATH}" .toml)"
: "${OUTPUT_STEM:=brats-met-2025-training-clean-v0-${config_stem}-content-audit}"

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
if [[ ! "${OUTPUT_STEM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_STEM may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

if [[ -L "${DATA_ROOT}" || ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT must be an existing directory, not a symlink" >&2
  exit 2
fi
data_root="$(cd "${DATA_ROOT}" && pwd -P)"

if [[ -L "${DATA_GATE_BUNDLE}" || ! -d "${DATA_GATE_BUNDLE}" ]]; then
  echo "DATA_GATE_BUNDLE must be an existing directory, not a symlink" >&2
  exit 2
fi
data_gate_bundle="$(cd "${DATA_GATE_BUNDLE}" && pwd -P)"
manifest_path="${data_gate_bundle}/filtered.manifest.json"
split_path="${data_gate_bundle}/subject-split.json"
case_grid_manifest_path="${data_gate_bundle}/case-grid-manifest.json"
for artifact in "${manifest_path}" "${split_path}" "${case_grid_manifest_path}"; do
  if [[ -L "${artifact}" || ! -f "${artifact}" ]]; then
    echo "Data-gate input must be a regular file, not a symlink: ${artifact}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output_path="${CONTENT_AUDIT_OUTPUT_PATH:-${output_dir}/${OUTPUT_STEM}.json}"
state_dir="${CONTENT_AUDIT_STATE_DIR:-${output_dir}/${OUTPUT_STEM}.state}"
log_path="${output_dir}/${OUTPUT_STEM}-%j.out"

if [[ "${output_path}" != /* || "${state_dir}" != /* ]]; then
  echo "CONTENT_AUDIT_OUTPUT_PATH and CONTENT_AUDIT_STATE_DIR must be absolute" >&2
  exit 2
fi
if [[ -e "${output_path}" || -L "${output_path}" ]]; then
  echo "Refusing to overwrite an existing content-audit output: ${output_path}" >&2
  exit 2
fi
if [[ -L "${state_dir}" ]]; then
  echo "CONTENT_AUDIT_STATE_DIR must not be a symlink" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
config_path="${launch_dir}/${CONFIG_RELATIVE_PATH}"
if [[ -L "${config_path}" || ! -f "${config_path}" ]] || \
  ! git -C "${launch_dir}" ls-files --error-unmatch -- \
    "${CONFIG_RELATIVE_PATH}" >/dev/null 2>&1; then
  echo "Experiment config must be a committed regular file at LAUNCH_SHA" >&2
  exit 2
fi

export LAUNCH_SHA
export DATA_ROOT="${data_root}"
export FILTERED_MANIFEST_PATH="${manifest_path}"
export SUBJECT_SPLIT_PATH="${split_path}"
export CASE_GRID_MANIFEST_PATH="${case_grid_manifest_path}"
export EXPECTED_FILTERED_MANIFEST_SHA256
export EXPECTED_SUBJECT_SPLIT_SHA256
export EXPECTED_CASE_GRID_MANIFEST_SHA256
export CONFIG_RELATIVE_PATH
export CONTENT_AUDIT_OUTPUT_PATH="${output_path}"
export CONTENT_AUDIT_STATE_DIR="${state_dir}"

sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${log_path}" \
  --export=ALL \
  "${launch_dir}/slurm/content_audit.sbatch"
