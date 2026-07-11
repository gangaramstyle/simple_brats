#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${RAW_MANIFEST_PATH:?Set RAW_MANIFEST_PATH to the already-built raw manifest}"
: "${EXPECTED_RAW_MANIFEST_SHA256:?Set EXPECTED_RAW_MANIFEST_SHA256 to its canonical SHA-256}"
: "${FILTER_SPEC_RELATIVE_PATH:=protocols/brats_met_2025_cross_subject_duplicate_quarantine.json}"
: "${EXPECTED_FILTER_SPEC_SHA256:?Set EXPECTED_FILTER_SPEC_SHA256 to the canonical filter-spec SHA-256}"
: "${SPLIT_SEED:=0}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/data-gates}"
: "${OUTPUT_STEM:=brats-met-2025-training-clean-v0}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full 40-character lowercase commit ID" >&2
  exit 2
fi
if [[ ! "${EXPECTED_RAW_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_RAW_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ ! "${EXPECTED_FILTER_SPEC_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_FILTER_SPEC_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ ! "${SPLIT_SEED}" =~ ^[0-9]+$ ]]; then
  echo "SPLIT_SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${OUTPUT_STEM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_STEM may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! "${FILTER_SPEC_RELATIVE_PATH}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "FILTER_SPEC_RELATIVE_PATH contains unsupported characters" >&2
  exit 2
fi
if [[ "${FILTER_SPEC_RELATIVE_PATH}" == /* ]]; then
  echo "FILTER_SPEC_RELATIVE_PATH must be relative to the pinned checkout" >&2
  exit 2
fi
IFS='/' read -r -a filter_components <<<"${FILTER_SPEC_RELATIVE_PATH}"
for component in "${filter_components[@]}"; do
  if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
    echo "FILTER_SPEC_RELATIVE_PATH must not contain empty, dot, or dot-dot components" >&2
    exit 2
  fi
done

if [[ -L "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT must not be a symlink" >&2
  exit 2
fi
data_root="$(cd "${DATA_ROOT}" && pwd -P)"

if [[ -L "${RAW_MANIFEST_PATH}" || ! -f "${RAW_MANIFEST_PATH}" ]]; then
  echo "RAW_MANIFEST_PATH must name an existing regular file, not a symlink" >&2
  exit 2
fi
raw_manifest_dir="$(cd "$(dirname "${RAW_MANIFEST_PATH}")" && pwd -P)"
raw_manifest_path="${raw_manifest_dir}/$(basename "${RAW_MANIFEST_PATH}")"

mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output_bundle_path="${output_dir}/${OUTPUT_STEM}"
log_path="${output_dir}/${OUTPUT_STEM}-%j.out"
if [[ -e "${output_bundle_path}" || -L "${output_bundle_path}" ]]; then
  echo "Refusing to overwrite an existing data-gate bundle: ${output_bundle_path}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"

if ! tracked_filter_path="$(
  git -C "${launch_dir}" ls-files --error-unmatch -- "${FILTER_SPEC_RELATIVE_PATH}"
)"; then
  echo "Filter spec is not committed at LAUNCH_SHA: ${FILTER_SPEC_RELATIVE_PATH}" >&2
  exit 2
fi
if [[ "${tracked_filter_path}" != "${FILTER_SPEC_RELATIVE_PATH}" ]]; then
  echo "Filter path did not resolve to one exact tracked file" >&2
  exit 2
fi
filter_spec_path="${launch_dir}/${FILTER_SPEC_RELATIVE_PATH}"
if [[ -L "${filter_spec_path}" || ! -f "${filter_spec_path}" ]]; then
  echo "Committed filter spec must be a regular file, not a symlink" >&2
  exit 2
fi
filter_spec_dir="$(cd "$(dirname "${filter_spec_path}")" && pwd -P)"
if [[ "${filter_spec_dir}" != "${launch_dir}" && "${filter_spec_dir}" != "${launch_dir}/"* ]]; then
  echo "Filter spec escapes the pinned checkout" >&2
  exit 2
fi

export LAUNCH_SHA
export DATA_ROOT="${data_root}"
export RAW_MANIFEST_PATH="${raw_manifest_path}"
export EXPECTED_RAW_MANIFEST_SHA256
export FILTER_SPEC_RELATIVE_PATH
export EXPECTED_FILTER_SPEC_SHA256
export SPLIT_SEED
export OUTPUT_BUNDLE_PATH="${output_bundle_path}"

sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${log_path}" \
  --export=ALL \
  "${launch_dir}/slurm/filtered_data_gate.sbatch"
