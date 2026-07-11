#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed filtered data-gate bundle}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256 to filtered.manifest.json SHA}"
: "${EXPECTED_SPLIT_SHA256:?Set EXPECTED_SPLIT_SHA256 to subject-split.json SHA}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set EXPECTED_CASE_GRID_MANIFEST_SHA256 to case-grid-manifest.json SHA}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/real-io-pilots}"
: "${OUTPUT_STEM:=brats-met-2025-small-real-io-v0}"
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
if [[ ! "${OUTPUT_STEM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_STEM may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! "${CONFIG_RELATIVE_PATH}" =~ ^[A-Za-z0-9._/-]+$ || "${CONFIG_RELATIVE_PATH}" == /* ]]; then
  echo "CONFIG_RELATIVE_PATH must be a safe relative path" >&2
  exit 2
fi
IFS='/' read -r -a config_components <<<"${CONFIG_RELATIVE_PATH}"
for component in "${config_components[@]}"; do
  if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
    echo "CONFIG_RELATIVE_PATH must not contain empty, dot, or dot-dot components" >&2
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
for artifact in filtered.manifest.json subject-split.json case-grid-manifest.json; do
  if [[ -L "${data_gate_bundle}/${artifact}" || ! -f "${data_gate_bundle}/${artifact}" ]]; then
    echo "Missing regular data-gate artifact: ${data_gate_bundle}/${artifact}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output_bundle_path="${output_dir}/${OUTPUT_STEM}"
log_path="${output_dir}/${OUTPUT_STEM}-%j.out"
if [[ -e "${output_bundle_path}" || -L "${output_bundle_path}" ]]; then
  echo "Refusing to overwrite an existing pilot bundle: ${output_bundle_path}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
if ! tracked_config_path="$(
  git -C "${launch_dir}" ls-files --error-unmatch -- "${CONFIG_RELATIVE_PATH}"
)"; then
  echo "Pilot config is not committed at LAUNCH_SHA: ${CONFIG_RELATIVE_PATH}" >&2
  exit 2
fi
if [[ "${tracked_config_path}" != "${CONFIG_RELATIVE_PATH}" ]]; then
  echo "Config path did not resolve to one exact tracked file" >&2
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
export PILOT_EPOCH
export PILOT_BAG_INDEX
export PILOT_CASE_ID

sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${log_path}" \
  --export=ALL \
  "${launch_dir}/slurm/real_io_pilot.sbatch"
