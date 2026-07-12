#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed data gate}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_SPLIT_SHA256:?Set EXPECTED_SPLIT_SHA256}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set expected case-grid SHA}"
: "${SEGMENTATION_LABEL_AUDIT_PATH:?Set the real label-audit JSON path}"
: "${EXPECTED_SEGMENTATION_LABEL_AUDIT_SHA256:?Set the label-audit file SHA}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/heldout-evaluation}"
: "${OUTPUT_STEM:=brats-met-4mm-robust-binary-patches-v0}"
: "${PROBE_TRAIN_SUBJECT_COUNT:=128}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase Git SHA" >&2
  exit 2
fi
if [[ "${CONFIG_RELATIVE_PATH}" != "configs/v0_cross_matching_small.toml" ]]; then
  echo "Held-out v0 evaluation is locked to the small 4 mm config" >&2
  exit 2
fi
for directory in "${DATA_ROOT}" "${DATA_GATE_BUNDLE}"; do
  if [[ -L "${directory}" || ! -d "${directory}" ]]; then
    echo "Input directory must exist and not be a symlink: ${directory}" >&2
    exit 2
  fi
done
if [[ -L "${SEGMENTATION_LABEL_AUDIT_PATH}" || \
      ! -f "${SEGMENTATION_LABEL_AUDIT_PATH}" ]]; then
  echo "SEGMENTATION_LABEL_AUDIT_PATH must be a regular non-symlink file" >&2
  exit 2
fi

data_root="$(cd "${DATA_ROOT}" && pwd -P)"
data_gate_bundle="$(cd "${DATA_GATE_BUNDLE}" && pwd -P)"
label_audit_path="$(cd "$(dirname "${SEGMENTATION_LABEL_AUDIT_PATH}")" && pwd -P)/$(basename "${SEGMENTATION_LABEL_AUDIT_PATH}")"
mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output="${output_dir}/${OUTPUT_STEM}.json"
if [[ -e "${output}" || -L "${output}" ]]; then
  echo "Refusing to overwrite ${output}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
export LAUNCH_SHA
export DATA_ROOT="${data_root}"
export DATA_GATE_BUNDLE="${data_gate_bundle}"
export EXPECTED_MANIFEST_SHA256
export EXPECTED_SPLIT_SHA256
export EXPECTED_CASE_GRID_MANIFEST_SHA256
export SEGMENTATION_LABEL_AUDIT_PATH="${label_audit_path}"
export EXPECTED_SEGMENTATION_LABEL_AUDIT_SHA256
export CONFIG_RELATIVE_PATH
export PROBE_TRAIN_SUBJECT_COUNT
export EVALUATION_PATCH_MANIFEST_OUTPUT="${output}"

sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${output_dir}/${OUTPUT_STEM}-%j.out" \
  --export=ALL \
  "${launch_dir}/slurm/materialize_heldout_evaluation.sbatch"
