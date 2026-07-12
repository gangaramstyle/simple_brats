#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed data-gate bundle}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_SPLIT_SHA256:?Set EXPECTED_SPLIT_SHA256}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set EXPECTED_CASE_GRID_MANIFEST_SHA256}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/throughput-smoke}"
: "${OUTPUT_STEM:=small-4mm-a40-optimized-throughput-v0}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase commit ID" >&2
  exit 2
fi
for value in \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${EXPECTED_SPLIT_SHA256}" \
  "${EXPECTED_CASE_GRID_MANIFEST_SHA256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every expected data artifact SHA must be lowercase SHA-256" >&2
    exit 2
  fi
done
case "${CONFIG_RELATIVE_PATH}" in
  configs/v0_cross_matching_small.toml|configs/v0_cross_matching_small_8mm.toml) ;;
  *)
    echo "Throughput smoke requires one of the two registered scale-matched configs" >&2
    exit 2
    ;;
esac
if [[ ! "${OUTPUT_STEM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_STEM contains unsafe characters" >&2
  exit 2
fi
for directory in "${DATA_ROOT}" "${DATA_GATE_BUNDLE}"; do
  if [[ -L "${directory}" || ! -d "${directory}" ]]; then
    echo "Input must be an existing non-symlink directory: ${directory}" >&2
    exit 2
  fi
done

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
if [[ -e "${output_bundle_path}" || -L "${output_bundle_path}" ]]; then
  echo "Refusing to overwrite throughput-smoke output: ${output_bundle_path}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
if ! git -C "${launch_dir}" ls-files --error-unmatch -- \
  "${CONFIG_RELATIVE_PATH}" \
  src/simple_brats/a40_throughput_smoke.py \
  slurm/a40_throughput_smoke.sbatch >/dev/null 2>&1; then
  echo "Throughput-smoke code/config is not committed at LAUNCH_SHA" >&2
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

sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${output_dir}/${OUTPUT_STEM}-%j.out" \
  --export=ALL \
  "${launch_dir}/slurm/a40_throughput_smoke.sbatch"
