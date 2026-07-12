#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_GATE_BUNDLE:?Set DATA_GATE_BUNDLE to the completed data-gate bundle}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_SPLIT_SHA256:?Set EXPECTED_SPLIT_SHA256}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?Set EXPECTED_CASE_GRID_MANIFEST_SHA256}"
: "${CONFIG_RELATIVE_PATH:=configs/v0_cross_matching_small.toml}"
: "${EXPECTED_CONFIG_SHA256:=a261de64b08e19390a952a1d151066a10540acea55859d661cd0293848fd6bd3}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/cohort-preflights}"
: "${OUTPUT_STEM:=}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase commit ID" >&2
  exit 2
fi
for value in \
  "${EXPECTED_MANIFEST_SHA256}" \
  "${EXPECTED_SPLIT_SHA256}" \
  "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  "${EXPECTED_CONFIG_SHA256}"; do
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every expected artifact SHA must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
case "${CONFIG_RELATIVE_PATH}:${EXPECTED_CONFIG_SHA256}" in
  configs/v0_cross_matching_small.toml:a261de64b08e19390a952a1d151066a10540acea55859d661cd0293848fd6bd3) arm_slug="32mm-4mm-tensor8" ;;
  configs/v0_cross_matching_small_8mm.toml:7ce7024c902e33878f019c1eac963d9c1e4da085261c9402b32123656d92a3bf) arm_slug="64mm-8mm-tensor8" ;;
  *)
    echo "Cohort preflight requires one exact registered scale-matched config and digest" >&2
    exit 2
    ;;
esac
if [[ -z "${OUTPUT_STEM}" ]]; then
  OUTPUT_STEM="brats-met-train-1044-cold-path-${arm_slug}-v1"
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
if [[ -L "${output_bundle_path}" ]] || \
  [[ -e "${output_bundle_path}" && ! -d "${output_bundle_path}" ]]; then
  echo "Existing output bundle must be a non-symlink directory" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
if ! git -C "${launch_dir}" ls-files --error-unmatch -- \
  "${CONFIG_RELATIVE_PATH}" >/dev/null 2>&1; then
  echo "Registered config is not committed at LAUNCH_SHA" >&2
  exit 2
fi

export LAUNCH_SHA
export DATA_ROOT="${data_root}"
export DATA_GATE_BUNDLE="${data_gate_bundle}"
export EXPECTED_MANIFEST_SHA256
export EXPECTED_SPLIT_SHA256
export EXPECTED_CASE_GRID_MANIFEST_SHA256
export EXPECTED_CONFIG_SHA256
export CONFIG_RELATIVE_PATH
export OUTPUT_BUNDLE_PATH="${output_bundle_path}"

sbatch_args=(
  --parsable
  --chdir="${launch_dir}"
  --output="${output_dir}/${OUTPUT_STEM}-%j.out"
  --export=ALL
)
if [[ -n "${AFTER_JOB_ID:-}" ]]; then
  if [[ ! "${AFTER_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "AFTER_JOB_ID must be a numeric Slurm job ID" >&2
    exit 2
  fi
  sbatch_args+=(--dependency="afterok:${AFTER_JOB_ID}")
fi
sbatch "${sbatch_args[@]}" "${launch_dir}/slurm/cohort_preflight.sbatch"
