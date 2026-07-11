#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${DATA_ROOT:?Set DATA_ROOT to the MET release root}"
: "${DATA_SOURCE:=BraTS-MET}"
: "${DATA_RELEASE:?Set DATA_RELEASE to the exact release identifier}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/manifests}"
: "${OUTPUT_STEM:=brats-met-training}"

if [[ ! "${OUTPUT_STEM}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "OUTPUT_STEM may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

data_root="$(cd "${DATA_ROOT}" && pwd -P)"
mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
manifest_path="${output_dir}/${OUTPUT_STEM}.manifest.json"
split_path="${output_dir}/${OUTPUT_STEM}.split.json"
log_path="${output_dir}/${OUTPUT_STEM}-%j.out"

if [[ -e "${manifest_path}" || -e "${split_path}" ]]; then
  echo "Refusing to overwrite an existing manifest or split" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"

sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${log_path}" \
  --export="ALL,LAUNCH_SHA=${LAUNCH_SHA},DATA_ROOT=${data_root},DATA_SOURCE=${DATA_SOURCE},DATA_RELEASE=${DATA_RELEASE},MANIFEST_PATH=${manifest_path},SPLIT_PATH=${split_path}" \
  "${launch_dir}/slurm/build_manifest.sbatch"
