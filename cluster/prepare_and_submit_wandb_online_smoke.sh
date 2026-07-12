#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to verify}"
: "${OUTPUT_DIR:=${HOME}/simple_brats_artifacts/wandb-smoke}"
: "${OUTPUT_STEM:=wandb-online-${LAUNCH_SHA:0:12}}"
: "${WANDB_MODE:=online}"
: "${WANDB_PROJECT:=simple-brats}"
: "${WANDB_ENTITY:=}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full lowercase commit ID" >&2
  exit 2
fi
if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "Connectivity smoke requires WANDB_MODE=online" >&2
  exit 2
fi
for value in "${OUTPUT_STEM}" "${WANDB_PROJECT}"; do
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Output stem and W&B project must use safe non-empty names" >&2
    exit 2
  fi
done
if [[ -n "${WANDB_ENTITY}" && ! "${WANDB_ENTITY}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "WANDB_ENTITY contains unsupported characters" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
output_dir="$(cd "${OUTPUT_DIR}" && pwd -P)"
output="${output_dir}/${OUTPUT_STEM}.json"
if [[ -e "${output}" || -L "${output}" ]]; then
  echo "Refusing to overwrite ${output}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"
(cd "${launch_dir}" && uv sync --frozen --extra tracking --no-build-package wandb) >&2
if ! WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  "${launch_dir}/.venv/bin/wandb" login --verify </dev/null >&2; then
  echo "Login node could not verify W&B credentials and server connectivity" >&2
  exit 2
fi

export LAUNCH_SHA WANDB_MODE WANDB_PROJECT
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
else
  unset WANDB_ENTITY
fi
export WANDB_SMOKE_OUTPUT="${output}"
sbatch --parsable \
  --chdir="${launch_dir}" \
  --output="${output_dir}/${OUTPUT_STEM}-%j.out" \
  --export=ALL \
  "${launch_dir}/slurm/wandb_online_smoke.sbatch"
