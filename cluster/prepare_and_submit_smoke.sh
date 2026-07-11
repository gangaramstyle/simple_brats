#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${SYNTHETIC_DATASET_ID:=synthetic-smoke-v0}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launch_dir="$(bash "${script_dir}/prepare_launch.sh")"

sbatch --parsable \
  --chdir="${launch_dir}" \
  --export="ALL,LAUNCH_SHA=${LAUNCH_SHA},SYNTHETIC_DATASET_ID=${SYNTHETIC_DATASET_ID}" \
  "${launch_dir}/slurm/smoke.sbatch"
