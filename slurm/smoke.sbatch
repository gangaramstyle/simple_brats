#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-smoke
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${SYNTHETIC_DATASET_ID:?SYNTHETIC_DATASET_ID was not exported}"

actual_sha="$(git rev-parse HEAD)"
if [[ "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout mismatch: expected ${LAUNCH_SHA}, got ${actual_sha}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Runtime launch tree contains modified tracked files" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing uv-managed .venv/bin/python; prepare dependencies before sbatch" >&2
  exit 2
fi

mkdir -p runs/slurm
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

srun .venv/bin/python -m simple_brats.cli smoke \
  --device cuda \
  --config configs/v0_cross_matching.toml \
  --synthetic-dataset-id "${SYNTHETIC_DATASET_ID}" \
  --expected-git-sha "${LAUNCH_SHA}"
