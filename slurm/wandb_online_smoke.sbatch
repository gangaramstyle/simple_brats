#!/usr/bin/env bash
#SBATCH --job-name=simple-brats-wandb-online
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --propagate=NONE

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${WANDB_SMOKE_OUTPUT:?WANDB_SMOKE_OUTPUT was not exported}"
: "${WANDB_MODE:?WANDB_MODE was not exported}"
: "${WANDB_PROJECT:?WANDB_PROJECT was not exported}"
: "${WANDB_ENTITY:=}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]] || \
  [[ "$(git rev-parse HEAD)" != "${LAUNCH_SHA}" ]]; then
  echo "Runtime checkout does not match literal LAUNCH_SHA" >&2
  exit 2
fi
if git symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "Runtime checkout must be detached" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Runtime launch tree contains modified tracked files" >&2
  exit 2
fi
if [[ "${WANDB_MODE}" != "online" ]]; then
  echo "Connectivity smoke requires WANDB_MODE=online" >&2
  exit 2
fi
if [[ -e "${WANDB_SMOKE_OUTPUT}" || -L "${WANDB_SMOKE_OUTPUT}" ]]; then
  echo "Refusing to overwrite W&B connectivity output" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Pinned Python environment is missing" >&2
  exit 2
fi
lock_sha="$(sha256sum uv.lock | awk '{print $1}')"
if [[ ! -f .venv/.simple-brats-lock-sha256 ]] || \
  [[ "$(<.venv/.simple-brats-lock-sha256)" != "${lock_sha}" ]]; then
  echo "Runtime uv environment does not match the pinned lock" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export WANDB_MODE WANDB_PROJECT
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
else
  unset WANDB_ENTITY
fi
srun .venv/bin/python -m simple_brats.wandb_smoke \
  --output "${WANDB_SMOKE_OUTPUT}" \
  --expected-git-sha "${LAUNCH_SHA}" \
  --repo-root .
