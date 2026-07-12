#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_BUNDLE:?Set OUTPUT_BUNDLE to a completed or active long-run bundle}"
: "${WANDB_PROJECT:=simple-brats}"

if [[ -L "${OUTPUT_BUNDLE}" || ! -d "${OUTPUT_BUNDLE}" ]]; then
  echo "OUTPUT_BUNDLE must be an existing non-symlink directory" >&2
  exit 2
fi
bundle="$(cd "${OUTPUT_BUNDLE}" && pwd -P)"
if [[ ! -d "${bundle}/wandb" ]]; then
  echo "No offline W&B directory exists in ${bundle}" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ ! -x "${repo_root}/.venv/bin/wandb" ]]; then
  echo "Pinned W&B CLI is unavailable; run uv sync --frozen --extra tracking" >&2
  exit 2
fi

found=0
while IFS= read -r -d '' offline_run; do
  found=1
  "${repo_root}/.venv/bin/wandb" sync --project "${WANDB_PROJECT}" "${offline_run}"
done < <(find "${bundle}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print0)
if (( found == 0 )); then
  echo "No offline W&B runs found in ${bundle}/wandb" >&2
  exit 2
fi
