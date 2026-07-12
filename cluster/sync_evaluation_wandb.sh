#!/usr/bin/env bash
set -euo pipefail

: "${EVALUATION_OUTPUT_DIR:?Set the directory containing offline evaluation runs}"
: "${WANDB_PROJECT:=simple-brats}"

if [[ -L "${EVALUATION_OUTPUT_DIR}" || ! -d "${EVALUATION_OUTPUT_DIR}" ]]; then
  echo "EVALUATION_OUTPUT_DIR must be an existing non-symlink directory" >&2
  exit 2
fi
directory="$(cd "${EVALUATION_OUTPUT_DIR}" && pwd -P)"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ ! -x "${repo_root}/.venv/bin/wandb" ]]; then
  echo "Pinned W&B CLI is unavailable; run uv sync --frozen --extra tracking" >&2
  exit 2
fi

found=0
while IFS= read -r -d '' offline_run; do
  found=1
  "${repo_root}/.venv/bin/wandb" sync --project "${WANDB_PROJECT}" "${offline_run}"
done < <(find "${directory}" -type d -name 'offline-run-*' -print0)
if (( found == 0 )); then
  echo "No offline W&B evaluation runs found under ${directory}" >&2
  exit 2
fi
