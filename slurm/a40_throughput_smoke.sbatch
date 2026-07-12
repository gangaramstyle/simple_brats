#!/usr/bin/env bash
#SBATCH --job-name=brats-a40-throughput
#SBATCH --partition=ai
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE

set -euo pipefail
umask 077

: "${LAUNCH_SHA:?LAUNCH_SHA was not exported}"
: "${DATA_ROOT:?DATA_ROOT was not exported}"
: "${DATA_GATE_BUNDLE:?DATA_GATE_BUNDLE was not exported}"
: "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 was not exported}"
: "${EXPECTED_SPLIT_SHA256:?EXPECTED_SPLIT_SHA256 was not exported}"
: "${EXPECTED_CASE_GRID_MANIFEST_SHA256:?EXPECTED_CASE_GRID_MANIFEST_SHA256 was not exported}"
: "${CONFIG_RELATIVE_PATH:?CONFIG_RELATIVE_PATH was not exported}"
: "${OUTPUT_BUNDLE_PATH:?OUTPUT_BUNDLE_PATH was not exported}"

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
if [[ "${CONFIG_RELATIVE_PATH}" != "configs/v0_cross_matching_small.toml" ]]; then
  echo "Throughput smoke is locked to the registered small 4 mm config" >&2
  exit 2
fi
for artifact in filtered.manifest.json subject-split.json case-grid-manifest.json; do
  if [[ -L "${DATA_GATE_BUNDLE}/${artifact}" || ! -f "${DATA_GATE_BUNDLE}/${artifact}" ]]; then
    echo "Missing regular data-gate artifact: ${DATA_GATE_BUNDLE}/${artifact}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_BUNDLE_PATH}" || -L "${OUTPUT_BUNDLE_PATH}" ]]; then
  echo "Refusing to overwrite throughput-smoke output: ${OUTPUT_BUNDLE_PATH}" >&2
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE

srun .venv/bin/python -m simple_brats.a40_throughput_smoke \
  --data-root "${DATA_ROOT}" \
  --manifest "${DATA_GATE_BUNDLE}/filtered.manifest.json" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --split "${DATA_GATE_BUNDLE}/subject-split.json" \
  --expected-split-sha256 "${EXPECTED_SPLIT_SHA256}" \
  --case-grid-manifest "${DATA_GATE_BUNDLE}/case-grid-manifest.json" \
  --expected-case-grid-manifest-sha256 "${EXPECTED_CASE_GRID_MANIFEST_SHA256}" \
  --config "${CONFIG_RELATIVE_PATH}" \
  --output-dir "${OUTPUT_BUNDLE_PATH}" \
  --expected-git-sha "${LAUNCH_SHA}" \
  --repo-root . \
  --device cuda
