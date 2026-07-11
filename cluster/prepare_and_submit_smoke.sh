#!/usr/bin/env bash
set -euo pipefail

: "${LAUNCH_SHA:?Set LAUNCH_SHA to the exact commit to launch}"
: "${SYNTHETIC_DATASET_ID:=synthetic-smoke-v0}"

if [[ ! "${LAUNCH_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCH_SHA must be a full 40-character lowercase commit ID" >&2
  exit 2
fi

seed_root="$(git rev-parse --show-toplevel)"
git -C "${seed_root}" fetch origin "${LAUNCH_SHA}"

launch_base="${SIMPLE_BRATS_LAUNCH_ROOT:-${HOME}/.cache/simple_brats/launches}"
launch_dir="${launch_base}/${LAUNCH_SHA}"
lock_dir="${launch_base}/.${LAUNCH_SHA}.prepare-lock"
mkdir -p "${launch_base}"
if ! mkdir "${lock_dir}"; then
  echo "Another process is preparing ${LAUNCH_SHA}; retry after it finishes" >&2
  exit 3
fi
staging=""
cleanup() {
  [[ -z "${staging}" || ! -d "${staging}" ]] || rm -rf "${staging}"
  rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -e "${launch_dir}" && ! -d "${launch_dir}/.git" ]]; then
  echo "Launch path exists but is not a complete git clone: ${launch_dir}" >&2
  exit 2
fi
if [[ ! -d "${launch_dir}/.git" ]]; then
  staging="${launch_dir}.tmp.$$"
  git clone --no-checkout "${seed_root}" "${staging}"
  git -C "${staging}" checkout --detach "${LAUNCH_SHA}"
  mv "${staging}" "${launch_dir}"
  staging=""
fi

actual_sha="$(git -C "${launch_dir}" rev-parse HEAD)"
if [[ "${actual_sha}" != "${LAUNCH_SHA}" ]]; then
  echo "Checkout mismatch: expected ${LAUNCH_SHA}, got ${actual_sha}" >&2
  exit 2
fi
if [[ -n "$(git -C "${launch_dir}" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing a modified immutable launch tree: ${launch_dir}" >&2
  exit 2
fi

if [[ ! -x "${launch_dir}/.venv/bin/python" ]]; then
  (cd "${launch_dir}" && uv sync --frozen --extra dev)
fi
mkdir -p "${launch_dir}/runs/slurm"

sbatch --parsable \
  --chdir="${launch_dir}" \
  --export="ALL,LAUNCH_SHA=${LAUNCH_SHA},SYNTHETIC_DATASET_ID=${SYNTHETIC_DATASET_ID}" \
  "${launch_dir}/slurm/smoke.sbatch"
