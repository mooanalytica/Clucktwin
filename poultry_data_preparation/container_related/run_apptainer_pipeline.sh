#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODULE_PARENT="$(cd "${MODULE_DIR}/.." && pwd)"
CONTAINER_ROOT="${MODULE_DIR}/containers"
SIF_PATH="${CONTAINER_ROOT}/poultry_data_preparation.sif"

is_wsl() {
  grep -qi "microsoft" /proc/sys/kernel/osrelease 2>/dev/null
}

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "Missing SIF image: ${SIF_PATH}"
  echo "Build it first with:"
  echo "  bash ${SCRIPT_DIR}/build_local_sif.sh"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage:"
  echo "  bash ${SCRIPT_DIR}/run_apptainer_pipeline.sh --config /path/to/config.yaml --stage all [other args]"
  echo
  echo "Run this from the dataset or project root you want to bind into the container."
  exit 1
fi

if [[ -z "${APPTAINER_CACHEDIR:-}" ]]; then
  if is_wsl; then
    export APPTAINER_CACHEDIR="${HOME}/apptainer_poultry_cache"
  else
    export APPTAINER_CACHEDIR="${CONTAINER_ROOT}/apptainer_cache"
  fi
fi

if [[ -z "${APPTAINER_TMPDIR:-}" ]]; then
  if is_wsl; then
    export APPTAINER_TMPDIR="${HOME}/apptainer_poultry_tmp"
  else
    export APPTAINER_TMPDIR="${CONTAINER_ROOT}/apptainer_tmp"
  fi
fi

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

apptainer exec \
  --cleanenv \
  --env "PYTHONPATH=${PWD}:${MODULE_PARENT}:/opt/poultry_project" \
  --bind "${PWD}:${PWD}" \
  --bind "${MODULE_PARENT}:${MODULE_PARENT}" \
  --pwd "${PWD}" \
  "${SIF_PATH}" \
  python -m poultry_data_preparation.src.main "$@"
