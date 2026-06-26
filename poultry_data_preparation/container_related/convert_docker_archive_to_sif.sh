#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_ROOT="${MODULE_DIR}/containers"
DOCKER_TAR_GZ="${CONTAINER_ROOT}/poultry_data_preparation_docker.tar.gz"
DOCKER_TAR="${CONTAINER_ROOT}/poultry_data_preparation_docker.tar"
SIF_PATH="${CONTAINER_ROOT}/poultry_data_preparation.sif"

is_wsl() {
  grep -qi "microsoft" /proc/sys/kernel/osrelease 2>/dev/null
}

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

mkdir -p "${CONTAINER_ROOT}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

if [[ -f "${DOCKER_TAR_GZ}" && ( ! -f "${DOCKER_TAR}" || "${DOCKER_TAR_GZ}" -nt "${DOCKER_TAR}" ) ]]; then
  echo "[build] Decompressing Docker archive to ${DOCKER_TAR}"
  gunzip -c "${DOCKER_TAR_GZ}" > "${DOCKER_TAR}"
elif [[ ! -f "${DOCKER_TAR}" ]]; then
  echo "Missing Docker archive. Expected:"
  echo "  ${DOCKER_TAR}"
  echo "or"
  echo "  ${DOCKER_TAR_GZ}"
  exit 1
fi

echo "[build] APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}"
echo "[build] APPTAINER_TMPDIR=${APPTAINER_TMPDIR}"
echo "[build] SIF path: ${SIF_PATH}"

apptainer build --force "${SIF_PATH}" "docker-archive://${DOCKER_TAR}"

echo "[done] Created SIF at: ${SIF_PATH}"
