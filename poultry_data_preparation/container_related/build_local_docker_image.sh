#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_ROOT="${MODULE_DIR}/containers"
IMAGE_TAG="${IMAGE_TAG:-poultry_data_preparation:latest}"
DOCKER_ARCHIVE_TAR="${CONTAINER_ROOT}/poultry_data_preparation_docker.tar"
DOCKER_ARCHIVE_GZ="${CONTAINER_ROOT}/poultry_data_preparation_docker.tar.gz"

mkdir -p "${CONTAINER_ROOT}"
rm -f "${DOCKER_ARCHIVE_TAR}" "${DOCKER_ARCHIVE_GZ}"

echo "[build] Module dir: ${MODULE_DIR}"
echo "[build] Image tag: ${IMAGE_TAG}"
echo "[build] Archive path: ${DOCKER_ARCHIVE_GZ}"

docker build -t "${IMAGE_TAG}" -f "${MODULE_DIR}/container_related/Dockerfile" "${MODULE_DIR}"
docker save "${IMAGE_TAG}" | gzip > "${DOCKER_ARCHIVE_GZ}"

echo "[done] Created Docker archive at: ${DOCKER_ARCHIVE_GZ}"
