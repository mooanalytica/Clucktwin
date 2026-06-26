#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_ROOT="${MODULE_DIR}/containers"
SIF_PATH="${CONTAINER_ROOT}/poultry_data_preparation.sif"

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "Missing SIF image: ${SIF_PATH}"
  echo "Build it first with:"
  echo "  bash ${SCRIPT_DIR}/build_local_sif.sh"
  exit 1
fi

echo "[smoke] Checking Python in container"
apptainer exec --cleanenv "${SIF_PATH}" python --version

echo "[smoke] Checking ffmpeg in container"
apptainer exec --cleanenv "${SIF_PATH}" ffmpeg -version

echo "[smoke] Checking exiftool in container"
apptainer exec --cleanenv "${SIF_PATH}" exiftool -ver

echo "[smoke] Running pytest inside container against the bound module workspace"
apptainer exec \
  --cleanenv \
  --env "PYTHONPATH=/workspace:/workspace/poultry_data_preparation:/opt/poultry_project" \
  --bind "${MODULE_DIR}:/workspace/poultry_data_preparation" \
  --pwd /workspace/poultry_data_preparation \
  "${SIF_PATH}" \
  python -m pytest tests -q
