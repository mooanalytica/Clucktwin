#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/build_local_docker_image.sh"
"${SCRIPT_DIR}/convert_docker_archive_to_sif.sh"
