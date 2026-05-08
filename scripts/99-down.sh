#!/usr/bin/env bash
# container 종료 + network 제거 (image/volume은 보존)
# - 완전 정리는: docker compose down --rmi local -v
# - usage: bash scripts/99-down.sh

set -euo pipefail
cd "$(dirname "$0")/.."

docker compose down "$@"
