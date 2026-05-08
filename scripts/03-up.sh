#!/usr/bin/env bash
# container 기동 (detached, sleep infinity로 상주)
# - 후속 작업은 'docker compose exec gameformer ...' 또는 04~06 script로
# - usage: bash scripts/03-up.sh

set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d
docker compose ps
