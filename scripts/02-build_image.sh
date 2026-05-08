#!/usr/bin/env bash
# docker image build (host 실행)
# - 이미지 tag/repo는 .env 의 IMAGE_REPO/IMAGE_TAG 로 제어
# - usage: bash scripts/02-build_image.sh [extra docker compose build flags]

set -euo pipefail
cd "$(dirname "$0")/.."

docker compose build "$@"
docker images "$(grep -E '^IMAGE_REPO=' .env 2>/dev/null | cut -d= -f2 || echo gameformer)"
