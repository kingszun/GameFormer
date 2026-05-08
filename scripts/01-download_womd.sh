#!/usr/bin/env bash
# WOMD scenario tfrecord download.
#
# Wrapper that delegates to scripts/01-download_womd.py.
# - image: /opt/venv/gcs/bin/python (KAK-18 aux venv with google-cloud-storage)
# - host : system python (host 에 google-cloud-storage 또는 google-cloud-sdk 필요)
#
# Prerequisites:
#   1. Waymo Open Dataset license 동의 - https://waymo.com/open/licensing/
#   2. ADC 또는 gcloud auth login (Waymo 등록 계정과 동일).
#      cloud pod 는 GOOGLE_APPLICATION_CREDENTIALS env 로 ADC json 경로 명시.
#
# Usage:
#   bash scripts/01-download_womd.sh
#   WOMD_SUBSET=training_20s WOMD_SHARDS=2 bash scripts/01-download_womd.sh
#   WOMD_SHARDS=all bash scripts/01-download_womd.sh   # 전체 (수백 GB)

set -euo pipefail

WOMD_VERSION="${WOMD_VERSION:-1_2_1}"
WOMD_SUBSET="${WOMD_SUBSET:-validation_interactive}"
WOMD_DEST="${WOMD_DEST:-${DATASET_HOME:-./data}/raw}"
WOMD_SHARDS="${WOMD_SHARDS:-2}"

if [ -x /opt/venv/gcs/bin/python ]; then
    PYTHON=/opt/venv/gcs/bin/python
else
    PYTHON="${PYTHON:-python3}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$PYTHON" "${SCRIPT_DIR}/01-download_womd.py" \
    --version "$WOMD_VERSION" \
    --subset "$WOMD_SUBSET" \
    --shards "$WOMD_SHARDS" \
    --dest "$WOMD_DEST"
