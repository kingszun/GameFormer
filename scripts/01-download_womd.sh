#!/usr/bin/env bash
# WOMD scenario tfrecord download (host 실행 전제)
#
# Prerequisites:
#   1. Waymo Open Dataset license 동의 - https://waymo.com/open/licensing/
#   2. host에 google-cloud-cli 설치 + gcloud auth login (Waymo 등록 계정과 동일)
#
# Usage:
#   bash download/download_womd.sh
#   WOMD_SUBSET=training_20s WOMD_SHARDS=2 bash download/download_womd.sh
#   WOMD_SHARDS=all bash download/download_womd.sh   # 전체 (수백 GB)

set -euo pipefail

WOMD_VERSION="${WOMD_VERSION:-1_2_1}"
WOMD_SUBSET="${WOMD_SUBSET:-validation_interactive}"
WOMD_DEST="${WOMD_DEST:-./data/raw}"
WOMD_SHARDS="${WOMD_SHARDS:-2}"

BUCKET="gs://waymo_open_dataset_motion_v_${WOMD_VERSION}/uncompressed/scenario/${WOMD_SUBSET}"
TARGET="${WOMD_DEST}/${WOMD_SUBSET}"

mkdir -p "${TARGET}"

if ! command -v gsutil >/dev/null 2>&1; then
    echo "ERROR: gsutil not installed. Install google-cloud-cli first." >&2
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q '@'; then
    echo "ERROR: gcloud not authenticated. Run: gcloud auth login" >&2
    exit 1
fi

echo "version: ${WOMD_VERSION}"
echo "subset:  ${WOMD_SUBSET}"
echo "source:  ${BUCKET}"
echo "target:  ${TARGET}"
echo "shards:  ${WOMD_SHARDS}"

if [ "${WOMD_SHARDS}" = "all" ]; then
    gsutil -m cp -r "${BUCKET}/*" "${TARGET}/"
else
    gsutil ls "${BUCKET}/" \
        | grep -E '\.tfrecord' \
        | head -n "${WOMD_SHARDS}" \
        | while read -r url; do
            gsutil cp "${url}" "${TARGET}/"
        done
fi

echo "done"
ls -lh "${TARGET}" | head -n 10
