#!/usr/bin/env bash
# WOMD raw tfrecord -> open_loop_planning .npz preprocess
#
# host (docker compose) / pod (직접) 자동 dispatch.
#
# Env vars:
#   MODE              host | pod | auto (default auto - /.dockerenv 또는 IN_POD env 면 pod)
#   WOMD_SUBSET       raw subset 디렉토리명 (default: training_20s)
#   SPLIT             결과 split 이름 (default: train) - data/processed/open_loop/${SPLIT}/
#   DATASET_HOME      data root (host default: /workspace/GameFormer/data, pod default: /workspace/data)
#
# Usage:
#   bash scripts/05-open_loop_preprocess.sh
#   WOMD_SUBSET=validation_interactive SPLIT=valid bash scripts/05-open_loop_preprocess.sh
#   IN_POD=1 bash scripts/05-open_loop_preprocess.sh   # pod 모드 명시

set -euo pipefail
cd "$(dirname "$0")/.."

WOMD_SUBSET="${WOMD_SUBSET:-training_20s}"
SPLIT="${SPLIT:-train}"

# MODE auto-detect
if [ -z "${MODE:-}" ]; then
    if [ -f /.dockerenv ] || [ -n "${IN_POD:-}" ]; then
        MODE=pod
    else
        MODE=host
    fi
fi

# path defaults by MODE (절대 path, container/pod 시점)
if [ "$MODE" = "pod" ]; then
    DATASET_HOME="${DATASET_HOME:-/workspace/data}"
else
    DATASET_HOME="${DATASET_HOME:-/workspace/GameFormer/data}"
fi

INPUT="${DATASET_HOME}/raw/${WOMD_SUBSET}"
OUTPUT="${DATASET_HOME}/processed/open_loop/${SPLIT}"

CMD="mkdir -p '${OUTPUT}' && \
cd open_loop_planning && \
python data_process.py \
    --load_path '${INPUT}' \
    --save_path '${OUTPUT}' \
    --use_multiprocessing"

echo "[scripts/05] MODE=${MODE} DATASET_HOME=${DATASET_HOME}"
echo "[scripts/05] INPUT=${INPUT}"
echo "[scripts/05] OUTPUT=${OUTPUT}"

if [ "$MODE" = "pod" ]; then
    bash -lc "$CMD"
else
    docker compose exec gameformer bash -lc "$CMD"
fi
