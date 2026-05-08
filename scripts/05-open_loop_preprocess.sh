#!/usr/bin/env bash
# WOMD raw tfrecord -> open_loop_planning용 .npz preprocess (container 내부 실행)
#
# Env vars:
#   WOMD_SUBSET   raw subset 디렉토리명 (default: training_20s)
#   SPLIT         처리 결과 split 이름 (default: train) - data/processed/open_loop/${SPLIT}/
#
# Usage:
#   bash scripts/05-open_loop_preprocess.sh
#   WOMD_SUBSET=validation_interactive SPLIT=valid bash scripts/05-open_loop_preprocess.sh

set -euo pipefail
cd "$(dirname "$0")/.."

WOMD_SUBSET="${WOMD_SUBSET:-training_20s}"
SPLIT="${SPLIT:-train}"

INPUT="/workspace/GameFormer/data/raw/${WOMD_SUBSET}"
OUTPUT="/workspace/GameFormer/data/processed/open_loop/${SPLIT}"

docker compose exec gameformer bash -lc "
mkdir -p '${OUTPUT}'
cd open_loop_planning
python data_process.py \
    --load_path '${INPUT}' \
    --save_path '${OUTPUT}' \
    --use_multiprocessing
"
