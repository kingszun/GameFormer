#!/usr/bin/env bash
# open_loop_planning 학습 (container 내부 실행)
#
# Env vars:
#   TRAIN_SPLIT   train data split 이름 (default: train) - data/processed/open_loop/${TRAIN_SPLIT}/
#   VALID_SPLIT   valid data split 이름 (default: train, smoke test 시 동일 사용)
#   BATCH_SIZE    batch size (default: 8 - 3060 12GB 기준 안전치, H200은 32~64까지 가능)
#   EPOCHS        train epoch 수 (default: 1, 본격 학습은 20)
#   LR            learning rate (default: 1e-4)
#   LEVELS        reasoning levels (default: 4)
#   NAME          log/checkpoint 이름 (default: Exp1) - training_log/${NAME}/
#
# Usage:
#   bash scripts/06-open_loop_train.sh
#   BATCH_SIZE=32 EPOCHS=20 NAME=run01 bash scripts/06-open_loop_train.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALID_SPLIT="${VALID_SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
LEVELS="${LEVELS:-4}"
NAME="${NAME:-Exp1}"

TRAIN_SET="/workspace/GameFormer/data/processed/open_loop/${TRAIN_SPLIT}"
VALID_SET="/workspace/GameFormer/data/processed/open_loop/${VALID_SPLIT}"

docker compose exec gameformer bash -lc "
cd open_loop_planning
python train.py \
    --name ${NAME} \
    --train_set '${TRAIN_SET}' \
    --valid_set '${VALID_SET}' \
    --batch_size ${BATCH_SIZE} \
    --train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --levels ${LEVELS}
"
