#!/usr/bin/env bash
# open_loop_planning 학습
#
# host (docker compose) / pod (직접) 자동 dispatch.
# train.py 의 산출물 위치는 TRAINING_LOG_HOME env 로 외부화 (KAK-33).
#
# Env vars:
#   MODE                host | pod | auto (default auto - /.dockerenv 또는 IN_POD env 면 pod)
#   TRAIN_SPLIT         train data split (default: train)
#   VALID_SPLIT         valid data split (default: train, smoke 시 동일)
#   BATCH_SIZE          batch size (default: 8 - 3060 12GB 기준 안전치)
#   EPOCHS              train epoch (default: 1)
#   LR                  learning rate (default: 1e-4)
#   LEVELS              reasoning levels (default: 4)
#   NAME                log/checkpoint 이름 (default: Exp1)
#   DATASET_HOME        data root (host default: /workspace/GameFormer/data, pod default: /workspace/data)
#   TRAINING_LOG_HOME   train output root (host default: /workspace/GameFormer/training_log, pod default: /workspace/logs/runs)
#
# Usage:
#   bash scripts/06-open_loop_train.sh
#   BATCH_SIZE=32 EPOCHS=20 NAME=run01 bash scripts/06-open_loop_train.sh
#   IN_POD=1 BATCH_SIZE=32 NAME=cloud_full bash scripts/06-open_loop_train.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALID_SPLIT="${VALID_SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-4}"
LEVELS="${LEVELS:-4}"
NAME="${NAME:-Exp1}"

# MODE auto-detect
if [ -z "${MODE:-}" ]; then
    if [ -f /.dockerenv ] || [ -n "${IN_POD:-}" ]; then
        MODE=pod
    else
        MODE=host
    fi
fi

# path defaults by MODE (절대 path)
# KAK-56: pod 의 TRAINING_LOG_HOME 을 /workspace/logs/runs 로 일관 — entrypoint 의 log auto-tail 에 포함
if [ "$MODE" = "pod" ]; then
    DATASET_HOME="${DATASET_HOME:-/workspace/data}"
    TRAINING_LOG_HOME="${TRAINING_LOG_HOME:-${LOG_PATH:-/workspace/logs/$(hostname)}/runs}"
else
    DATASET_HOME="${DATASET_HOME:-/workspace/GameFormer/data}"
    TRAINING_LOG_HOME="${TRAINING_LOG_HOME:-/workspace/GameFormer/training_log}"
fi

TRAIN_SET="${DATASET_HOME}/processed/open_loop/${TRAIN_SPLIT}"
VALID_SET="${DATASET_HOME}/processed/open_loop/${VALID_SPLIT}"

CMD="mkdir -p '${TRAINING_LOG_HOME}' && \
cd open_loop_planning && \
TRAINING_LOG_HOME='${TRAINING_LOG_HOME}' python train.py \
    --name '${NAME}' \
    --train_set '${TRAIN_SET}' \
    --valid_set '${VALID_SET}' \
    --batch_size ${BATCH_SIZE} \
    --train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --levels ${LEVELS}"

echo "[scripts/06] MODE=${MODE} DATASET_HOME=${DATASET_HOME}"
echo "[scripts/06] TRAINING_LOG_HOME=${TRAINING_LOG_HOME}"
echo "[scripts/06] NAME=${NAME} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS}"

if [ "$MODE" = "pod" ]; then
    bash -lc "$CMD"
else
    docker compose exec gameformer bash -lc "$CMD"
fi
