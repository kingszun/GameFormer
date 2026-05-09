#!/bin/bash
# open_loop_planning training from scratch (pCloud public link, no auth)
#
# 동작:
#   1. pCloud public download (open_loop train_tar + valid_tar) — auth 필요 없음
#   2. parallel untar (xargs -P 64)
#   3. cleanup tar dirs
#   4. launch training (single GPU)
#
# 환경 변수:
#   PARALLEL_DL          pCloud download parallel (default 8)
#   PARALLEL_UNTAR       untar parallel (default 64)
#   BATCH_SIZE           batch size (default 128)
#   EPOCHS               train_epochs (default 20)
#   LR                   learning_rate (default 2.83e-4 — sqrt scaling from paper 1e-4)
#   LEVELS               reasoning levels (default 4)
#   NAME                 run name (default op_full)
#   DATA_BASE            data root (default /workspace/data/processed/open_loop)
#   LOG_PATH             log root (default /workspace/logs)
#   CODE_TRAIN           pCloud share code for train_tar (default p3vctalK)
#   CODE_VALID           pCloud share code for valid_tar (default zaM7)

set -e
PARALLEL_DL=${PARALLEL_DL:-8}
PARALLEL_UNTAR=${PARALLEL_UNTAR:-64}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-20}
LR=${LR:-2.83e-4}
LEVELS=${LEVELS:-4}
NAME=${NAME:-op_full}
DATA_BASE=${DATA_BASE:-/workspace/data/processed/open_loop}
LOG_PATH=${LOG_PATH:-/workspace/logs}
CODE_TRAIN=${CODE_TRAIN:-p3vctalK}
CODE_VALID=${CODE_VALID:-zaM7}

TRAIN_TAR=$DATA_BASE/train_tar
VALID_TAR=$DATA_BASE/valid_tar
TRAIN_DIR=$DATA_BASE/train
VALID_DIR=$DATA_BASE/valid

mkdir -p $LOG_PATH $TRAIN_TAR $VALID_TAR $TRAIN_DIR $VALID_DIR
CHAIN_LOG=$LOG_PATH/chain_open_loop.log
ts() { date -u +'%Y-%m-%d %H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $CHAIN_LOG; touch $LOG_PATH/chain.error; exit 1; }
exec > >(tee -a $CHAIN_LOG) 2>&1

REPO_ROOT=$(dirname $(dirname $(dirname $(readlink -f $0))))/..
HELPER=$REPO_ROOT/scripts/cloud/transfer/pcloud-public-download.py

echo "[$(ts)] === pod-train-from-scratch open_loop ==="
echo "[$(ts)] BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS LR=$LR LEVELS=$LEVELS"
echo "[$(ts)] CODE_TRAIN=$CODE_TRAIN CODE_VALID=$CODE_VALID"

# === STEP 1: download from pCloud public ===
echo "[$(ts)] === STEP 1: download train_tar (pCloud public) ==="
START=$(date +%s)
python3 $HELPER $CODE_TRAIN $TRAIN_TAR --parallel $PARALLEL_DL >> $LOG_PATH/dl_op_train.log 2>&1 || abort "train_tar download failed"
echo "[$(ts)] train_tar download $((($(date +%s) - START)))s"
echo "[$(ts)] === STEP 1b: download valid_tar (pCloud public) ==="
START=$(date +%s)
python3 $HELPER $CODE_VALID $VALID_TAR --parallel $PARALLEL_DL >> $LOG_PATH/dl_op_valid.log 2>&1 || abort "valid_tar download failed"
echo "[$(ts)] valid_tar download $((($(date +%s) - START)))s"

TPARTS=$(ls $TRAIN_TAR | wc -l)
VPARTS=$(ls $VALID_TAR | wc -l)
[ "$TPARTS" = "45" ] || abort "train_tar parts $TPARTS != 45 expected"
[ "$VPARTS" = "19" ] || abort "valid_tar parts $VPARTS != 19 expected"
echo "[$(ts)] train=$TPARTS parts, valid=$VPARTS parts"

# === STEP 2: parallel untar train (8 batches) ===
echo "[$(ts)] === STEP 2: parallel untar train (P=$PARALLEL_UNTAR) ==="
cd $DATA_BASE
BATCHES=$(ls $TRAIN_TAR | grep -oE 'open_loop_train_b[0-9]+_[0-9]+' | sort -u)
START=$(date +%s)
echo "$BATCHES" | xargs -P $PARALLEL_UNTAR -I {} bash -c '
    BATCH=$1
    BS=$(date +%s)
    cd '"$DATA_BASE"'
    cat '"$TRAIN_TAR"'/${BATCH}.tar.part_* | tar -xf - 2>/dev/null
    BE=$(date +%s)
    echo "[$(date -u +%H:%M:%S)] $BATCH untar $((BE - BS))s" >> '"$CHAIN_LOG"'
' bash {}
TCNT=$(find train -type f -name '*.npz' | wc -l)
echo "[$(ts)] train untar done — $TCNT files, $((($(date +%s) - START)))s"

# === STEP 3: untar valid (chain v3 single batch) ===
echo "[$(ts)] === STEP 3: untar valid ==="
START=$(date +%s)
cat $VALID_TAR/valid.tar.part_* | tar -xf - || abort "valid untar failed"
VCNT=$(find valid -type f -name '*.npz' | wc -l)
echo "[$(ts)] valid untar done — $VCNT files, $((($(date +%s) - START)))s"

# === STEP 4: cleanup tar dirs ===
echo "[$(ts)] === STEP 4: cleanup tar dirs ==="
rm -rf $TRAIN_TAR $VALID_TAR
df -h / | tail -1

# === STEP 5: launch training ===
echo "[$(ts)] === STEP 5: launch training (1 GPU) ==="
mkdir -p $LOG_PATH/runs
cd /workspace/GameFormer/open_loop_planning
TRAIN_LOG=$LOG_PATH/train_${NAME}.log
TRAINING_LOG_HOME=$LOG_PATH/runs nohup python train.py \
    --name=$NAME \
    --train_set=$TRAIN_DIR \
    --valid_set=$VALID_DIR \
    --batch_size=$BATCH_SIZE \
    --train_epochs=$EPOCHS \
    --learning_rate=$LR \
    --levels=$LEVELS \
    > $TRAIN_LOG 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $LOG_PATH/train_${NAME}.pid
sleep 5
ALIVE=$(kill -0 $TRAIN_PID 2>/dev/null && echo yes || echo no)
echo "[$(ts)] training launched pid=$TRAIN_PID alive=$ALIVE log=$TRAIN_LOG"
touch $LOG_PATH/chain.done
echo "[$(ts)] === chain done ==="
