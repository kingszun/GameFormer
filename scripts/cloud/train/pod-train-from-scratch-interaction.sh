#!/bin/bash
# interaction_prediction training from scratch (pCloud public link, no auth)
#
# 동작:
#   1. pCloud public download (interaction train_tar + valid_tar) — auth 필요 없음
#   2. parallel untar (xargs -P 64)
#   3. cleanup tar dirs
#   4. launch training (multi-GPU DDP)
#
# 환경 변수:
#   PARALLEL_DL          pCloud download parallel (default 16)
#   PARALLEL_UNTAR       untar parallel (default 64)
#   NPROC                GPU 수 (default 4)
#   BATCH_SIZE           per-GPU batch (default 64 — paper 16 의 4x)
#   EPOCHS               training_epochs (default 30, paper)
#   LR                   learning_rate (default 2e-4 — sqrt scaling from paper 1e-4)
#   WORKERS              dataloader workers per GPU (default 16)
#   NAME                 run name (default ip_full)
#   MASTER_PORT          DDP master port (default 28596)
#   DATA_BASE            data root (default /workspace/data/processed/interaction)
#   LOG_PATH             log root (default /workspace/logs)
#   CODE_TRAIN           pCloud share code for train_tar (default kt4)
#   CODE_VALID           pCloud share code for valid_tar (default SYpctalK)

set -e
PARALLEL_DL=${PARALLEL_DL:-16}
PARALLEL_UNTAR=${PARALLEL_UNTAR:-64}
NPROC=${NPROC:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-30}
LR=${LR:-2e-4}
WORKERS=${WORKERS:-16}
NAME=${NAME:-ip_full}
MASTER_PORT=${MASTER_PORT:-28596}
DATA_BASE=${DATA_BASE:-/workspace/data/processed/interaction}
LOG_PATH=${LOG_PATH:-/workspace/logs}
CODE_TRAIN=${CODE_TRAIN:-kt4}
CODE_VALID=${CODE_VALID:-SYpctalK}

TRAIN_TAR=$DATA_BASE/train_tar
VALID_TAR=$DATA_BASE/valid_tar
TRAIN_DIR=$DATA_BASE/train
VALID_DIR=$DATA_BASE/valid

mkdir -p $LOG_PATH $TRAIN_TAR $VALID_TAR $TRAIN_DIR $VALID_DIR
CHAIN_LOG=$LOG_PATH/chain_interaction.log
ts() { date -u +'%Y-%m-%d %H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $CHAIN_LOG; touch $LOG_PATH/chain.error; exit 1; }
exec > >(tee -a $CHAIN_LOG) 2>&1

REPO_ROOT=$(dirname $(dirname $(dirname $(readlink -f $0))))/..
HELPER=$REPO_ROOT/scripts/cloud/transfer/pcloud-public-download.py

echo "[$(ts)] === pod-train-from-scratch interaction ==="
echo "[$(ts)] NPROC=$NPROC BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS LR=$LR WORKERS=$WORKERS"
echo "[$(ts)] CODE_TRAIN=$CODE_TRAIN CODE_VALID=$CODE_VALID"

# === STEP 1: download from pCloud public ===
echo "[$(ts)] === STEP 1: download train_tar (pCloud public, ~1.1 TiB) ==="
START=$(date +%s)
python3 $HELPER $CODE_TRAIN $TRAIN_TAR --parallel $PARALLEL_DL >> $LOG_PATH/dl_int_train.log 2>&1 || abort "train_tar download failed"
echo "[$(ts)] train_tar download $((($(date +%s) - START)))s"
echo "[$(ts)] === STEP 1b: download valid_tar (pCloud public, ~24 GiB) ==="
START=$(date +%s)
python3 $HELPER $CODE_VALID $VALID_TAR --parallel $PARALLEL_DL >> $LOG_PATH/dl_int_valid.log 2>&1 || abort "valid_tar download failed"
echo "[$(ts)] valid_tar download $((($(date +%s) - START)))s"

TPARTS=$(ls $TRAIN_TAR | wc -l)
[ "$TPARTS" = "147" ] || abort "train_tar parts $TPARTS != 147 expected"
echo "[$(ts)] train_tar=$TPARTS parts"

# === STEP 2: untar valid (b0149 full set 만; b0050 partial 은 skip) ===
echo "[$(ts)] === STEP 2: untar valid (b0149 full set) ==="
cd $DATA_BASE
START=$(date +%s)
cat $VALID_TAR/interaction_valid_b0000_0149.tar.part_* | tar -xf - || abort "valid untar failed"
VCNT=$(find valid -type f -name '*.npz' | wc -l)
echo "[$(ts)] valid done — $VCNT files, $((($(date +%s) - START)))s"

# === STEP 3: parallel untar train (19 batches) ===
echo "[$(ts)] === STEP 3: parallel untar train (P=$PARALLEL_UNTAR) ==="
BATCHES=$(ls $TRAIN_TAR | grep -oE 'interaction_train_b[0-9]+_[0-9]+' | sort -u)
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

# === STEP 4: cleanup tar dirs ===
echo "[$(ts)] === STEP 4: cleanup tar dirs ==="
rm -rf $TRAIN_TAR $VALID_TAR
df -h / | tail -1

# === STEP 5: launch training (DDP) ===
echo "[$(ts)] === STEP 5: launch training (DDP NPROC=$NPROC) ==="
mkdir -p $LOG_PATH/runs
cd /workspace/GameFormer/interaction_prediction
TRAIN_LOG=$LOG_PATH/train_${NAME}.log
nohup python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT \
    train.py \
    --batch_size=$BATCH_SIZE \
    --training_epochs=$EPOCHS \
    --learning_rate=$LR \
    --train_set=$TRAIN_DIR \
    --valid_set=$VALID_DIR \
    --name=$NAME \
    --workers=$WORKERS \
    > $TRAIN_LOG 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $LOG_PATH/train_${NAME}.pid
sleep 5
ALIVE=$(kill -0 $TRAIN_PID 2>/dev/null && echo yes || echo no)
echo "[$(ts)] training launched pid=$TRAIN_PID alive=$ALIVE log=$TRAIN_LOG"
touch $LOG_PATH/chain.done
echo "[$(ts)] === chain done ==="
