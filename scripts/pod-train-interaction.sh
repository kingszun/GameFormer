#!/bin/bash
# interaction_prediction training chain (pod, multi-GPU DDP)
#
# 동작:
#   1. rclone download wait (interaction/train_tar + valid_tar)
#   2. untar valid (b0149 full set 만)
#   3. untar train (모든 batch)
#   4. cleanup tar dirs (free disk)
#   5. launch training (DDP)
#
# 환경 변수:
#   NPROC               GPU 수 (default 4)
#   BATCH_SIZE          per-GPU batch (default 16, paper)
#   EPOCHS              training_epochs (default 30, paper)
#   LR                  learning_rate (default 1e-4, paper)
#   WORKERS             dataloader workers per GPU (default 8)
#   NAME                run name (default ip_full)
#   MASTER_PORT         DDP master port (default 28596)
#   DATA_BASE           data root (default /workspace/data/processed/interaction)
#   LOG_PATH            log root (default /workspace/logs)

set -e
NPROC=${NPROC:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-30}
LR=${LR:-1e-4}
WORKERS=${WORKERS:-8}
NAME=${NAME:-ip_full}
MASTER_PORT=${MASTER_PORT:-28596}
DATA_BASE=${DATA_BASE:-/workspace/data/processed/interaction}
LOG_PATH=${LOG_PATH:-/workspace/logs}
TRAIN_TAR=$DATA_BASE/train_tar
VALID_TAR=$DATA_BASE/valid_tar
TRAIN_DIR=$DATA_BASE/train
VALID_DIR=$DATA_BASE/valid

mkdir -p $LOG_PATH $TRAIN_DIR $VALID_DIR

CHAIN_LOG=$LOG_PATH/chain_interaction.log
ts() { date -u +'%Y-%m-%d %H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $CHAIN_LOG; touch $LOG_PATH/chain.error; exit 1; }
exec > >(tee -a $CHAIN_LOG) 2>&1

echo "[$(ts)] === pod-train-interaction chain start ==="
echo "[$(ts)] NPROC=$NPROC BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS LR=$LR"

# === STEP 1: wait for rclone download ===
echo "[$(ts)] === STEP 1: wait for rclone download ==="
WAIT_COUNT=0
while pgrep -f 'rclone copy pcloud' > /dev/null; do
    PROCS=$(pgrep -f 'rclone copy pcloud' | wc -l)
    SIZE=$(du -sh $TRAIN_TAR 2>/dev/null | awk '{print $1}')
    PARTS=$(ls $TRAIN_TAR 2>/dev/null | wc -l)
    echo "[$(ts)] rclone procs=$PROCS train_tar=$SIZE parts=$PARTS/147"
    sleep 60
    WAIT_COUNT=$((WAIT_COUNT + 1))
    [ $WAIT_COUNT -gt 90 ] && abort "rclone wait > 90 min"
done
echo "[$(ts)] rclone all done"

TRAIN_PARTS=$(ls $TRAIN_TAR | wc -l)
VALID_PARTS=$(ls $VALID_TAR | wc -l)
[ "$TRAIN_PARTS" = "147" ] || abort "train_tar parts $TRAIN_PARTS != 147 expected"
echo "[$(ts)] train_tar=$TRAIN_PARTS parts, valid_tar=$VALID_PARTS parts"

# === STEP 2: untar valid (b0149 full set 만) ===
echo "[$(ts)] === STEP 2: untar valid (b0149 full set) ==="
cd $DATA_BASE
START=$(date +%s)
cat $VALID_TAR/interaction_valid_b0000_0149.tar.part_* | tar -xf - || abort "valid untar failed"
VALID_COUNT=$(find valid -type f -name '*.npz' | wc -l)
echo "[$(ts)] valid done: $VALID_COUNT files, $(($(date +%s) - START))s"

# === STEP 3: untar train (all batches) ===
echo "[$(ts)] === STEP 3: untar train (all batches) ==="
BATCHES=$(ls $TRAIN_TAR | grep -oE 'interaction_train_b[0-9]+_[0-9]+' | sort -u)
echo "[$(ts)] $(echo "$BATCHES" | wc -l) batches to untar"
for batch in $BATCHES; do
    BS=$(date +%s)
    cat $TRAIN_TAR/${batch}.tar.part_* | tar -xf - || abort "$batch untar failed"
    BE=$(date +%s)
    CNT=$(find train -type f -name '*.npz' | wc -l)
    echo "[$(ts)] $batch untar $((BE - BS))s (total $CNT files)"
done

# === STEP 4: cleanup tar dirs ===
echo "[$(ts)] === STEP 4: cleanup tar dirs ==="
rm -rf $TRAIN_TAR $VALID_TAR
df -h / | tail -1

# === STEP 5: launch training (DDP) ===
echo "[$(ts)] === STEP 5: launch training (NPROC=$NPROC DDP) ==="
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
