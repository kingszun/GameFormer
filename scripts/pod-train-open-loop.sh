#!/bin/bash
# open_loop_planning training chain (pod, single GPU)
#
# 동작:
#   1. rclone download wait (open_loop/train_tar + valid_tar)
#   2. untar train (모든 batch — open_loop_train_bXXXX_YYYY.tar.part_*)
#   3. untar valid (chain v3 single batch — valid.tar.part_*)
#   4. cleanup tar dirs (free disk)
#   5. launch training (single GPU)
#
# 환경 변수:
#   BATCH_SIZE          batch (default 64)
#   EPOCHS              train_epochs (default 20)
#   LR                  learning_rate (default 2e-4, paper sqrt scaling)
#   LEVELS              reasoning levels (default 4)
#   NAME                run name (default op_full)
#   DATA_BASE           data root (default /workspace/data/processed/open_loop)
#   LOG_PATH            log root (default /workspace/logs)

set -e
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-20}
LR=${LR:-2e-4}
LEVELS=${LEVELS:-4}
NAME=${NAME:-op_full}
DATA_BASE=${DATA_BASE:-/workspace/data/processed/open_loop}
LOG_PATH=${LOG_PATH:-/workspace/logs}
TRAIN_TAR=$DATA_BASE/train_tar
VALID_TAR=$DATA_BASE/valid_tar
TRAIN_DIR=$DATA_BASE/train
VALID_DIR=$DATA_BASE/valid

mkdir -p $LOG_PATH $TRAIN_DIR $VALID_DIR

CHAIN_LOG=$LOG_PATH/chain_open_loop.log
ts() { date -u +'%Y-%m-%d %H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $CHAIN_LOG; touch $LOG_PATH/chain.error; exit 1; }
exec > >(tee -a $CHAIN_LOG) 2>&1

echo "[$(ts)] === pod-train-open-loop chain start ==="
echo "[$(ts)] BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS LR=$LR LEVELS=$LEVELS"

# === STEP 1: wait for rclone download ===
echo "[$(ts)] === STEP 1: wait for rclone download ==="
WAIT_COUNT=0
while pgrep -f 'rclone copy pcloud' > /dev/null; do
    PROCS=$(pgrep -f 'rclone copy pcloud' | wc -l)
    TSIZE=$(du -sh $TRAIN_TAR 2>/dev/null | awk '{print $1}')
    VSIZE=$(du -sh $VALID_TAR 2>/dev/null | awk '{print $1}')
    TPARTS=$(ls $TRAIN_TAR 2>/dev/null | wc -l)
    VPARTS=$(ls $VALID_TAR 2>/dev/null | wc -l)
    echo "[$(ts)] rclone procs=$PROCS train=$TSIZE/$TPARTS parts valid=$VSIZE/$VPARTS parts"
    sleep 60
    WAIT_COUNT=$((WAIT_COUNT + 1))
    [ $WAIT_COUNT -gt 60 ] && abort "rclone wait > 60 min"
done
echo "[$(ts)] rclone all done"

TRAIN_PARTS=$(ls $TRAIN_TAR | wc -l)
VALID_PARTS=$(ls $VALID_TAR | wc -l)
[ "$TRAIN_PARTS" = "45" ] || abort "train_tar parts $TRAIN_PARTS != 45 expected"
[ "$VALID_PARTS" = "19" ] || abort "valid_tar parts $VALID_PARTS != 19 expected"
echo "[$(ts)] train_tar=$TRAIN_PARTS parts, valid_tar=$VALID_PARTS parts"

# === STEP 2: untar train (all batches) ===
echo "[$(ts)] === STEP 2: untar train (all batches) ==="
cd $DATA_BASE
BATCHES=$(ls $TRAIN_TAR | grep -oE 'open_loop_train_b[0-9]+_[0-9]+' | sort -u)
echo "[$(ts)] $(echo "$BATCHES" | wc -l) batches to untar"
for batch in $BATCHES; do
    BS=$(date +%s)
    cat $TRAIN_TAR/${batch}.tar.part_* | tar -xf - || abort "$batch untar failed"
    BE=$(date +%s)
    CNT=$(find train -type f -name '*.npz' | wc -l)
    echo "[$(ts)] $batch untar $((BE - BS))s (total $CNT files)"
done

# === STEP 3: untar valid (chain v3 single batch) ===
echo "[$(ts)] === STEP 3: untar valid (chain v3) ==="
START=$(date +%s)
cat $VALID_TAR/valid.tar.part_* | tar -xf - || abort "valid untar failed"
VCNT=$(find valid -type f -name '*.npz' | wc -l)
echo "[$(ts)] valid done: $VCNT files, $(($(date +%s) - START))s"

# === STEP 4: cleanup tar dirs ===
echo "[$(ts)] === STEP 4: cleanup tar dirs ==="
rm -rf $TRAIN_TAR $VALID_TAR
df -h / | tail -1

# === STEP 5: launch training (single GPU) ===
echo "[$(ts)] === STEP 5: launch training (single GPU) ==="
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
