#!/bin/bash
# pod 안에서 4 subset sequential preprocess (volume direct write, pCloud upload 없음)
#
# 4 subset 처리: interaction train, open_loop train, interaction valid, open_loop valid
# 각 subset 의 shard_range 환경변수로 받음 (multi-pod 분담).
# SAVE_PATH 가 volume mount (multi-pod 같은 path file unique 라 안전).
# 학습 pod 가 same volume mount 로 직접 read.
#
# 환경 변수:
#   WORKERS              multiprocessing pool (default 28)
#   LOG_PATH             log dir (default /workspace/logs/$(hostname))
#   PROC_BASE            processed root (default /workspace/processed)
#   RAW_BASE             raw root (default /workspace/raw)
#   SHARD_TRAIN_INT      training shard range (interaction)
#   SHARD_TRAIN_OL       training_20s shard range (open_loop)
#   SHARD_VALID_INT      validation_interactive shard range (interaction)
#   SHARD_VALID_OL       validation shard range (open_loop)
#   SUBSETS              처리할 subset list (default "interaction_train open_loop_train interaction_valid open_loop_valid")
#
# 예 (multi-pod wrapper 가 호출):
#   SHARD_TRAIN_INT=0:167 SHARD_TRAIN_OL=0:58 SHARD_VALID_INT=0:25 SHARD_VALID_OL=0:25 \
#   bash pod-preprocess-volume-direct.sh

set -e
ulimit -n 65536

WORKERS=${WORKERS:-28}
LOG_PATH=${LOG_PATH:-/workspace/logs/$(hostname)}
PROC_BASE=${PROC_BASE:-/workspace/processed}
RAW_BASE=${RAW_BASE:-/workspace/raw}
SUBSETS=${SUBSETS:-"interaction_train open_loop_train interaction_valid open_loop_valid"}

mkdir -p $LOG_PATH

ts() { date -u +'%H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $LOG_PATH/preprocess_volume.error; exit 1; }

# subset config: name | raw_subdir | preprocess_dir (cwd) | save_subdir | shard_range_env
declare -A CFG
CFG[interaction_train]="training|interaction_prediction|interaction/train|$SHARD_TRAIN_INT"
CFG[open_loop_train]="training_20s|open_loop_planning|open_loop/train|$SHARD_TRAIN_OL"
CFG[interaction_valid]="validation_interactive|interaction_prediction|interaction/valid|$SHARD_VALID_INT"
CFG[open_loop_valid]="validation|open_loop_planning|open_loop/valid|$SHARD_VALID_OL"

echo "[$(ts)] === preprocess volume direct start (workers=$WORKERS, hostname=$(hostname)) ==="
echo "[$(ts)] subsets: $SUBSETS"

for subset in $SUBSETS; do
    cfg=${CFG[$subset]:-}
    [ -z "$cfg" ] && abort "unknown subset: $subset"
    IFS='|' read -r raw_sub cwd save_sub shard_range <<< "$cfg"

    LOAD_PATH=$RAW_BASE/$raw_sub
    SAVE_PATH=$PROC_BASE/$save_sub
    PREPROC_LOG=$LOG_PATH/preprocess_${subset}.log

    if [ ! -d "$LOAD_PATH" ]; then
        echo "[$(ts)] SKIP $subset — $LOAD_PATH 없음"
        continue
    fi

    SUBSET_START=$(date +%s)
    BEFORE=$(find $SAVE_PATH -name '*.npz' 2>/dev/null | wc -l)
    mkdir -p $SAVE_PATH

    echo
    echo "[$(ts)] === $subset (shard_range=${shard_range:-all}) raw=$LOAD_PATH save=$SAVE_PATH ==="
    cd /workspace/GameFormer/$cwd
    python data_process.py \
        --load_path $LOAD_PATH \
        --save_path $SAVE_PATH \
        --use_multiprocessing \
        --processes $WORKERS \
        ${shard_range:+--shard_range $shard_range} \
        > $PREPROC_LOG 2>&1 || abort "$subset preprocess failed (log: $PREPROC_LOG)"
    AFTER=$(find $SAVE_PATH -name '*.npz' | wc -l)
    SUBSET_END=$(date +%s)
    echo "[$(ts)] $subset done — $((AFTER - BEFORE)) new files ($AFTER total at $SAVE_PATH), $((SUBSET_END - SUBSET_START))s"
done

touch $LOG_PATH/preprocess_volume.done
echo
echo "[$(ts)] === ALL subsets done ==="
