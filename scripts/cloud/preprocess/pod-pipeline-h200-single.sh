#!/bin/bash
# Single big pod (H200 8 GPU 등) 의 download wait + untar + preprocess chain
#
# 동작:
#   1. raw + valid_tar download 끝까지 wait (rclone process check)
#   2. valid_tar untar → /root/data/processed/open_loop/valid
#   3. preprocess sequential — open_loop_train, interaction_valid, interaction_train (open_loop_valid 는 untar 라 skip)
#      - LOAD_PATH = container disk raw
#      - SAVE_PATH = container disk processed
#      - upload 안 함 (학습 same pod 에서 직접 read)
#
# 환경 변수:
#   WORKERS              multiprocessing pool (default 192)
#   RAW_BASE             raw root (default /root/data/raw)
#   PROC_BASE            processed root (default /root/data/processed)
#   VALID_TAR_BASE       valid_tar chunks dir (default /root/data/valid_tar)
#   SUBSETS              처리 순서 (default "open_loop_train interaction_valid interaction_train")

set -e
ulimit -n 65536

WORKERS=${WORKERS:-192}
RAW_BASE=${RAW_BASE:-/root/data/raw}
PROC_BASE=${PROC_BASE:-/root/data/processed}
VALID_TAR_BASE=${VALID_TAR_BASE:-/root/data/valid_tar}
LOG_PATH=${LOG_PATH:-/workspace/logs/$(hostname)}
SUBSETS=${SUBSETS:-"open_loop_train interaction_valid interaction_train"}

mkdir -p $LOG_PATH $PROC_BASE

ts() { date -u +'%H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $LOG_PATH/pipeline.error; exit 1; }

echo "[$(ts)] === H200 single pod pipeline start ==="
echo "[$(ts)] workers=$WORKERS raw_base=$RAW_BASE proc_base=$PROC_BASE"

# === STEP 1: wait for download ===
echo
echo "[$(ts)] === STEP 1: wait for raw + valid_tar download ==="
while true; do
    DL_PROCS=$(ps -ef | grep 'rclone copy pcloud' | grep -v grep | wc -l)
    if [ "$DL_PROCS" = "0" ]; then
        echo "[$(ts)] all rclone copy done"
        break
    fi
    echo "[$(ts)] $DL_PROCS rclone download still running, sleep 60"
    sleep 60
done

RAW_SIZE=$(du -sh $RAW_BASE 2>/dev/null | awk '{print $1}')
TAR_SIZE=$(du -sh $VALID_TAR_BASE 2>/dev/null | awk '{print $1}')
echo "[$(ts)] download done — raw=$RAW_SIZE valid_tar=$TAR_SIZE"

# === STEP 2: untar valid ===
echo
echo "[$(ts)] === STEP 2: untar valid_tar → $PROC_BASE/open_loop/valid ==="
mkdir -p $PROC_BASE/open_loop/valid
cd $PROC_BASE/open_loop/valid
UNTAR_LOG=$LOG_PATH/untar_valid.log
{ time cat $VALID_TAR_BASE/*.tar.part_* | tar -xf - --strip-components=1; } > $UNTAR_LOG 2>&1 || abort "untar valid failed (log: $UNTAR_LOG)"
VALID_COUNT=$(find $PROC_BASE/open_loop/valid -name '*.npz' | wc -l)
echo "[$(ts)] untar done — $VALID_COUNT files"
echo "[$(ts)] cleanup valid_tar (free disk)"
rm -rf $VALID_TAR_BASE

# === STEP 3: preprocess subsets ===
echo
echo "[$(ts)] === STEP 3: preprocess subsets ==="

# subset config: name | raw_subdir | preprocess_dir (cwd) | save_subdir
declare -A CFG
CFG[open_loop_train]="training_20s|open_loop_planning|open_loop/train"
CFG[interaction_valid]="validation_interactive|interaction_prediction|interaction/valid"
CFG[interaction_train]="training|interaction_prediction|interaction/train"

for subset in $SUBSETS; do
    cfg=${CFG[$subset]:-}
    [ -z "$cfg" ] && abort "unknown subset: $subset"
    IFS='|' read -r raw_sub cwd save_sub <<< "$cfg"

    LOAD_PATH=$RAW_BASE/$raw_sub
    SAVE_PATH=$PROC_BASE/$save_sub
    PREPROC_LOG=$LOG_PATH/preprocess_${subset}.log

    [ -d "$LOAD_PATH" ] || abort "$subset load_path 없음: $LOAD_PATH"
    mkdir -p $SAVE_PATH

    SUBSET_START=$(date +%s)
    BEFORE=$(find $SAVE_PATH -name '*.npz' 2>/dev/null | wc -l)
    echo
    echo "[$(ts)] === $subset start: raw=$LOAD_PATH save=$SAVE_PATH workers=$WORKERS ==="
    cd /workspace/GameFormer/$cwd
    python data_process.py \
        --load_path $LOAD_PATH \
        --save_path $SAVE_PATH \
        --use_multiprocessing \
        --processes $WORKERS \
        > $PREPROC_LOG 2>&1 || abort "$subset preprocess failed (log: $PREPROC_LOG)"
    AFTER=$(find $SAVE_PATH -name '*.npz' | wc -l)
    SUBSET_END=$(date +%s)
    echo "[$(ts)] $subset done — $((AFTER - BEFORE)) new files ($AFTER total), $((SUBSET_END - SUBSET_START))s"
done

touch $LOG_PATH/preprocess_h200.done
echo
echo "[$(ts)] === ALL preprocess done ==="
df -h / | tail -1
