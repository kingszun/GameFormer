#!/bin/bash
# pod 안에서 1 subset preprocess + pCloud upload (batch + cleanup).
#
# Why:
# - 각 pod 가 자기 shard range 만 pCloud raw 에서 download (network volume mount 없음)
# - container disk 256 GB 안에서 batch 단위 (raw stay + processed/tar batch cleanup) 안전 처리
# - pCloud 의 processed 결과 (학습 pod 에서 download)
#
# 환경 변수:
#   SUBSET                 open_loop_train|open_loop_valid|interaction_train|interaction_valid (required)
#   SHARD_RANGE            sorted file count slice (s:e). 예: 0:167 (required)
#   WORKERS                multiprocessing.Pool processes (default 28)
#   BATCH_SHARDS           1 batch 의 shard 수 (default 50)
#   CHUNK_SIZE             tar split chunk size (default 8G)
#   TRANSFERS              rclone upload --transfers (default 8)
#   DOWNLOAD_TRANSFERS     rclone download --transfers (default 8)
#   RAW_BASE               raw download root (default /root/raw)
#   STAGING_BASE           container disk staging (default /root/staging)
#   TAR_BASE               tar chunk root (default /root/tar)
#   PCLOUD_RAW_BASE        pCloud raw base (default 06_Datasets/gameformer/raw)
#   PCLOUD_DST_BASE        pCloud processed base (default 06_Datasets/gameformer/processed)
#   LOG_PATH               log dir (default /workspace/logs/$(hostname))

set -e
ulimit -n 65536

WORKERS=${WORKERS:-28}
SUBSET=${SUBSET:?required: open_loop_train|open_loop_valid|interaction_train|interaction_valid}
SHARD_RANGE=${SHARD_RANGE:?required: s:e (sorted file count slice)}
BATCH_SHARDS=${BATCH_SHARDS:-50}
CHUNK_SIZE=${CHUNK_SIZE:-8G}
TRANSFERS=${TRANSFERS:-8}
DOWNLOAD_TRANSFERS=${DOWNLOAD_TRANSFERS:-8}
RAW_BASE=${RAW_BASE:-/root/raw}
STAGING_BASE=${STAGING_BASE:-/root/staging}
TAR_BASE=${TAR_BASE:-/root/tar}
PCLOUD_RAW_BASE=${PCLOUD_RAW_BASE:-06_Datasets/gameformer/raw}
PCLOUD_DST_BASE=${PCLOUD_DST_BASE:-06_Datasets/gameformer/processed}
LOG_PATH=${LOG_PATH:-/workspace/logs/$(hostname)}

mkdir -p $LOG_PATH

ts() { date -u +'%H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $LOG_PATH/preprocess_pcloud_${SUBSET}.error; exit 1; }

# subset config: name | raw_subdir | preprocess_dir (cwd) | save_subdir
declare -A CFG
CFG[open_loop_train]="training_20s|open_loop_planning|open_loop/train"
CFG[open_loop_valid]="validation|open_loop_planning|open_loop/valid"
CFG[interaction_train]="training|interaction_prediction|interaction/train"
CFG[interaction_valid]="validation_interactive|interaction_prediction|interaction/valid"

cfg=${CFG[$SUBSET]:-}
[ -z "$cfg" ] && abort "unknown subset: $SUBSET"
IFS='|' read -r raw_sub cwd save_sub <<< "$cfg"

LOAD_PATH=$RAW_BASE/$raw_sub
SAVE_PATH=$STAGING_BASE/$save_sub
TAR_DIR_BASE=$TAR_BASE/${SUBSET}
PCLOUD_RAW=$PCLOUD_RAW_BASE/$raw_sub
PCLOUD_DST=$PCLOUD_DST_BASE/${save_sub}_tar

IFS=':' read -r S_START S_END <<< "$SHARD_RANGE"
[ -z "$S_END" ] && abort "SHARD_RANGE format invalid: $SHARD_RANGE (expected s:e)"
TOTAL=$((S_END - S_START))
[ "$TOTAL" -gt 0 ] || abort "SHARD_RANGE range empty: $SHARD_RANGE"

mkdir -p $LOAD_PATH $SAVE_PATH $TAR_DIR_BASE
which rclone > /dev/null || abort "rclone 없음 — apt install rclone"
[ -f /root/.config/rclone/rclone.conf ] || abort "rclone config 없음 — host scp 필요"
rclone listremotes | grep -q '^pcloud:' || abort "pcloud remote 없음"

echo "[$(ts)] === preprocess + pCloud upload start ==="
echo "[$(ts)] subset=$SUBSET range=$SHARD_RANGE total=$TOTAL workers=$WORKERS batch=$BATCH_SHARDS"
echo "[$(ts)] load=$LOAD_PATH save=$SAVE_PATH tar=$TAR_DIR_BASE"
echo "[$(ts)] raw_src=pcloud:$PCLOUD_RAW dst=pcloud:$PCLOUD_DST"

# === STEP 1: download raw shard range (pcloud → container disk) ===
echo
echo "[$(ts)] === STEP 1: list pcloud raw + download shard range ==="
LIST_FILE=$LOG_PATH/raw_files_${SUBSET}.txt
rclone lsf pcloud:$PCLOUD_RAW | sort | sed -n "$((S_START+1)),${S_END}p" > $LIST_FILE
LISTED=$(wc -l < $LIST_FILE)
[ "$LISTED" = "$TOTAL" ] || abort "list mismatch: expected $TOTAL got $LISTED (pcloud:$PCLOUD_RAW 의 파일 수가 부족하거나 SHARD_RANGE 가 범위 밖)"
echo "[$(ts)] selected $LISTED files (first: $(head -1 $LIST_FILE), last: $(tail -1 $LIST_FILE))"

DOWNLOAD_LOG=$LOG_PATH/rclone_download_${SUBSET}.log
DL_START=$(date +%s)
rclone copy pcloud:$PCLOUD_RAW $LOAD_PATH \
    --files-from $LIST_FILE \
    --transfers $DOWNLOAD_TRANSFERS \
    --multi-thread-streams 4 \
    --buffer-size 64M \
    --stats 30s \
    --log-file $DOWNLOAD_LOG --log-level INFO || abort "raw download failed (log: $DOWNLOAD_LOG)"
GOT=$(ls $LOAD_PATH | wc -l)
[ "$GOT" = "$TOTAL" ] || abort "download incomplete: $GOT/$TOTAL (log: $DOWNLOAD_LOG)"
RAW_SIZE=$(du -sh $LOAD_PATH | awk '{print $1}')
DL_END=$(date +%s)
echo "[$(ts)] download done — $GOT files, size $RAW_SIZE, $((DL_END - DL_START))s"
df -h / | tail -1

# === STEP 2: batch loop (preprocess → tar+split → upload + verify → cleanup) ===
echo
echo "[$(ts)] === STEP 2: batch loop (batch_shards=$BATCH_SHARDS) ==="

BATCH_IDX=0
for ((bs=0; bs < TOTAL; bs += BATCH_SHARDS)); do
    be=$((bs + BATCH_SHARDS))
    [ $be -gt $TOTAL ] && be=$TOTAL
    BATCH_IDX=$((BATCH_IDX + 1))
    GLOBAL_BS=$((S_START + bs))
    GLOBAL_BE_INCL=$((S_START + be - 1))
    BATCH_TAG="${SUBSET}_b$(printf '%04d' $GLOBAL_BS)_$(printf '%04d' $GLOBAL_BE_INCL)"
    BATCH_START=$(date +%s)
    echo
    echo "[$(ts)] -- batch $BATCH_IDX: local shards $bs:$be (global $GLOBAL_BS-$GLOBAL_BE_INCL inclusive) --"

    # 2.1 preprocess (container disk) + heartbeat (30s 간격 진척 main log 에)
    PREPROC_LOG=$LOG_PATH/preprocess_${BATCH_TAG}.log
    cd /workspace/GameFormer/$cwd
    BEFORE=$(find $SAVE_PATH -type f -name '*.npz' 2>/dev/null | wc -l)
    python data_process.py \
        --load_path $LOAD_PATH \
        --save_path $SAVE_PATH \
        --use_multiprocessing \
        --processes $WORKERS \
        --shard_range $bs:$be \
        > $PREPROC_LOG 2>&1 &
    PYTHON_PID=$!
    PRE_HEARTBEAT_FC=$BEFORE
    HEARTBEAT_START=$(date +%s)
    while kill -0 $PYTHON_PID 2>/dev/null; do
        sleep 30
        kill -0 $PYTHON_PID 2>/dev/null || break
        NOW_FC=$(find $SAVE_PATH -type f -name '*.npz' 2>/dev/null | wc -l)
        DELTA=$((NOW_FC - PRE_HEARTBEAT_FC))
        ELAPSED=$(($(date +%s) - HEARTBEAT_START))
        ALIVE=$(pgrep -P $PYTHON_PID | wc -l)
        echo "[$(ts)] $BATCH_TAG progress — total $((NOW_FC - BEFORE)) files (last 30s +$DELTA, ${ELAPSED}s elapsed, alive worker $ALIVE)"
        PRE_HEARTBEAT_FC=$NOW_FC
    done
    wait $PYTHON_PID
    RET=$?
    [ $RET -eq 0 ] || abort "$BATCH_TAG preprocess failed exit=$RET (log: $PREPROC_LOG)"
    AFTER=$(find $SAVE_PATH -type f -name '*.npz' | wc -l)
    BATCH_FC=$((AFTER - BEFORE))
    [ "$BATCH_FC" -gt 0 ] || abort "$BATCH_TAG produced 0 files"
    STAGE_SIZE=$(du -sh $SAVE_PATH | awk '{print $1}')
    echo "[$(ts)] preprocess done — $BATCH_FC new files (staging $STAGE_SIZE)"

    # 2.2 tar+split (per-batch dir)
    TAR_DIR=$TAR_DIR_BASE/$BATCH_TAG
    mkdir -p $TAR_DIR
    cd $(dirname $SAVE_PATH)
    tar -cf - $(basename $SAVE_PATH) | split -b $CHUNK_SIZE - $TAR_DIR/${BATCH_TAG}.tar.part_ || abort "$BATCH_TAG tar+split failed"
    CHUNK_COUNT=$(ls $TAR_DIR | wc -l)
    TAR_SIZE=$(find $TAR_DIR -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f\n", s}')
    echo "[$(ts)] tar+split done — $CHUNK_COUNT chunks, $(numfmt --to=iec $TAR_SIZE)"

    # 2.3 upload + size verify (이 batch 의 chunk 만 매칭 — concurrent multi-pod 안전)
    UPLOAD_LOG=$LOG_PATH/rclone_upload_${BATCH_TAG}.log
    rclone copy $TAR_DIR pcloud:$PCLOUD_DST \
        --transfers $TRANSFERS \
        --multi-thread-streams 4 \
        --buffer-size 64M \
        --stats 30s \
        --log-file $UPLOAD_LOG --log-level INFO || abort "$BATCH_TAG upload failed (log: $UPLOAD_LOG)"
    DST_BATCH_SIZE=$(rclone size pcloud:$PCLOUD_DST --include "${BATCH_TAG}.tar.part_*" --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("bytes",0))' 2>/dev/null || echo 0)
    if [ "$DST_BATCH_SIZE" != "$TAR_SIZE" ]; then
        abort "$BATCH_TAG size MISMATCH — local_tar=$TAR_SIZE pcloud_batch=$DST_BATCH_SIZE (cleanup 안 함, 수동 확인 필요)"
    fi
    echo "[$(ts)] upload + verify ✓ — pcloud batch size $(numfmt --to=iec $DST_BATCH_SIZE)"

    # 2.4 cleanup batch (find -delete 로 ARG_MAX 회피 — 200k+ small file 안전)
    find $SAVE_PATH -mindepth 1 -delete
    rm -rf $TAR_DIR
    BATCH_END=$(date +%s)
    echo "[$(ts)] cleanup done — batch $BATCH_IDX total $((BATCH_END - BATCH_START))s"
    df -h / | tail -1
done

# === STEP 3: cleanup raw ===
echo
echo "[$(ts)] === STEP 3: cleanup raw download ==="
rm -rf $LOAD_PATH
df -h / | tail -1

touch $LOG_PATH/preprocess_${SUBSET}.done
echo
echo "[$(ts)] === ALL DONE — subset=$SUBSET range=$SHARD_RANGE ==="
