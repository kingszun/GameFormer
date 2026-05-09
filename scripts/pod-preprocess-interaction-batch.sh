#!/bin/bash
# pod 안에서 interaction preprocess 를 batch 단위로 처리 (disk overflow 회피)
#
# 1000 shard / BATCH_SHARDS shard 단위 batch
# 각 batch: preprocess → tar+split → pCloud upload → size 검증 → cleanup → 다음 batch
#
# 사용:
#   bash scripts/pod-preprocess-interaction-batch.sh
#
# 환경 변수:
#   WORKERS         multiprocessing.Pool processes (default 90)
#   BATCH_SHARDS    1 batch 의 shard 수 (default 100)
#   CHUNK_SIZE      tar split chunk size (default 8G)
#   TRANSFERS       rclone --transfers (default 8)
#   LOAD_PATH       raw input dir (default /workspace/data/raw/training)
#   SAVE_PATH       processed output dir (default /workspace/data/processed/interaction/train)
#   PCLOUD_DST      pCloud dst path (default 06_Datasets/gameformer/processed/interaction/train_tar)
#
# prerequisites:
#   - data_process.py 의 --shard_range patch
#   - rclone install + ~/.config/rclone/rclone.conf (host 에서 scp)

set -e
ulimit -n 65536

WORKERS=${WORKERS:-28}
BATCH_SHARDS=${BATCH_SHARDS:-50}
CHUNK_SIZE=${CHUNK_SIZE:-8G}
TRANSFERS=${TRANSFERS:-8}
LOAD_PATH=${LOAD_PATH:-/workspace/raw/training}
SAVE_PATH=${SAVE_PATH:-/root/staging/interaction/train}
TAR_DIR_BASE=${TAR_DIR_BASE:-/root/staging/tar_interaction}
PCLOUD_DST=${PCLOUD_DST:-06_Datasets/gameformer/processed/interaction/train_tar}
LOGS=${LOGS:-/workspace/logs}
SHARD_START=${SHARD_START:-0}
SHARD_END=${SHARD_END:-}

ts() { date -u +'%H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $LOGS/interaction_batch.error; exit 1; }

mkdir -p $SAVE_PATH $LOGS $TAR_DIR_BASE

[ -d $LOAD_PATH ] || abort "load_path 없음: $LOAD_PATH"
which rclone > /dev/null 2>&1 || abort "rclone 없음 — apt install rclone"
[ -f /root/.config/rclone/rclone.conf ] || abort "rclone config 없음"
rclone listremotes | grep -q '^pcloud:' || abort "pcloud remote 없음"

TOTAL_SHARDS=$(ls $LOAD_PATH | wc -l)
[ -z "$SHARD_END" ] && SHARD_END=$TOTAL_SHARDS
echo "[$(ts)] === interaction batch start ==="
echo "[$(ts)] total_shards=$TOTAL_SHARDS shard_range=${SHARD_START}:${SHARD_END} batch_shards=$BATCH_SHARDS workers=$WORKERS chunk=$CHUNK_SIZE"
echo "[$(ts)] load=$LOAD_PATH save=$SAVE_PATH tar_base=$TAR_DIR_BASE dst=pcloud:$PCLOUD_DST"

for ((start=SHARD_START; start < SHARD_END; start += BATCH_SHARDS)); do
    end=$((start + BATCH_SHARDS))
    [ $end -gt $SHARD_END ] && end=$SHARD_END
    BATCH_TAG="batch_$(printf '%04d' $start)_$(printf '%04d' $end)"
    BATCH_START=$(date +%s)
    echo
    echo "[$(ts)] === $BATCH_TAG (shards $start:$end) ==="

    # 1. preprocess this batch
    PREPROC_LOG=$LOGS/preprocess_interaction_${BATCH_TAG}.log
    cd /workspace/GameFormer/interaction_prediction
    python data_process.py \
        --load_path $LOAD_PATH \
        --save_path $SAVE_PATH \
        --use_multiprocessing \
        --processes $WORKERS \
        --shard_range $start:$end \
        > $PREPROC_LOG 2>&1 || abort "preprocess $BATCH_TAG failed (log: $PREPROC_LOG)"
    BATCH_FILE_COUNT=$(find $SAVE_PATH -type f -name '*.npz' | wc -l)
    [ "$BATCH_FILE_COUNT" -gt 0 ] || abort "preprocess $BATCH_TAG made 0 files"
    echo "[$(ts)] preprocess done — $BATCH_FILE_COUNT files"

    # 2. tar + split (per-batch dir under TAR_DIR_BASE — multi-pod 격리)
    TAR_DIR=$TAR_DIR_BASE/$BATCH_TAG
    mkdir -p $TAR_DIR
    cd $(dirname $SAVE_PATH)
    tar -cf - $(basename $SAVE_PATH) | split -b $CHUNK_SIZE - $TAR_DIR/interaction_${BATCH_TAG}.tar.part_ || abort "tar+split $BATCH_TAG failed"
    CHUNK_COUNT=$(ls $TAR_DIR | wc -l)
    TAR_SIZE=$(find $TAR_DIR -type f -printf '%s\n' | awk '{s+=$1} END {printf "%.0f\n", s}')
    echo "[$(ts)] tar+split done — $CHUNK_COUNT chunks, $(numfmt --to=iec $TAR_SIZE)"

    # 3. parallel upload (pre/post diff verify, --json 으로 정확 byte)
    PRE_PCLOUD=$(rclone size pcloud:$PCLOUD_DST --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("bytes",0))' 2>/dev/null || echo 0)
    UPLOAD_LOG=$LOGS/rclone_interaction_${BATCH_TAG}.log
    rclone copy $TAR_DIR pcloud:$PCLOUD_DST \
        --transfers $TRANSFERS \
        --multi-thread-streams 4 \
        --buffer-size 64M \
        --stats 30s \
        --log-file $UPLOAD_LOG --log-level INFO || abort "rclone $BATCH_TAG failed"
    POST_PCLOUD=$(rclone size pcloud:$PCLOUD_DST --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("bytes",0))' 2>/dev/null || echo 0)
    DIFF=$((POST_PCLOUD - PRE_PCLOUD))
    echo "[$(ts)] upload done — pre=$(numfmt --to=iec $PRE_PCLOUD) post=$(numfmt --to=iec $POST_PCLOUD) diff=$(numfmt --to=iec $DIFF)"

    # 4. size 검증 (tar size == pcloud diff)
    if [ "$DIFF" != "$TAR_SIZE" ]; then
        abort "size MISMATCH $BATCH_TAG — tar=$TAR_SIZE pcloud_diff=$DIFF (cleanup 안 함)"
    fi
    echo "[$(ts)] verified ✓"

    # 5. cleanup (this batch 의 processed + tar dir)
    rm -rf $SAVE_PATH/*
    rm -rf $TAR_DIR
    BATCH_END=$(date +%s)
    echo "[$(ts)] cleanup done — $((BATCH_END - BATCH_START))s elapsed"
    df -h / | tail -1
done

touch $LOGS/interaction_preprocess.done
echo
echo "[$(ts)] === interaction preprocess + upload ALL DONE ==="
