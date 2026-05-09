#!/bin/bash
# pod 안에서 4 subset preprocess (container disk write + rclone copy → volume + cleanup)
#
# Why:
# - volume (MooseFS) 의 small file write 가 3.7배 느림 (측정값 8.3 MB/s vs container 31 MB/s)
# - container disk 에 write → batch 단위 rclone copy (multi-thread) → cleanup
# - interaction_train (per pod ~145 GB) 는 batch 처리 필수 (container disk 128 GB 내 안전)
#
# 환경 변수:
#   WORKERS              multiprocessing pool (default 28)
#   STAGING_BASE         container disk staging (default /root/staging)
#   VOLUME_DST_BASE      volume destination root (default /workspace/processed)
#   RAW_BASE             raw root (default /workspace/raw)
#   SHARD_TRAIN_INT      training shard range (interaction_train, e.g. 0:167)
#   SHARD_TRAIN_OL       training_20s shard range (open_loop_train)
#   SHARD_VALID_INT      validation_interactive shard range (interaction_valid)
#   SHARD_VALID_OL       validation shard range (open_loop_valid)
#   BATCH_OL_VALID       open_loop_valid batch size (default = 전체)
#   BATCH_OL_TRAIN       open_loop_train batch size (default 30 — 안전 margin)
#   BATCH_INT_VALID      interaction_valid batch size (default = 전체)
#   BATCH_INT_TRAIN      interaction_train batch size (default 50 — disk overflow 회피)
#   SUBSETS              처리 순서 (default "open_loop_valid open_loop_train interaction_valid interaction_train")
#   RCLONE_TRANSFERS     rclone copy --transfers (default 32)

set -e
ulimit -n 65536

WORKERS=${WORKERS:-28}
LOG_PATH=${LOG_PATH:-/workspace/logs/$(hostname)}
STAGING_BASE=${STAGING_BASE:-/root/staging}
VOLUME_DST_BASE=${VOLUME_DST_BASE:-/workspace/processed}
RAW_BASE=${RAW_BASE:-/workspace/raw}
SUBSETS=${SUBSETS:-"open_loop_valid open_loop_train interaction_valid interaction_train"}
RCLONE_TRANSFERS=${RCLONE_TRANSFERS:-32}

mkdir -p $LOG_PATH $STAGING_BASE

ts() { date -u +'%H:%M:%S'; }
abort() { echo "[$(ts)] ABORT: $1" | tee -a $LOG_PATH/preprocess_volume.error; exit 1; }

# subset config: name | raw_subdir | preprocess_dir (cwd) | save_subdir | shard_range_env | batch_size_env | batch_default
declare -A CFG
CFG[interaction_train]="training|interaction_prediction|interaction/train|$SHARD_TRAIN_INT|${BATCH_INT_TRAIN:-50}"
CFG[open_loop_train]="training_20s|open_loop_planning|open_loop/train|$SHARD_TRAIN_OL|${BATCH_OL_TRAIN:-30}"
CFG[interaction_valid]="validation_interactive|interaction_prediction|interaction/valid|$SHARD_VALID_INT|${BATCH_INT_VALID:-9999}"
CFG[open_loop_valid]="validation|open_loop_planning|open_loop/valid|$SHARD_VALID_OL|${BATCH_OL_VALID:-9999}"

echo "[$(ts)] === preprocess (container write + rclone copy) start (workers=$WORKERS, hostname=$(hostname)) ==="
echo "[$(ts)] subsets: $SUBSETS"
echo "[$(ts)] staging=$STAGING_BASE volume_dst=$VOLUME_DST_BASE"

for subset in $SUBSETS; do
    cfg=${CFG[$subset]:-}
    [ -z "$cfg" ] && abort "unknown subset: $subset"
    IFS='|' read -r raw_sub cwd save_sub shard_range batch_size <<< "$cfg"

    LOAD_PATH=$RAW_BASE/$raw_sub
    STAGING_PATH=$STAGING_BASE/$save_sub
    VOLUME_DST=$VOLUME_DST_BASE/$save_sub

    if [ ! -d "$LOAD_PATH" ]; then
        echo "[$(ts)] SKIP $subset — $LOAD_PATH 없음"
        continue
    fi

    # parse shard_range (default 0:total)
    if [ -n "$shard_range" ]; then
        IFS=':' read -r s_start s_end <<< "$shard_range"
    else
        s_start=0
        s_end=$(ls $LOAD_PATH | wc -l)
    fi
    SHARDS_THIS_SUBSET=$((s_end - s_start))

    SUBSET_START_TS=$(date +%s)
    mkdir -p $STAGING_PATH $VOLUME_DST
    echo
    echo "[$(ts)] === $subset (shard_range=${s_start}:${s_end}, batch=$batch_size) ==="
    echo "[$(ts)] raw=$LOAD_PATH staging=$STAGING_PATH dst=$VOLUME_DST"

    cd /workspace/GameFormer/$cwd

    BATCH_IDX=0
    for ((bs=s_start; bs<s_end; bs+=batch_size)); do
        be=$((bs + batch_size))
        [ $be -gt $s_end ] && be=$s_end
        BATCH_TAG="${subset}_b$(printf '%04d' $bs)_$(printf '%04d' $be)"
        BATCH_START=$(date +%s)
        BATCH_IDX=$((BATCH_IDX + 1))
        echo
        echo "[$(ts)] -- batch $BATCH_IDX: shards $bs:$be --"

        # 1. preprocess to staging (container disk)
        PREPROC_LOG=$LOG_PATH/preprocess_${BATCH_TAG}.log
        BEFORE=$(find $STAGING_PATH -name '*.npz' 2>/dev/null | wc -l)
        python data_process.py \
            --load_path $LOAD_PATH \
            --save_path $STAGING_PATH \
            --use_multiprocessing \
            --processes $WORKERS \
            --shard_range $bs:$be \
            > $PREPROC_LOG 2>&1 || abort "$BATCH_TAG preprocess failed (log: $PREPROC_LOG)"
        AFTER=$(find $STAGING_PATH -name '*.npz' | wc -l)
        BATCH_FILE_COUNT=$((AFTER - BEFORE))
        STAGE_SIZE=$(du -sh $STAGING_PATH | awk '{print $1}')
        echo "[$(ts)] preprocess done — $BATCH_FILE_COUNT new files (staging size $STAGE_SIZE)"

        # 2. rclone copy staging → volume (multi-thread)
        RCLONE_LOG=$LOG_PATH/rclone_copy_${BATCH_TAG}.log
        RCLONE_START=$(date +%s)
        rclone copy $STAGING_PATH $VOLUME_DST \
            --transfers $RCLONE_TRANSFERS \
            --buffer-size 64M \
            --stats 30s \
            --log-file $RCLONE_LOG --log-level INFO || abort "$BATCH_TAG rclone copy failed"
        RCLONE_END=$(date +%s)
        echo "[$(ts)] rclone copy done — $((RCLONE_END - RCLONE_START))s"

        # 3. cleanup staging
        rm -rf $STAGING_PATH/*
        BATCH_END=$(date +%s)
        echo "[$(ts)] cleanup done — batch total $((BATCH_END - BATCH_START))s"
        df -h / | tail -1
    done

    SUBSET_TOTAL=$(find $VOLUME_DST -name '*.npz' 2>/dev/null | wc -l)
    SUBSET_END_TS=$(date +%s)
    echo "[$(ts)] === $subset complete — volume total $SUBSET_TOTAL files, $((SUBSET_END_TS - SUBSET_START_TS))s ==="
done

touch $LOG_PATH/preprocess_volume.done
echo
echo "[$(ts)] === ALL subsets done ==="
