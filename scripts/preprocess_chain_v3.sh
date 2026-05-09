#!/bin/bash
set -e
ulimit -n 65536

LOGS=/workspace/logs
mkdir -p $LOGS
S3_OPTS="--region us-il-1 --endpoint-url https://s3api-us-il-1.runpod.io --profile runpod"
S3_BUCKET=svnweu0of5

HB_LOG=$LOGS/chain_v3_heartbeat.log
STAGE_FILE=$LOGS/chain_v3_stage
SYNC_TRACK=$LOGS/chain_v3_sync_pids
WORKERS_OPEN_LOOP=90
WORKERS_INTERACTION=90
STALL_THRESHOLD_S=300
STALL_ABORT_S=600

: > $HB_LOG
: > $STAGE_FILE
: > $SYNC_TRACK

set_stage() {
    echo "$1|$2|$3|$(date +%s)" > $STAGE_FILE
}

ts() { date -u +'%H:%M:%S'; }

abort() {
    local msg=$1
    echo "[$(ts)] ABORT: $msg" | tee -a $LOGS/chain.error
    [ -f $STAGE_FILE ] && cat $STAGE_FILE >> $LOGS/chain.error
    pkill -9 -P $$ 2>/dev/null || true
    exit 1
}

trap 'rc=$?; [ $rc -ne 0 ] && abort "trap exit rc=$rc"' EXIT

heartbeat_loop() {
    local last_count=0
    local last_change_ts=$(date +%s)
    while true; do
        sleep 60
        [ -f $STAGE_FILE ] || continue
        local stage=$(awk -F'|' '{print $1}' $STAGE_FILE)
        local outdir=$(awk -F'|' '{print $2}' $STAGE_FILE)
        local pid=$(awk -F'|' '{print $3}' $STAGE_FILE)

        local outpath=/workspace/data/processed/$outdir
        local count=0
        local mtime_iso='-'
        local latest_int=0
        if [ -d "$outpath" ]; then
            count=$(find $outpath -name "*.npz" 2>/dev/null | wc -l)
            local latest=$(find $outpath -name "*.npz" -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
            if [ -n "$latest" ]; then
                mtime_iso=$(date -u -d "@${latest%.*}" +'%H:%M:%S')
                latest_int=${latest%.*}
            fi
        fi
        local now=$(date +%s)
        if [ "$latest_int" -gt "$last_change_ts" ]; then
            last_change_ts=$latest_int
        fi
        local stale=$((now - last_change_ts))
        local delta=$((count - last_count))
        local worker=0
        if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
            worker=$(pgrep -P $pid 2>/dev/null | wc -l)
        fi
        echo "[$(ts)] stage=$stage outdir=$outdir pid=$pid file=$count delta=${delta}/min mtime=$mtime_iso stale=${stale}s worker=$worker" >> $HB_LOG
        last_count=$count

        if [ -n "$pid" ] && [ "$count" -gt 0 ] && [ "$delta" -eq 0 ] && [ "$stale" -gt "$STALL_THRESHOLD_S" ]; then
            echo "[$(ts)] STALL DETECTED stage=$stage pid=$pid stale=${stale}s — sending SIGUSR1 stack dump" >> $HB_LOG
            kill -USR1 $pid 2>/dev/null || true
            for cpid in $(pgrep -P $pid 2>/dev/null); do
                kill -USR1 $cpid 2>/dev/null || true
            done
            sleep 5
            if [ "$stale" -gt "$STALL_ABORT_S" ]; then
                echo "[$(ts)] STALL ABORT stage=$stage stale=${stale}s — killing pid $pid" >> $HB_LOG
                pkill -9 -P $pid 2>/dev/null || true
                kill -9 $pid 2>/dev/null || true
                abort "stage=$stage stalled ${stale}s, pid $pid killed"
            fi
        fi
    done
}

wait_download_complete() {
    local subset=$1
    local dest=/workspace/data/raw/$subset
    set_stage "download_wait_$subset" "raw/$subset" ""
    mkdir -p $dest
    aws s3 ls s3://$S3_BUCKET/raw/$subset/ $S3_OPTS | awk '{print $4}' | grep -v "^$" > $LOGS/list_$subset.txt
    local listed=$(wc -l < $LOGS/list_$subset.txt)
    while pgrep -af "aws s3 cp.*raw/$subset/" > /dev/null; do
        echo "[$(ts)] $subset cp in progress: $(ls $dest 2>/dev/null | wc -l)/$listed"
        sleep 30
    done
    local got=$(ls $dest 2>/dev/null | wc -l)
    if [ "$got" = "$listed" ]; then
        echo "[$(ts)] $subset complete ($got/$listed)"
    else
        echo "[$(ts)] $subset incomplete: $got/$listed — fill missing"
        cat $LOGS/list_$subset.txt | xargs -P 32 -I {} bash -c "[ -f $dest/{} ] || aws s3 cp s3://$S3_BUCKET/raw/$subset/{} $dest/{} $S3_OPTS --only-show-errors" 2>$LOGS/dl_fill_$subset.err || true
        got=$(ls $dest 2>/dev/null | wc -l)
        [ "$got" = "$listed" ] || abort "$subset still incomplete: $got/$listed"
    fi
}

run_preprocess() {
    local cwd=$1
    local subset=$2
    local outdir_full=$3
    local stage_outdir=$4
    local workers=$5
    cd /workspace/GameFormer/$cwd
    echo "[$(ts)] === preprocess $cwd subset=$subset outdir=$outdir_full ($workers worker) ==="
    python data_process.py --load_path /workspace/data/raw/$subset --save_path /workspace/data/processed/$outdir_full --use_multiprocessing --processes $workers > $LOGS/preprocess_$(basename $outdir_full).log 2>&1 &
    local pid=$!
    set_stage "preprocess_$(basename $outdir_full)" "$stage_outdir" $pid
    if ! wait $pid; then
        abort "preprocess $outdir_full failed"
    fi
    local got=$(find /workspace/data/processed/$outdir_full -name "*.npz" 2>/dev/null | wc -l)
    echo "[$(ts)] $outdir_full file count: $got"
    [ "$got" -gt 100 ] || abort "$outdir_full output too small: $got"
    set_stage "complete_preprocess_$(basename $outdir_full)" "$stage_outdir" ""
}

sync_bg() {
    local local_dir=$1
    local s3_path=$2
    local tag=$3
    nohup aws s3 sync $local_dir/ s3://$S3_BUCKET/$s3_path $S3_OPTS --only-show-errors > $LOGS/sync_$tag.log 2>&1 &
    local pid=$!
    echo "$tag|$pid" >> $SYNC_TRACK
    echo "[$(ts)] sync_bg $tag PID $pid"
}

wait_syncs() {
    local label=$1
    echo "[$(ts)] waiting all sync_bg pids ($label)"
    while read line; do
        local tag=$(echo $line | awk -F'|' '{print $1}')
        local pid=$(echo $line | awk -F'|' '{print $2}')
        if ps -p $pid > /dev/null 2>&1; then
            echo "[$(ts)] waiting sync $tag PID $pid"
            wait $pid && echo "[$(ts)] sync $tag DONE" || abort "sync $tag failed"
        fi
    done < $SYNC_TRACK
    : > $SYNC_TRACK
}

echo "[$(ts)] === chain v3 start (worker_open_loop=$WORKERS_OPEN_LOOP worker_interaction=$WORKERS_INTERACTION stall_threshold=${STALL_THRESHOLD_S}s) ==="

heartbeat_loop &
HB_PID=$!
echo "[$(ts)] heartbeat watchdog PID $HB_PID"

wait_download_complete validation
run_preprocess open_loop_planning validation open_loop/valid open_loop/valid $WORKERS_OPEN_LOOP
sync_bg /workspace/data/processed/open_loop/valid processed/open_loop/valid valid

wait_download_complete training_20s
run_preprocess open_loop_planning training_20s open_loop/train open_loop/train $WORKERS_OPEN_LOOP
sync_bg /workspace/data/processed/open_loop/train processed/open_loop/train train

wait_syncs "open_loop"
touch $LOGS/openloop_preprocess.done
echo "[$(ts)] === openloop_preprocess.done marker ==="

wait_download_complete training
run_preprocess interaction_prediction training interaction/train interaction/train $WORKERS_INTERACTION
sync_bg /workspace/data/processed/interaction/train processed/interaction/train interaction_train

wait_syncs "interaction"
touch $LOGS/interaction_preprocess.done
echo "[$(ts)] === interaction_preprocess.done marker ==="

set_stage "all_done" "" ""
kill $HB_PID 2>/dev/null || true
trap - EXIT
echo "[$(ts)] === ALL preprocess + sync done ==="
