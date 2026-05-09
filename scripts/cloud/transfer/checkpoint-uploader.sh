#!/bin/bash
# Checkpoint uploader: 학습 진행 중 새 .pth 가 생기면 pCloud 에 upload (학습 영향 없음)
#
# 동작:
#   - polling 으로 CKPT_DIR 내 .pth 감시
#   - 새 file 발견 시 size stable 한지 확인 (5s 후 size 비교) — 학습이 still writing 인 경우 skip
#   - rclone copy 로 pCloud upload (idempotent — 이미 있으면 skip)
#   - upload 완료된 file 은 (name, size) tuple 로 dedup
#
# Usage:
#   nohup bash checkpoint-uploader.sh <CKPT_DIR> <PCLOUD_DST> [POLL_SEC] > <LOG> 2>&1 &
#
# Example:
#   nohup bash checkpoint-uploader.sh \
#       /workspace/logs/runs/op_full_b128 \
#       pcloud:06_Datasets/gameformer/checkpoints/op_full_b128 \
#       60 > /workspace/logs/checkpoint_uploader.log 2>&1 &

CKPT_DIR=${1:?usage: $0 CKPT_DIR PCLOUD_DST [POLL_SEC]}
PCLOUD_DST=${2:?usage: $0 CKPT_DIR PCLOUD_DST [POLL_SEC]}
POLL=${3:-60}

ts() { date -u +'%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === checkpoint-uploader start ==="
echo "[$(ts)] CKPT_DIR=$CKPT_DIR"
echo "[$(ts)] PCLOUD_DST=$PCLOUD_DST"
echo "[$(ts)] POLL=${POLL}s"

declare -A UPLOADED

while true; do
    sleep $POLL
    [ -d "$CKPT_DIR" ] || continue

    for f in $CKPT_DIR/*.pth; do
        [ -f "$f" ] || continue
        name=$(basename $f)

        # size stable 인지 확인 (5s 후 변화 없으면 write 끝)
        s1=$(stat -c%s "$f" 2>/dev/null)
        sleep 5
        s2=$(stat -c%s "$f" 2>/dev/null)
        if [ "$s1" != "$s2" ]; then
            echo "[$(ts)] $name still writing (size $s1 → $s2), skip"
            continue
        fi

        # 이미 같은 (name, size) upload 됐으면 skip
        if [ "${UPLOADED[$name]}" = "$s2" ]; then
            continue
        fi

        # upload
        SIZE_MB=$((s2 / 1024 / 1024))
        echo "[$(ts)] upload $name (${SIZE_MB} MiB)"
        START=$(date +%s)
        if rclone copy "$f" $PCLOUD_DST \
                --transfers 4 \
                --multi-thread-streams 4 \
                --buffer-size 64M \
                --stats 30s 2>&1; then
            UPLOADED[$name]=$s2
            ELAPSED=$(($(date +%s) - START))
            echo "[$(ts)] uploaded $name in ${ELAPSED}s"
        else
            echo "[$(ts)] FAIL upload $name (will retry next cycle)"
        fi
    done
done
