#!/bin/bash
# pod 의 directory 를 tar+split (chunk) → pCloud upload → size 검증 → cleanup.
#
# 동작:
#   1. pod 안에서 src 를 tar -cf - | split -b CHUNK_SIZE 로 chunk 만들기 (별도 tar dir)
#   2. chunk 들을 pCloud 로 parallel upload (transfers, multi-thread-streams)
#   3. size 검증 (pod side total vs pCloud side total) — 일치 시에만 cleanup
#   4. 검증 통과 시 src + tar dir delete
#   5. 검증 실패 시 abort + 사용자 보고 (delete 안 함)
#
# 사용:
#   scripts/cloud-tar-upload-cleanup.sh <pod_id> <pod_src_path> <pcloud_dst_tar_path>
#
# 환경 변수:
#   CHUNK_SIZE          split 의 -b (default 8G)
#   TRANSFERS           rclone --transfers (default 8)
#   MULTI_STREAMS       rclone --multi-thread-streams (default 4)
#   BUFFER_SIZE         rclone --buffer-size (default 64M)
#   TAR_DIR             pod 의 tar chunk 저장 위치 (default = $(dirname $src)/tar_$(basename $src))
#   SKIP_CLEANUP=1      검증 통과해도 cleanup 안 함 (debug)
#
# 예:
#   scripts/cloud-tar-upload-cleanup.sh eehfdm47y7dn2a /workspace/data/processed/open_loop/valid 06_Datasets/gameformer/processed/open_loop/valid_tar

set -e
POD_ID=${1:?pod_id required}
SRC=${2:?pod src path required}
DST=${3:?pcloud dst tar path required}
CHUNK_SIZE=${CHUNK_SIZE:-8G}
TRANSFERS=${TRANSFERS:-8}
MULTI_STREAMS=${MULTI_STREAMS:-4}
BUFFER_SIZE=${BUFFER_SIZE:-64M}
SKIP_CLEANUP=${SKIP_CLEANUP:-0}

[ -f $HOME/.config/rclone/rclone.conf ] || { echo "host rclone config not found at ~/.config/rclone/rclone.conf"; exit 1; }
[ -f $HOME/.runpod/ssh/RunPod-Key-Go ] || { echo "RunPod ssh key not found"; exit 1; }

ts() { date -u +'%H:%M:%S'; }

echo "[$(ts)] pod info 조회"
INFO=$(runpodctl pod get $POD_ID -o json)
SSH_IP=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('ip',''))")
SSH_PORT=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('port',''))")
[ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ] && { echo "pod $POD_ID 의 ssh info 없음"; exit 1; }

SSH="ssh -i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_IP"
SCP="scp -i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -P $SSH_PORT"

TAG=$(basename $SRC)
TAR_DIR_DEFAULT=$(dirname $SRC)/tar_$TAG
TAR_DIR=${TAR_DIR:-$TAR_DIR_DEFAULT}
LOG_BASE=/workspace/logs/$(hostname)/tar_upload_${TAG}_$(date +%y%m%d_%H%M%S)

echo "[$(ts)] $POD_ID @ $SSH_IP:$SSH_PORT"
echo "[$(ts)] src=$SRC tar_dir=$TAR_DIR dst=$DST chunk=$CHUNK_SIZE"

echo "[$(ts)] rclone setup"
$SSH "
which rclone > /dev/null 2>&1 || (apt update && apt install -y rclone)
mkdir -p /root/.config/rclone /workspace/logs/\$(hostname) $TAR_DIR
"
$SCP $HOME/.config/rclone/rclone.conf root@$SSH_IP:/root/.config/rclone/rclone.conf

echo "[$(ts)] step 1/4: tar + split (chunk_size=$CHUNK_SIZE)"
$SSH "
test -d '$SRC' || { echo 'src dir 없음: $SRC'; exit 1; }
SRC_SIZE_BYTES=\$(du -sb '$SRC' | awk '{print \$1}')
SRC_FILE_COUNT=\$(find '$SRC' -type f | wc -l)
echo \"src: \$(numfmt --to=iec \$SRC_SIZE_BYTES) (\$SRC_FILE_COUNT files)\"

cd \$(dirname '$SRC')
TAR_LOG=${LOG_BASE}_tar.log
{ time tar -cf - \$(basename '$SRC') | split -b $CHUNK_SIZE - $TAR_DIR/${TAG}.tar.part_; } > \$TAR_LOG 2>&1 || { echo TAR_FAIL; cat \$TAR_LOG; exit 1; }
echo TAR_DONE
ls -lh $TAR_DIR | head -25
TAR_TOTAL=\$(du -sb $TAR_DIR | awk '{print \$1}')
echo \"tar: \$(numfmt --to=iec \$TAR_TOTAL) (\$(ls $TAR_DIR | wc -l) chunks)\"
"

echo "[$(ts)] step 2/4: parallel rclone copy (transfers=$TRANSFERS multi-streams=$MULTI_STREAMS)"
$SSH "
RCLONE_LOG=${LOG_BASE}_rclone.log
rclone copy $TAR_DIR pcloud:$DST --transfers $TRANSFERS --multi-thread-streams $MULTI_STREAMS --buffer-size $BUFFER_SIZE --stats 30s --log-file \$RCLONE_LOG --log-level INFO || { echo RCLONE_FAIL; tail -20 \$RCLONE_LOG; exit 1; }
echo RCLONE_DONE
"

echo "[$(ts)] step 3/4: size 검증"
SIZE_OK=$($SSH "
TAR_SIZE=\$(du -sb $TAR_DIR | awk '{print \$1}')
PCLOUD_SIZE=\$(rclone size pcloud:$DST 2>&1 | grep -oP 'Total size: \K[0-9]+' | head -1)
echo \"tar=\$(numfmt --to=iec \$TAR_SIZE) pcloud=\$(numfmt --to=iec \$PCLOUD_SIZE)\"
if [ \"\$TAR_SIZE\" = \"\$PCLOUD_SIZE\" ]; then echo VERIFIED; else echo MISMATCH; fi
")
echo "$SIZE_OK"
if ! echo "$SIZE_OK" | grep -q VERIFIED; then
    echo "[$(ts)] ABORT: size MISMATCH — cleanup 안 함. 수동 확인 필요"
    exit 1
fi

if [ "$SKIP_CLEANUP" = "1" ]; then
    echo "[$(ts)] step 4/4: SKIP_CLEANUP=1 — src + tar_dir 유지"
else
    echo "[$(ts)] step 4/4: cleanup (src + tar_dir delete)"
    $SSH "
    rm -rf '$SRC' '$TAR_DIR'
    df -h /workspace | tail -2
    echo CLEANUP_DONE
    "
fi

echo "[$(ts)] DONE — pcloud:$DST"
