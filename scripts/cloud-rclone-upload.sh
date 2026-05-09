#!/bin/bash
# pod 에서 rclone copy (background) 로 pcloud 에 upload.
#
# 사용:
#   scripts/cloud-rclone-upload.sh <pod_id> <pod_src_path> <pcloud_dst_path>
#
# 환경 변수:
#   TRANSFERS  rclone --transfers (default 32)
#   TAG        log file prefix (default = basename of src)
#
# 예:
#   scripts/cloud-rclone-upload.sh eehfdm47y7dn2a /workspace/data/processed/open_loop/valid 06_Datasets/gameformer/processed/open_loop/valid

set -e
POD_ID=${1:?pod_id required}
SRC=${2:?src path required}
DST=${3:?dst pcloud path required}
TRANSFERS=${TRANSFERS:-32}
TAG=${TAG:-$(basename $SRC)}

[ -f $HOME/.config/rclone/rclone.conf ] || { echo "host rclone config not found at ~/.config/rclone/rclone.conf"; exit 1; }
[ -f $HOME/.runpod/ssh/RunPod-Key-Go ] || { echo "RunPod ssh key not found at ~/.runpod/ssh/RunPod-Key-Go"; exit 1; }

ts() { date -u +'%H:%M:%S'; }

echo "[$(ts)] pod info 조회"
INFO=$(runpodctl pod get $POD_ID -o json)
SSH_IP=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('ip',''))")
SSH_PORT=$(echo "$INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('port',''))")
[ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ] && { echo "pod $POD_ID 의 ssh info 없음 (not ready)"; exit 1; }
echo "[$(ts)] $POD_ID @ $SSH_IP:$SSH_PORT"

SSH="ssh -i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_IP"
SCP="scp -i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -P $SSH_PORT"

echo "[$(ts)] rclone 확인 + config scp"
$SSH "
which rclone > /dev/null 2>&1 || (apt update && apt install -y rclone)
mkdir -p /root/.config/rclone /workspace/logs/\$(hostname)
"
$SCP $HOME/.config/rclone/rclone.conf root@$SSH_IP:/root/.config/rclone/rclone.conf
$SSH "rclone listremotes | grep -q '^pcloud:' || { echo 'pcloud remote 없음 — host config 확인'; exit 1; }"

LOG=/workspace/logs/$(hostname)/rclone_${TAG}_$(date +%y%m%d_%H%M%S).log
STDOUT=${LOG}.stdout

echo "[$(ts)] background rclone copy 시작 (transfers=$TRANSFERS)"
$SSH "
test -e '$SRC' || { echo 'src path 없음: $SRC'; exit 1; }
nohup rclone copy '$SRC' 'pcloud:$DST' --transfers $TRANSFERS --stats 60s --log-file $LOG --log-level INFO > $STDOUT 2>&1 &
RCLONE_PID=\$!
sleep 3
if ! ps -p \$RCLONE_PID > /dev/null 2>&1; then
    echo 'RCLONE DIED at startup! stdout:'
    cat $STDOUT
    exit 1
fi
echo \"PID=\$RCLONE_PID LOG=$LOG\"
"

echo
echo "[$(ts)] started — monitor:"
echo "  $SSH 'tail -F $LOG'"
echo "  $SSH 'rclone size pcloud:$DST'"
echo "  $SSH 'ps -p <PID>'"
