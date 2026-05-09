#!/bin/bash
# 6 pod 에 4 subset preprocess 분산 deploy (volume direct write).
#
# 각 pod 에 4 subset 의 shard_range 균등 분담 → sequential preprocess (volume mount 에 직접 write).
# pCloud upload 없음. 학습 pod 가 same volume mount 로 직접 read.
#
# 사용:
#   scripts/cloud-multi-pod-volume-direct.sh <pod_id_1> ... <pod_id_N>
#
# 환경 변수:
#   N_TRAIN_INT      training shard 수 (default 1000)
#   N_TRAIN_OL       training_20s shard 수 (default 344)
#   N_VALID_INT      validation_interactive shard 수 (default 150)
#   N_VALID_OL       validation shard 수 (default 150)
#   WORKERS          per pod (default 28)
#   GIT_BRANCH       git checkout (default main)

set -e

POD_IDS=("$@")
N=${#POD_IDS[@]}
[ "$N" -lt 1 ] && { echo "Usage: $0 <pod_id_1> ..."; exit 1; }

N_TRAIN_INT=${N_TRAIN_INT:-1000}
N_TRAIN_OL=${N_TRAIN_OL:-344}
N_VALID_INT=${N_VALID_INT:-150}
N_VALID_OL=${N_VALID_OL:-150}
WORKERS=${WORKERS:-28}
GIT_BRANCH=${GIT_BRANCH:-main}

[ -f $HOME/.runpod/ssh/RunPod-Key-Go ] || { echo "RunPod ssh key not found"; exit 1; }

ts() { date -u +'%H:%M:%S'; }

# shard 균등 분할 함수
range_for() {
    local total=$1; local idx=$2
    local per=$(( (total + N - 1) / N ))
    local s=$(( idx * per ))
    local e=$(( (idx + 1) * per ))
    [ $e -gt $total ] && e=$total
    echo "$s:$e"
}

echo "[$(ts)] N=$N pods, workers=$WORKERS"
echo "[$(ts)] subset shard counts: train_int=$N_TRAIN_INT train_ol=$N_TRAIN_OL valid_int=$N_VALID_INT valid_ol=$N_VALID_OL"
echo

for i in $(seq 0 $((N - 1))); do
    POD_ID=${POD_IDS[$i]}
    R_TRAIN_INT=$(range_for $N_TRAIN_INT $i)
    R_TRAIN_OL=$(range_for $N_TRAIN_OL $i)
    R_VALID_INT=$(range_for $N_VALID_INT $i)
    R_VALID_OL=$(range_for $N_VALID_OL $i)

    echo "[$(ts)] === pod $i: $POD_ID ==="
    echo "  train_int=$R_TRAIN_INT train_ol=$R_TRAIN_OL valid_int=$R_VALID_INT valid_ol=$R_VALID_OL"

    INFO=$(runpodctl pod get $POD_ID -o json)
    SSH_IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('ip',''))")
    SSH_PORT=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('port',''))")
    [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ] && { echo "  SKIP — pod ssh info 없음"; continue; }

    SSH_OPTS="-i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p $SSH_PORT"

    # 1. kill 기존 batch + git pull (새 script 받음)
    ssh $SSH_OPTS root@$SSH_IP "
pkill -9 -f pod-preprocess-interaction-batch 2>/dev/null || true
pkill -9 -f 'python data_process' 2>/dev/null || true
pkill -9 -f 'rclone copy' 2>/dev/null || true
sleep 2
if [ -d /workspace/GameFormer/.git ]; then
    cd /workspace/GameFormer && git fetch origin && git checkout $GIT_BRANCH && git pull origin $GIT_BRANCH 2>&1 | tail -2
else
    rm -rf /workspace/GameFormer
    git clone https://github.com/kingszun/GameFormer /workspace/GameFormer 2>&1 | tail -2
fi
"

    # 2. background launch (SUBSETS env override 가능)
    ssh $SSH_OPTS root@$SSH_IP "
nohup env \
    SHARD_TRAIN_INT='$R_TRAIN_INT' \
    SHARD_TRAIN_OL='$R_TRAIN_OL' \
    SHARD_VALID_INT='$R_VALID_INT' \
    SHARD_VALID_OL='$R_VALID_OL' \
    WORKERS=$WORKERS \
    ${SUBSETS:+SUBSETS=\"$SUBSETS\"} \
    bash /workspace/GameFormer/scripts/pod-preprocess-volume-direct.sh \
    > \$LOG_PATH/volume_direct_pod${i}.log 2>&1 &
PID=\$!
echo \"  PID: \$PID, LOG: \$LOG_PATH/volume_direct_pod${i}.log\"
sleep 3
ps -p \$PID > /dev/null 2>&1 && echo '  alive ✓' || echo '  DIED!'
"
    echo
done

echo "[$(ts)] === all $N pods restarted ==="
echo
echo "monitor (각 pod 의 hostname 안):"
for i in $(seq 0 $((N - 1))); do
    POD_ID=${POD_IDS[$i]}
    INFO=$(runpodctl pod get $POD_ID -o json 2>/dev/null)
    SSH_IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('ip',''))" 2>/dev/null)
    SSH_PORT=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('port',''))" 2>/dev/null)
    echo "  pod $i ($POD_ID): ssh -i ~/.runpod/ssh/RunPod-Key-Go -p $SSH_PORT root@$SSH_IP 'tail -F /workspace/logs/\$(hostname)/volume_direct_pod${i}.log'"
done
