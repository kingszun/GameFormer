#!/bin/bash
# 6 pod (또는 N pod) 에 interaction preprocess 분산 deploy.
#
# 동작:
#   1. 각 pod 의 SSH info 자동 추출
#   2. 1000 shard 를 N pod 로 균등 분할 (SHARD_START:SHARD_END)
#   3. 각 pod 에:
#      - rclone install (없으면) + config scp
#      - batch script + data_process.py shard_range patch 가 있는 코드 git pull
#      - background batch script 실행 (pod 별 PCLOUD_DST prefix 격리)
#   4. 모든 pod PID + log path 보고 — 사용자가 monitor
#
# 사용:
#   scripts/cloud-multi-pod-interaction.sh <pod_id_1> <pod_id_2> ... <pod_id_N>
#
# 환경 변수:
#   TOTAL_SHARDS    1000 (default — raw/training shard 수)
#   PCLOUD_DST_BASE 06_Datasets/gameformer/processed/interaction/train_tar (default)
#   BATCH_SHARDS    pod 안 batch 수 (default 50)
#   WORKERS         multiprocessing pool size per pod (default 28, 32 vCPU 의 90%)
#   GIT_BRANCH      pod 의 git checkout (default main)
#
# 예:
#   scripts/cloud-multi-pod-interaction.sh la3n5h3xm7701t 8puwluaqfkj5al jwyqmxeb9lp9ff j15appjm2ucgiv nclyuwiwn7iao3 neznkyihs72wjf

set -e

POD_IDS=("$@")
N=${#POD_IDS[@]}
[ "$N" -lt 1 ] && { echo "Usage: $0 <pod_id_1> [pod_id_2] ..."; exit 1; }

TOTAL_SHARDS=${TOTAL_SHARDS:-1000}
PCLOUD_DST_BASE=${PCLOUD_DST_BASE:-06_Datasets/gameformer/processed/interaction/train_tar}
BATCH_SHARDS=${BATCH_SHARDS:-50}
WORKERS=${WORKERS:-28}
GIT_BRANCH=${GIT_BRANCH:-main}

[ -f $HOME/.config/rclone/rclone.conf ] || { echo "host rclone config not found"; exit 1; }
[ -f $HOME/.runpod/ssh/RunPod-Key-Go ] || { echo "RunPod ssh key not found"; exit 1; }

ts() { date -u +'%H:%M:%S'; }

# shard 균등 분할
SHARDS_PER_POD=$(( (TOTAL_SHARDS + N - 1) / N ))
echo "[$(ts)] N=$N pods, total_shards=$TOTAL_SHARDS, shards_per_pod=$SHARDS_PER_POD, workers=$WORKERS, batch=$BATCH_SHARDS"
echo

for i in $(seq 0 $((N - 1))); do
    POD_ID=${POD_IDS[$i]}
    SHARD_START=$((i * SHARDS_PER_POD))
    SHARD_END=$(( (i + 1) * SHARDS_PER_POD ))
    [ $SHARD_END -gt $TOTAL_SHARDS ] && SHARD_END=$TOTAL_SHARDS
    PCLOUD_DST="${PCLOUD_DST_BASE}/pod${i}"

    echo "[$(ts)] === pod $i: $POD_ID (shards ${SHARD_START}:${SHARD_END}) ==="

    INFO=$(runpodctl pod get $POD_ID -o json)
    SSH_IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('ip',''))")
    SSH_PORT=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('port',''))")
    [ -z "$SSH_IP" ] || [ -z "$SSH_PORT" ] && { echo "  SKIP — pod $POD_ID ssh info 없음"; continue; }
    echo "  ssh: $SSH_IP:$SSH_PORT, pcloud_dst=$PCLOUD_DST"

    SSH_OPTS="-i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p $SSH_PORT"
    SCP_OPTS="-i $HOME/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -P $SSH_PORT"

    # 1. rclone install + config scp + git clone/pull (.git 까지 확인 — 불완전 clone 회피)
    ssh $SSH_OPTS root@$SSH_IP "
which rclone > /dev/null 2>&1 || (apt update && apt install -y rclone)
mkdir -p /root/.config/rclone /workspace/logs /root/staging
if [ -d /workspace/GameFormer/.git ]; then
    cd /workspace/GameFormer && git fetch origin && git checkout $GIT_BRANCH && git pull origin $GIT_BRANCH 2>&1 | tail -3
else
    rm -rf /workspace/GameFormer
    git clone https://github.com/kingszun/GameFormer /workspace/GameFormer 2>&1 | tail -3
fi
"
    scp $SCP_OPTS $HOME/.config/rclone/rclone.conf root@$SSH_IP:/root/.config/rclone/rclone.conf

    # 2. background batch script run
    ssh $SSH_OPTS root@$SSH_IP "
nohup env \
    SHARD_START=$SHARD_START \
    SHARD_END=$SHARD_END \
    BATCH_SHARDS=$BATCH_SHARDS \
    WORKERS=$WORKERS \
    PCLOUD_DST='$PCLOUD_DST' \
    bash /workspace/GameFormer/scripts/pod-preprocess-interaction-batch.sh \
    > /workspace/logs/interaction_batch_pod${i}.log 2>&1 &
BATCH_PID=\$!
echo \"  batch script PID: \$BATCH_PID\"
sleep 3
ps -p \$BATCH_PID > /dev/null 2>&1 && echo '  alive ✓' || echo '  DIED!'
"
    echo
done

echo "[$(ts)] === all $N pods deployed ==="
echo
echo "monitor:"
for i in $(seq 0 $((N - 1))); do
    POD_ID=${POD_IDS[$i]}
    INFO=$(runpodctl pod get $POD_ID -o json 2>/dev/null)
    SSH_IP=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('ip',''))" 2>/dev/null)
    SSH_PORT=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh',{}).get('port',''))" 2>/dev/null)
    echo "  pod $i ($POD_ID): ssh -i ~/.runpod/ssh/RunPod-Key-Go -p $SSH_PORT root@$SSH_IP 'tail -F /workspace/logs/interaction_batch_pod${i}.log'"
done
