#!/usr/bin/env bash
set -e

if [ "$(id -u)" = "0" ]; then
    if [ -n "$PUBLIC_KEY" ]; then
        mkdir -p /root/.ssh
        chmod 700 /root/.ssh
        printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi
    # ensure sshd host keys (idempotent), then start sshd as background daemon
    ssh-keygen -A >/dev/null 2>&1 || true
    /usr/sbin/sshd 2>/dev/null || true
fi

# KAK-56: per-pod log isolation
# multi-pod 가 같은 network volume mount 시 log 충돌 회피 위해 hostname prefix.
# LOG_PATH env 를 /etc/environment 에 추가 — ssh login shell + bash 모두 inherit.
# 모든 script 가 ${LOG_PATH} 사용 → entrypoint 한 곳 변경 시 일관 적용.
LOG_PATH=/workspace/logs/$(hostname)
mkdir -p $LOG_PATH
# remove old LOG_PATH line (idempotent), then append current
sed -i '/^LOG_PATH=/d' /etc/environment 2>/dev/null || true
echo "LOG_PATH=$LOG_PATH" >> /etc/environment
export LOG_PATH
echo "===== ENTRYPOINT: LOG_PATH = $LOG_PATH (hostname=$(hostname)) ====="
{
    declare -A TAILED
    while true; do
        while IFS= read -r f; do
            [ -f "$f" ] || continue
            if [ -z "${TAILED[$f]:-}" ]; then
                TAILED[$f]=1
                ( echo "===== TAIL START: $f ====="; tail -n 0 -F "$f" 2>/dev/null ) &
            fi
        done < <(find $LOG_PATH -type f -name '*.log' 2>/dev/null)
        sleep 5
    done
} &

exec "$@"
