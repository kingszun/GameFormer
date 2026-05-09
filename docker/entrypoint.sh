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

# KAK-56: auto-tail /workspace/logs/$(hostname)/**/*.log (per-pod) to entrypoint stdout
# 같은 network volume 을 multi-pod mount 시 다른 pod 의 log 를 stream 하지 않도록 hostname prefix 격리.
# pod 안 모든 작업이 /workspace/logs/$(hostname)/ 안에 log 작성하도록 LOGS env 일관.
LOGS_DIR=/workspace/logs/$(hostname)
mkdir -p $LOGS_DIR
echo "===== ENTRYPOINT: log dir = $LOGS_DIR (hostname=$(hostname)) ====="
{
    declare -A TAILED
    while true; do
        while IFS= read -r f; do
            [ -f "$f" ] || continue
            if [ -z "${TAILED[$f]:-}" ]; then
                TAILED[$f]=1
                ( echo "===== TAIL START: $f ====="; tail -n 0 -F "$f" 2>/dev/null ) &
            fi
        done < <(find $LOGS_DIR -type f -name '*.log' 2>/dev/null)
        sleep 5
    done
} &

exec "$@"
