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

# KAK-56: auto-tail /workspace/logs/**/*.log (recursive) to entrypoint stdout
# RunPod web UI 의 Container log tab 에서 작업 진행 상황 확인 가능
# nested directory (예: runs/<name>/train.log) 도 watch
mkdir -p /workspace/logs
{
    declare -A TAILED
    while true; do
        while IFS= read -r f; do
            [ -f "$f" ] || continue
            if [ -z "${TAILED[$f]:-}" ]; then
                TAILED[$f]=1
                ( echo "===== TAIL START: $f ====="; tail -n 0 -F "$f" 2>/dev/null ) &
            fi
        done < <(find /workspace/logs -type f -name '*.log' 2>/dev/null)
        sleep 5
    done
} &

exec "$@"
