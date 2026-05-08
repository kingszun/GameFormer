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

exec "$@"
