FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# =============================================================================
# Design (KAK-16):
# - base: nvidia/cuda 11.8 cudnn8 runtime ubuntu22.04 (compute backend only)
# - python + dep: uv-managed CPython 3.10 in /opt/uv-python (public dir
#   so compose uid 1000 can traverse; /root is 0700 by default)
# - venv: /opt/venv/gameformer (isolated, no system-site-packages, torch in venv)
# - PATH: venv/bin first → python/pip auto-resolve to venv binary
#
# User design (intentional dual-mode):
# - USER root (default). Same image runs in two scenarios:
#     compose (host)  → user 1000 override (host file ownership match)
#     RunPod pod      → root (sshd, /root/.ssh/authorized_keys, /workspace mount)
# - ENTRYPOINT root branch (`if id -u == 0`): PUBLIC_KEY → authorized_keys
#   + service ssh start. compose uid 1000 path skips the branch automatically.
# =============================================================================
USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX" \
    CMAKE_CUDA_ARCHITECTURES="86;90" \
    UV_PROJECT_ENVIRONMENT=/opt/venv/gameformer \
    UV_PYTHON=3.10 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/gameformer/bin:/usr/local/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        apt-transport-https \
        libgl1 \
        libglib2.0-0 \
        openssh-server \
        iproute2 \
    && mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /workspace/GameFormer

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
