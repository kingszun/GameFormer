## 01 - environment

### Dockerfile 개요

base: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` (KAK-16, 26-05-08)
- compute backend only — python / torch / dep 은 모두 venv 안
- 이전 stack `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime` 에서 변경. base 와 dep 분리, lock 으로 reproducibility 강화

추가 apt: `git`, `curl`, `ca-certificates`, `gnupg`, `apt-transport-https`, `libgl1`, `libglib2.0-0`, `openssh-server`, `iproute2`, `google-cloud-cli`
- `libgl1`, `libglib2.0-0` — matplotlib backend 의존성
- `openssh-server` — RunPod "SSH over exposed TCP" 모드 (scp 가능)
- `iproute2` — `ss` (sshd listen 검증)
- `google-cloud-cli` — `gsutil` 로 WOMD bucket 접근. ADC env 로 인증 (KAK-7)

python + 패키지 관리: `uv` 0.11.7 (`ghcr.io/astral-sh/uv:0.11.7` 에서 binary copy)
- python: uv-managed CPython 3.10 in `/opt/uv-python` (UV_PYTHON_INSTALL_DIR public dir → compose uid 1000 traversal 가능)
- venv: `/opt/venv/gameformer` (UV_PROJECT_ENVIRONMENT). isolated, no system-site-packages, torch 도 venv 안
- pyproject.toml + uv.lock 으로 dep 정의 + reproducible lock
- PATH 우선: `/opt/venv/gameformer/bin` → `python`/`pip` 자동 venv 사용

dep (pyproject.toml):
- `torch==2.3.1+cu118` (PyTorch index `https://download.pytorch.org/whl/cu118`)
- `tensorflow==2.11.0`, `waymo-open-dataset-tf-2-11-0`
- `numpy<1.24`, `protobuf<3.20` (TF 2.11 호환)
- `shapely`, `matplotlib`, `tqdm`, `pandas`, `scipy`

ENV: `TORCH_CUDA_ARCH_LIST=8.6;9.0+PTX`, `CMAKE_CUDA_ARCHITECTURES=86;90`
- 3060 (sm_86) + H200 (sm_90 + PTX forward-compat) 동시 cover
- pre-built wheel 사용 시 직접 효과는 없지만 미래에 custom op 추가 시 의도 명시

USER 설계 (intentional dual-mode):
- `USER root` (default). 같은 image 가 두 시나리오에서 동작
  - compose (host) → `user: 1000:1000` runtime override (host file ownership 일치)
  - RunPod pod → root (sshd, /root/.ssh/authorized_keys, /workspace/data mount)
- ENTRYPOINT root 분기 (`if id -u == 0`): PUBLIC_KEY → authorized_keys + sshd start. compose uid 1000 path 자동 skip

ENTRYPOINT 동작 (`docker/entrypoint.sh`):
- root 일 때만:
  - `PUBLIC_KEY` env 가 있으면 `/root/.ssh/authorized_keys` 에 write (mode 600). RunPod 가 console 등록 public key 를 이 env 로 inject
  - `ssh-keygen -A` (idempotent) + `/usr/sbin/sshd` 직접 호출 (nvidia/cuda base 는 `service` 명령 부재)
- compose `user: 1000:1000` → root 분기 자체 skip
- `CMD ["sleep", "infinity"]` — RunPod default startup 으로도 pod 유지. compose 는 `command:` override
- sshd config: `PermitRootLogin yes`, `PasswordAuthentication no` (key only)

### compose.yaml 핵심

| field | 값 | 의미 |
| --- | --- | --- |
| `image` | `${IMAGE_REPO}:${IMAGE_TAG}` | Docker Hub `kingszun/gameformer:cu118-py310-torch2.3.1` |
| `gpus` | `${DOCKER_GPUS:-all}` | 3060 host=1, H200 host=N 자동. 명시 override 시 env 한 줄 |
| `user` | `${USER_UID:-1000}:${USER_GID:-1000}` | mount된 file이 host에서 root 소유로 보이지 않도록 |
| `ipc` | `host` | DataLoader workers 공유 메모리 |
| `shm_size` | `${SHM_SIZE:-8gb}` | DDP NCCL 통신 + DataLoader queue |
| `ulimits.memlock` | -1 | NCCL pinned memory |
| `volumes` | `.:/workspace/GameFormer` + `${DATASET_HOME:-./data}:/workspace/GameFormer/data` | repo + dataset 분리 mount |
| `command` | `${COMPOSE_COMMAND:-sleep infinity}` | up -d 후 exec 패턴 |

env 주입:
- `TF_FORCE_GPU_ALLOW_GROWTH=true` — TF가 GPU memory 선점하지 않도록 (torch와 동거)
- `TF_CPP_MIN_LOG_LEVEL=2` — TF 노이즈 억제
- `HOME=/tmp` — non-root user의 ~/.cache 등 write 가능 위치

### data 경로 매핑

host symlink 패턴 (recogdrive 참조):
```
ln -s /mnt/e/datasets/womd ./data
```
compose가 `${DATASET_HOME:-./data}`를 host에서 resolve → 실제 storage 경로 bind-mount.

container 내부 view:
- raw: `/workspace/GameFormer/data/raw/${WOMD_SUBSET}/`
- processed: `/workspace/GameFormer/data/processed/{open_loop,interaction}/${SPLIT}/`

### 환경 변수 (.env)

| var | default | 비고 |
| --- | --- | --- |
| COMPOSE_PROJECT_NAME | gameformer | |
| IMAGE_REPO | gameformer | local 빌드 시 |
| IMAGE_TAG | cu118-py310-torch2.3.1 | semver 의미 없음, stack 식별자 |
| USER_UID, USER_GID | 1000, 1000 | host와 일치 권장 |
| DOCKER_GPUS | all | integer 또는 "all" |
| SHM_SIZE | 8gb | |
| TORCH_CUDA_ARCH_LIST | 8.6;9.0+PTX | 3060 + H200 cover |
| CMAKE_CUDA_ARCHITECTURES | 86;90 | |
| DATASET_HOME | ./data | symlink 권장 |
| COMPOSE_COMMAND | sleep infinity | up -d 후 exec |

### 검증된 hardware path

| GPU | sm | wheel binary | 동작 확인 |
| --- | --- | --- | --- |
| RTX 3060 12GB | 8.6 | cu118 wheel native | 직접 학습 통과 |
| H200 141GB | 9.0 | cu118 wheel native | 미테스트 (cloud 단계) |
| H100 80GB | 9.0 | cu118 wheel native | 미테스트 (cloud 단계) |

### Docker Hub registry

`docker.io/kingszun/gameformer:cu118-py310-torch2.3.1`
- 5949 MB compressed (17 layer), public
- pull: `docker pull kingszun/gameformer:cu118-py310-torch2.3.1`
- 코드는 image에 포함 안 됨 — cloud pod에서 별도 git clone 필요
- 내장: sshd, gcloud/gsutil, uv, venv (torch + TF + waymo SDK 등 모든 dep). pod 에서 추가 install 없이 즉시 사용 가능
