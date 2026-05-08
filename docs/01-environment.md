## 01 - environment

### Dockerfile 개요

base: `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime`
- python 3.10, torch 2.3.1+cu118, cuDNN 8.7
- runtime variant (4 GB) — devel 불필요 (custom CUDA extension 없음)

추가 apt: `git`, `curl`, `ca-certificates`, `libgl1`, `libglib2.0-0`
- `libgl1`, `libglib2.0-0` — matplotlib backend 의존성

추가 pip: `tensorflow==2.11.0`, `waymo-open-dataset-tf-2-11-0`, `shapely`, `matplotlib`, `tqdm`, `pandas`, `scipy`
- pin: `numpy<1.24`, `protobuf<3.20` — TF 2.11 호환

env: `TORCH_CUDA_ARCH_LIST=8.6;9.0+PTX`, `CMAKE_CUDA_ARCHITECTURES=86;90`
- 3060 (sm_86) + H200 (sm_90 + PTX forward-compat) 동시 cover
- pre-built wheel을 쓰므로 이 env가 실제 효과를 발휘하지는 않지만 (custom extension build 시점 의도 표시), cloud 이식성 의도 명시 + 미래에 custom op 추가 시 자동 적용.

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
- 4784 MB, public
- pull: `docker pull kingszun/gameformer:cu118-py310-torch2.3.1`
- 코드는 image에 포함 안 됨 — cloud pod에서 별도 git clone 필요
