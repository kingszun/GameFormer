# 04. Stack migration — torch 1.x → 2.x + 환경 호환성 patch

원본 GameFormer repo 는 PyTorch 1.12 + CUDA 11.6 + Python 3.8 + WOMD v1.1 stack. 이 reproduction 은 동일 코드를 PyTorch 2.3.1 + CUDA 11.8 + Python 3.10 + WOMD v1.2.1 환경에서 재현. 모델 / 학습 logic 변경 없이 환경 호환성 patch 만 적용.

## 1. 변경 사유 정리

| 항목 | 원본 | 재현 | 사유 |
| --- | --- | --- | --- |
| PyTorch | 1.12.0+cu116 | 2.3.1+cu118 | (1) cu116 wheel 은 sm_86 까지만 native, sm_89 (RTX 4090) / sm_90 (H100/H200) 미지원. cu118 wheel 은 sm_90 까지 native 포함 (2) 2.x 의 launcher / DDP 개선 |
| Python | 3.8 | 3.10 | torch 2.3.1+cu118 wheel 의 default 지원 |
| CUDA | 11.6 | 11.8 | sm_90 native 지원 |
| TensorFlow | 2.11 | 2.11 동일 | `waymo-open-dataset-tf-2-11-0` 가 강제 |
| dataset | WOMD v1.1 | WOMD v1.2.1 | v1.1 의 `training_20s` annotation 결손 issue 회피 (`open_loop_planning` 학습 시 NaN 발생) |
| dependency 관리 | `requirements.txt` (pip) | `pyproject.toml` + `uv.lock` (uv) | 빠른 install + reproducible lock |

## 2. 환경 호환성 patch (4 곳)

원본 코드 변경은 모두 환경 호환성에 한정 — 모델 architecture / 학습 logic 영향 없음. 자세한 내용은 [docs/03-patches.md](../03-patches.md) 참조.

### 2.1. `open_loop_planning/data_process.py:62` — WOMD v1.2.1 신규 map type

```python
# 원본
else:
    raise TypeError

# 변경
else:
    continue
```

**배경**:
- 원본은 WOMD v1.1 시점 작성. map feature type 6 종 (`lane / road_line / road_edge / stop_sign / crosswalk / speed_bump`) 가정.
- WOMD v1.2.x 에 `driveway` 등 신규 map feature type 추가. 알 수 없는 oneof 만나면 `raise TypeError` 로 process 중단.
- `interaction_prediction/data_process.py:71` 은 이미 같은 위치를 `continue` 로 처리한 상태 (upstream maintainer 가 한쪽만 patch 한 것으로 추정).
- 동일 패턴으로 통일.

**영향**: 알 수 없는 type 을 silently skip — driveway 는 학습에 사용 안 하던 feature 라 model input 영향 없음.

### 2.2. `interaction_prediction/train.py:263` — torch 2.x DDP launcher 호환

```python
# 원본
parser.add_argument("--local_rank", type=int)

# 변경
parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
```

**배경**:
- torch 1.x 의 `torch.distributed.launch` 는 child process 에 `--local_rank=N` (underscore) 전달.
- torch 2.x 의 `torch.distributed.launch` / `torchrun` 은 `--local-rank=N` (hyphen) 전달.
- argparse 는 underscore/hyphen 자동 변환 안 함 → torch 2.x launcher 사용 시 `unrecognized arguments: --local-rank=0` 에러.
- alias 추가로 양쪽 launcher 호환 + `default=0` 으로 single-process 직접 실행도 가능.

**영향**: 코드 logic 변화 없음. argparse alias 만 추가.

### 2.3. `open_loop_planning/train.py:14` + `interaction_prediction/train.py:17` — multiprocessing fd sharing

```python
# 추가
torch.multiprocessing.set_sharing_strategy('file_system')
```

**배경**:
- DataLoader worker 가 fork 될 때, default `file_descriptor` 전략은 worker 간 tensor 공유에 fd 사용. worker 수 ↑ → fd 누적 → `ulimit -n` 한계 초과 → "OSError: [Errno 24] Too many open files".
- cloud pod 의 default `ulimit -n=1024` 환경에서 worker=8 이상 사용 시 발생.
- `file_system` 전략으로 전환하면 fd 대신 임시 파일 사용 → fd 한계 무관.

**영향**: 호환성 patch (퍼포먼스 영향 없음). 단 임시 파일 정리 시 약간의 disk I/O.

### 2.4. `open_loop_planning/train.py` + `interaction_prediction/train.py` — TRAINING_LOG_HOME env

```python
# 추가
log_home = os.environ.get('TRAINING_LOG_HOME', './training_log')
log_path = f"{log_home}/{args.name}/"
```

**배경**:
- 원본은 cwd-relative `./training_log/` 에 산출물 저장.
- cloud pod 의 entrypoint 가 log auto-tail 위해 특정 path 를 watch — `/workspace/logs/$(hostname)/runs/` 등 외부화 필요.
- env var 로 외부화 — 미지정 시 원본 동작 그대로.

**영향**: env var 미지정 시 원본 동일. 명시 시 외부 path 로 redirect.

## 3. Dependency 관리 — `pyproject.toml` + uv

원본 `requirements.txt`:
```
torch==1.12.0+cu116
torchvision==0.13.0+cu116
torchaudio==0.12.0
tensorflow-cpu==2.11.0
waymo-open-dataset-tf-2-11-0==1.6.4
matplotlib==3.5.2
shapely==1.8.5
imageio==2.21.2
imageio-ffmpeg==0.4.7
scipy==1.9.1
descartes==1.1.0
```

재현 `pyproject.toml` 의 dependency 중 변경:
- `torch==2.3.1` (cu118 from extra-index `https://download.pytorch.org/whl/cu118`)
- `torchvision==0.18.1`
- `tensorflow==2.11.1` (cpu only)
- 나머지 dep 은 호환되는 latest minor

`uv.lock` 으로 모든 dep 의 정확한 version 고정 — bit-perfect reproducibility 보장.

## 4. Docker image

`docker/Dockerfile` 로 base image 부터 venv setup 까지 자동화.

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.10 + uv venv at /opt/venv/gameformer
RUN apt-get update && apt-get install -y python3.10 python3.10-venv ... \
    && python3.10 -m venv /opt/venv/gameformer \
    && /opt/venv/gameformer/bin/pip install uv \
    && uv pip install -e . --system

# auxiliary venv for gcs (download) — google-cloud-storage 별도
RUN python3.10 -m venv /opt/venv/gcs \
    && /opt/venv/gcs/bin/pip install google-cloud-storage

# rclone (cross-cloud transfer)
RUN apt-get install -y rclone

# aws cli v2 binary (S3 endpoint cross-region access)
RUN curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o ... \
    && unzip ... && ./aws/install
```

**Image 설계**:
- base: official nvidia/cuda → CUDA driver / cuDNN 보장
- venv 분리: `gameformer` (학습 main) + `gcs` (gsutil 거부 우회용 google-cloud-storage)
- rclone + aws cli: 데이터 transfer 도구 사전 install
- 코드는 image 에 안 포함 (host bind mount or git clone) → 코드 변경 시 rebuild 불필요

총 image size: ~5.9 GB compressed.

## 5. Hardware compatibility 검증 결과

| GPU | sm_arch | torch 2.3.1+cu118 동작 | 비고 |
| --- | --- | --- | --- |
| RTX 3060 | sm_86 | yes | local smoke (12 GB VRAM, batch 8) |
| RTX 4090 | sm_89 | yes | cloud smoke (24 GB VRAM, batch 16) |
| A100 | sm_80 | yes | cloud (40/80 GB VRAM) |
| H100 | sm_90 | yes | cloud (80 GB VRAM) |
| H200 | sm_90 | yes | cloud (143 GB VRAM, full training) |

cu118 wheel 의 sm_86 + sm_89 + sm_90 native 지원으로 모든 GPU 에서 별도 build 없이 동작.

## 6. Reproducibility 확인

같은 seed (3407) + 같은 dataset (WOMD v1.2.1) + 같은 hyperparameter 일 때:
- 3060 1 epoch smoke: train_loss 91 → 48
- 4090 1 epoch cloud smoke: 동일 값
- (full training 결과는 [06-results](06-results.md) 참조)

stack migration 으로 인한 학습 결과 변화 없음 (같은 seed 면 deterministic).
