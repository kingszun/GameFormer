## scripts

GameFormer 재현용 자동화 script. 번호 순서대로 1회씩 실행하면 환경 구성부터 학습까지 진행된다.

### 실행 환경 분리

- 호스트 실행: `01-download_womd.sh`, `02-build_image.sh`, `03-up.sh`, `99-down.sh` — host에서 직접
- 컨테이너 실행: `04-smoke_test.sh`, `05-open_loop_preprocess.sh`, `06-open_loop_train.sh` — host에서 호출하지만 내부적으로 `docker compose exec`로 컨테이너에 진입

### 사전 준비 (1회)

호스트에 다음이 갖춰져 있어야 한다.

- `docker` (compose v2 포함), `nvidia-container-toolkit`
- `google-cloud-cli` (`gsutil`, `gcloud`) — 01번 script 전제
- Waymo Open Dataset license 동의 (https://waymo.com/open/licensing/) + `gcloud auth login` 으로 동일 계정 인증
- `data` symlink가 dataset 저장 host 경로를 가리켜야 함. 예시:

```
DATASET_HOST_PATH=/mnt/e/datasets/womd
mkdir -p $DATASET_HOST_PATH
ln -s $DATASET_HOST_PATH ./data
```

- `.env` 생성:

```
cp .env.example .env
```

### script 목록과 실행 순서

| 순서 | script | 실행 위치 | 목적 | 주요 env var |
| --- | --- | --- | --- | --- |
| 01 | `01-download_womd.sh` | host | WOMD scenario tfrecord 일부 shard download | `WOMD_VERSION`, `WOMD_SUBSET`, `WOMD_SHARDS`, `WOMD_DEST` |
| 02 | `02-build_image.sh` | host | docker image build (`gameformer:cu118-py310-torch2.3.1`) | `IMAGE_REPO`, `IMAGE_TAG` (.env) |
| 03 | `03-up.sh` | host | container 기동 (`sleep infinity`로 상주) | `DOCKER_GPUS`, `USER_UID/GID`, `DATASET_HOME` (.env) |
| 04 | `04-smoke_test.sh` | container | torch CUDA + GPU matmul + waymo proto import 검증 | - |
| 05 | `05-open_loop_preprocess.sh` | container | tfrecord -> .npz preprocess | `WOMD_SUBSET`, `SPLIT` |
| 06 | `06-open_loop_train.sh` | container | open_loop_planning 학습 | `BATCH_SIZE`, `EPOCHS`, `LR`, `LEVELS`, `NAME`, `TRAIN_SPLIT`, `VALID_SPLIT` |
| 99 | `99-down.sh` | host | container 종료 (image/volume 보존) | - |

### 일반 실행 흐름 (smoke test)

3060 12GB 기준 최소 검증 흐름. WOMD validation 1~2 shard로 학습 path 동작만 확인.

```
WOMD_SUBSET=training_20s WOMD_SHARDS=2 bash scripts/01-download_womd.sh
bash scripts/02-build_image.sh
bash scripts/03-up.sh
bash scripts/04-smoke_test.sh
bash scripts/05-open_loop_preprocess.sh
BATCH_SIZE=8 EPOCHS=1 bash scripts/06-open_loop_train.sh
```

### H200 서버 이식

같은 image, 같은 compose, 같은 script. 차이는 `.env` 와 batch/epoch override만.

```
git clone <repo> && cd GameFormer
ln -s /path/to/dataset/on/h200/host ./data
cp .env.example .env
bash scripts/02-build_image.sh
bash scripts/03-up.sh
WOMD_SHARDS=all bash scripts/01-download_womd.sh
bash scripts/05-open_loop_preprocess.sh
BATCH_SIZE=64 EPOCHS=20 NAME=run01 bash scripts/06-open_loop_train.sh
```

### env var 우선순위

각 script는 다음 순서로 설정값을 결정한다.

1. command line 앞에 붙인 env (예: `BATCH_SIZE=32 bash scripts/06-...sh`)
2. shell session에 export된 값
3. script 내부 default

`.env` 파일은 docker compose가 host -> container env 주입 시 사용. script 자체의 동작 (batch_size 등)은 `.env`로 덮이지 않으므로 위 1, 2번 방식 사용.

### 산출물 위치

- raw tfrecord: `data/raw/${WOMD_SUBSET}/`
- preprocessed npz: `data/processed/open_loop/${SPLIT}/`
- training log/checkpoint: `open_loop_planning/training_log/${NAME}/`

### 자주 쓰는 보조 command

container shell 진입:

```
docker compose exec gameformer bash
```

GPU 사용량 모니터:

```
docker compose exec gameformer nvidia-smi
```

container 재시작 (코드만 수정한 경우):

```
docker compose restart
```

image 재빌드 (Dockerfile 수정 시):

```
bash scripts/02-build_image.sh --no-cache
```
