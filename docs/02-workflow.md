## 02 - workflow

scripts/README.md 와 함께 참조. 이 문서는 각 단계의 의도와 환경 의존성을 설명.

### 단계 분류

- 호스트 단독 (cloud 이전 단계): WOMD download, image build, container 기동
- 컨테이너 내부 (host에서 호출): smoke test, preprocess, train

### 1단계 — host 환경 점검

다음을 host에서 1회 확인:

- `docker --version`, `docker compose version` (Compose v2 필수)
- `nvidia-container-toolkit` 설치 + `docker info | grep nvidia`에 `nvidia` runtime 등록
- `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 로 GPU passthrough 확인
- `gcloud --version` (WOMD download용, host 실행)

### 2단계 — Waymo dataset 인증 + symlink

- https://waymo.com/open/licensing/ 에서 Waymo Open Dataset license 동의 (사람 단계)
- `gcloud auth login` (Waymo 등록한 동일 Google account, host에서 1회)
- dataset 저장 host 경로 결정 + symlink:

```
DATASET_HOST_PATH=/mnt/e/datasets/womd
mkdir -p $DATASET_HOST_PATH
ln -s $DATASET_HOST_PATH ./data
```

- `.env` 생성: `cp .env.example .env`

### 3단계 — WOMD shard download

`scripts/01-download_womd.sh`. host 실행, gsutil로 GCS bucket에서 shard 일부 받음.

env:
- `WOMD_VERSION` — default `1_2_1` (v1.2.1, v1.1은 `training_20s` annotation 결손 있음)
- `WOMD_SUBSET` — default `validation_interactive`. open_loop는 `training_20s`, interaction은 `training`
- `WOMD_SHARDS` — default `2`, `all` 가능

shard 크기 차이 — 동일 shard count여도 subset에 따라 datasize 다름:

| subset | 1 shard 평균 | 1000 shard 합계 (전체) | 용도 |
| --- | --- | --- | --- |
| training_20s | ~80 MB | ~80 GB | open_loop_planning |
| training | ~440 MB | ~440 GB | interaction_prediction |
| validation_interactive | ~50 MB | ~50 GB | inter eval |
| validation | ~80 MB | ~80 GB | open_loop eval |

### 4단계 — image build + container 기동

```
bash scripts/02-build_image.sh   # 5~10분 (TF/waymo install 부분이 무거움)
bash scripts/03-up.sh            # 즉시
```

container는 `sleep infinity` 로 상주. 이후 단계는 모두 `docker compose exec gameformer ...` 또는 4~6 script.

### 5단계 — smoke test

```
bash scripts/04-smoke_test.sh
```

확인 사항:
- torch 2.3.1, cu118, GPU device 인식
- GPU matmul 정상
- waymo proto import 정상

실패 시: nvidia-container-toolkit 또는 GPU passthrough 문제. host에서 `docker run --rm --gpus all ...` 부터 점검.

### 6단계 — preprocess

```
WOMD_SUBSET=training_20s bash scripts/05-open_loop_preprocess.sh
```

각 raw shard를 scene 단위로 parse → scene내 timestep slice 별로 `.npz` 저장.

소요 시간 (3060):
- training_20s 1 shard (~64 scene): ~8분 (multiprocessing 2 workers 기준)
- training 1 shard (~500 scene): ~9분 (interaction은 scene당 sample 수 적음)

### 7단계 — train

open_loop_planning (single GPU):
```
BATCH_SIZE=8 EPOCHS=1 bash scripts/06-open_loop_train.sh
```

interaction_prediction (DDP, 현재 script로 wrapping 안 됨):
```
docker compose exec gameformer bash -lc "
cd interaction_prediction && python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=28596 train.py \
    --batch_size=4 --workers=2 --training_epochs=1 \
    --train_set=/workspace/GameFormer/data/processed/interaction/train \
    --valid_set=/workspace/GameFormer/data/processed/interaction/train \
    --name=smoke
"
```

3060 12GB 기준 batch_size 안전치:
- open_loop: 8 (32까지도 fit 가능 추정, 미실측)
- interaction: 4 (model 무거움 + neighbor 수 많음)

### 산출물 위치

- raw: `data/raw/${WOMD_SUBSET}/*.tfrecord*`
- processed: `data/processed/{open_loop,interaction}/${SPLIT}/*.npz`
- training log: `{interaction_prediction,open_loop_planning}/training_log/${NAME}/`
  - `train.log` — text log
  - `train_log.csv` — epoch별 metric
  - `epochs_N.pth` 또는 `predictor_N_*.pth` — checkpoint

### container 정리

```
bash scripts/99-down.sh   # container 종료, image/volume 보존
docker compose down --rmi local -v   # 완전 정리 (image까지 삭제)
```
