## scripts

GameFormer 재현용 자동화 script. 용도별로 folder 구분.

```
scripts/
├── README.md # this file
├── local/ # host docker compose smoke (3060 등)
│ ├── 01-download_womd.{py,sh} # WOMD shard download (gsutil ADC)
│ ├── 02-build_image.sh # docker image build
│ ├── 03-up.sh # container 기동
│ ├── 04-smoke_test.sh # CUDA + waymo proto 검증
│ ├── 05-open_loop_preprocess.sh # tfrecord → npz
│ ├── 06-open_loop_train.sh # open_loop 학습 (host docker compose)
│ └── 99-down.sh # container 종료
└── cloud/ # cloud (RunPod) 작업
    ├── preprocess/ # preprocessing chain (host → pCloud)
    │ ├── pod-preprocess-pcloud-batch.sh # 8 pod 분산 preprocess + pCloud upload
    │ ├── pod-preprocess-interaction-batch.sh
    │ ├── pod-preprocess-volume-direct.sh
    │ ├── pod-pipeline-h200-single.sh # single H200 pod chain (download wait + untar + preprocess)
    │ ├── preprocess_chain_v3.sh # validation chain
    │ └── cloud-multi-pod-{interaction,volume-direct}.sh # orchestrators
    ├── transfer/ # data transfer (rclone, pCloud public download)
    │ ├── pcloud-public-download.py # NEW: pCloud public link parallel download (auth 필요 X)
    │ ├── cloud-rclone-{download,upload}.sh # rclone (pCloud OAuth)
    │ └── cloud-tar-upload-cleanup.sh
    └── train/ # 학습 launcher (pod 안에서 실행)
        ├── pod-train-from-scratch-open-loop.sh # NEW: pCloud public + untar + train (auth X)
        ├── pod-train-from-scratch-interaction.sh # NEW: 동일 (interaction, DDP)
        ├── pod-train-open-loop.sh # rclone (auth) 버전 — chain after rclone done
        └── pod-train-interaction.sh # 동일 (interaction, DDP)
```

### local/ — 호스트 docker compose smoke

3060 12GB 등 host 에서 docker compose 로 1~2 shard 학습 path 동작 검증.

#### 사전 준비 (1회)

- `docker` (compose v2), `nvidia-container-toolkit`
- `google-cloud-cli` (`gsutil`, `gcloud`) — 01번 script 전제
- Waymo Open Dataset license + `gcloud auth login`
- `data` symlink:

```
DATASET_HOST_PATH=/mnt/e/datasets/womd
mkdir -p $DATASET_HOST_PATH
ln -s $DATASET_HOST_PATH ./data
cp .env.example .env
```

#### 실행 흐름 (smoke test)

```
WOMD_SUBSET=training_20s WOMD_SHARDS=2 bash scripts/local/01-download_womd.sh
bash scripts/local/02-build_image.sh
bash scripts/local/03-up.sh
bash scripts/local/04-smoke_test.sh
bash scripts/local/05-open_loop_preprocess.sh
BATCH_SIZE=8 EPOCHS=1 bash scripts/local/06-open_loop_train.sh
```

#### 산출물 위치 (local)

- raw tfrecord: `data/raw/${WOMD_SUBSET}/`
- preprocessed npz: `data/processed/open_loop/${SPLIT}/`
- training log/checkpoint: `open_loop_planning/training_log/${NAME}/`

### cloud/ — RunPod 학습 workflow

본격 학습 (full WOMD, H200/H100). data 가 이미 pCloud 에 preprocess + tar+split 으로 올라가 있다고 가정.

#### pCloud public share codes

| subset | code | files | size | path |
|---|---|---|---|---|
| open_loop/train_tar | `p3vctalK` | 45 | 323 GiB | `pcloud:06_Datasets/gameformer/processed/open_loop/train_tar/` |
| open_loop/valid_tar | `zaM7` | 19 | 147 GiB | `pcloud:.../open_loop/valid_tar/` |
| interaction/train_tar | `kt4` | 147 | 1135 GiB | `pcloud:.../interaction/train_tar/` |
| interaction/valid_tar | `SYpctalK` | 4 | 23 GiB | `pcloud:.../interaction/valid_tar/` |

short link: `http://u.pc.cd/<code>` (예: `http://u.pc.cd/kt4`)

#### 학습 시작 (pCloud public, auth X)

RunPod pod (H200 등) 에서 `git clone https://github.com/kingszun/GameFormer` 후:

```
# open_loop (1×H200)
nohup bash scripts/cloud/train/pod-train-from-scratch-open-loop.sh > /workspace/logs/launch.log 2>&1 &

# interaction (4×H200 DDP)
nohup bash scripts/cloud/train/pod-train-from-scratch-interaction.sh > /workspace/logs/launch.log 2>&1 &
```

기본 환경 변수 (override 가능):

| script | BATCH_SIZE | EPOCHS | LR | NPROC | WORKERS | NAME |
|---|---|---|---|---|---|---|
| open_loop | 128 | 20 | 2.83e-4 | 1 | 8 | op_full |
| interaction | 64/GPU | 30 | 2e-4 | 4 | 16 | ip_full |

paper baseline (batch 32 lr 1e-4) 대비 sqrt scaling 적용 (paper deviation acceptable.3).

각 script 의 chain step:
1. pCloud public download (transfers per default)
2. parallel untar (xargs -P 64)
3. cleanup tar dirs
4. launch training (background, PID 저장)

#### Helper: pcloud-public-download.py

pCloud public link 의 folder share download. auth 필요 X.

```
python3 scripts/cloud/transfer/pcloud-public-download.py <CODE> <DEST_DIR> [--parallel N]
```

- `<CODE>`: pCloud public link code (예: `kt4`)
- `<DEST_DIR>`: download 받을 directory
- `--parallel`: parallel curl 수 (default 8)
- idempotent: size 일치하면 skip
- 실패 시 retry 3 회 (exponential backoff)

API: `https://api.pcloud.com/showpublink` + `getpublinkdownload`.

#### Preprocess (data 가 없는 경우, multi-pod 분산)

핵심 pattern:

- `pod-preprocess-pcloud-batch.sh`: pod 안에서 8 pod 분산 preprocess + pCloud tar upload
- `pod-preprocess-interaction-batch.sh`: interaction (single threaded data_process.py)
- `pod-pipeline-h200-single.sh`: H200 single big pod chain
- `preprocess_chain_v3.sh`: validation 처리 chain

각 script 의 환경 변수는 script 내부 `# 환경 변수:` section 참조.

#### Transfer (rclone, pCloud OAuth)

기존 방식 — rclone.conf 의 pcloud remote 사용 (host scp 또는 wrapper):

- `cloud-rclone-download.sh` / `cloud-rclone-upload.sh`
- `cloud-tar-upload-cleanup.sh`: preprocess 결과 tar+split + upload + cleanup

### env var 우선순위

각 script 는 다음 순서로 설정값 결정.

1. command line 앞 env (예: `BATCH_SIZE=64 bash scripts/local/06-...sh`)
2. shell export 값
3. script 내부 default

local/ 의 `.env` 파일은 docker compose 가 host → container env 주입 시 사용 (script 자체 동작은 위 1, 2번).
