## 08 - Storage 구조 표준

GameFormer project 의 data 가 보관되는 위치 + lifecycle + 사용 패턴 표준화.

| storage | 위치 | persistent | 사용 |
|---|---|---|---|
| host symlink | `data -> /mnt/e/datasets/womd` | host disk | local 3060 학습 (KAK-9 smoke) |
| RunPod network volume | `svnweu0of5` (US-IL-1, 1024 GB) | ✓ | cloud preprocess + 학습 입력 |
| RunPod container disk | pod local (RTX 3090 = 1204 GB, CPU pod = 20 GB) | pod 까지 | preprocess 중간 결과, prefetch raw, log |
| 외부 cloud (pCloud) | `06_Datasets/gameformer/` (US, 7.5 TB total) | ✓ RunPod 와 무관 | backup, region 자유 학습, lock-in 회피 |

storage 정책 일반은 [`~/.claude/rules/kck-runpod.md` 의 Storage section](#) 참조. 이 doc 는 GameFormer 의 구체 구조.

---

### 1. pCloud (외부 cloud, 표준)

학습 데이터의 source of truth. region 무관 — H100/H200 가용 region 자유 선택 가능.

```
pcloud:06_Datasets/gameformer/
├── raw/                                   # WOMD v1.2.1 raw, 532 GB
│   ├── validation/                        # 150 shard, 39 GB
│   ├── training_20s/                      # 344 shard, 30 GB
│   ├── validation_interactive/            # 150 shard, 38 GB
│   └── training/                          # 1000 shard, 425 GB
├── processed/                             # preprocess 결과
│   ├── open_loop/
│   │   ├── valid/                         # 158 GB, ~308K npz
│   │   └── train/                         # ~330 GB, ~650K npz
│   └── interaction/
│       ├── valid/                         # ~30 GB, 86,958 npz (KAK-44)
│       └── train/                         # ~870 GB, ~1.7M npz
└── output/                                # 학습 결과
    ├── open_loop_full/
    │   ├── checkpoints/                   # predictor_<epoch>.pth
    │   └── logs/                          # train_<run>.log, tensorboard
    └── interaction_full/
        ├── checkpoints/
        └── logs/
```

#### lifecycle

- raw — 한 번 upload 후 변경 없음 (read-only)
- processed — preprocess 끝난 후 upload, 학습에서 read
- output — 학습 중 checkpoint 매 epoch upload, 학습 끝나면 final upload

#### upload 패턴

```
scripts/cloud-rclone-upload.sh <pod_id> /workspace/data/processed/open_loop/valid 06_Datasets/gameformer/processed/open_loop/valid
```

자세한 사용은 `scripts/cloud-rclone-upload.sh` 의 header.

---

### 2. RunPod network volume (`svnweu0of5`, US-IL-1, 1024 GB)

cloud preprocess 의 input source + sync target.

```
svnweu0of5:/                               # MooseFS mount /workspace
├── raw/                                   # WOMD raw (multi-pod 공유 source)
│   ├── validation/
│   ├── training_20s/
│   ├── validation_interactive/
│   └── training/
├── processed/                             # chain v3 의 sync_bg 결과 (pagination bug 로 부분 fail 가능)
│   ├── open_loop/{valid,train}/
│   └── interaction/{valid,train}/
├── logs/                                  # 작업 log (chain v3 heartbeat 등)
└── GameFormer/                            # 코드 (git clone, 또는 host scp)
```

#### lifecycle

- raw — 1회 download (CPU pod KAK-38 으로 cloud 에서 받음, 또는 host 에서 scp)
- processed — chain v3 또는 별도 sync 결과. **aws s3 sync pagination bug 회피 필요** (rclone 권장)
- 현재 1024 GB / 사용 656 GB (raw 530 + processed 일부) — interaction 대비 부족 가능. pCloud sync 후 정리 필요

#### multi-pod 동시 mount

검증됨 (KAK-41, 26-05-09). 같은 region 의 여러 pod 가 동시 mount + read OK. write path 분리 시 contention 작음.

---

### 3. RunPod container disk (pod local)

pod 별 ephemeral disk (resize destructive — 큰 disk 필요 시 새 pod create).

#### RTX 3090 6 GPU (chain v3, EU-CZ-1)

container disk 1204 GB (max). svnweu0of5 mount X (다른 region) — cross-region S3 endpoint 사용.

```
/workspace/                                # container disk
├── data/
│   ├── raw/                               # cross-region S3 endpoint 으로 prefetch
│   │   ├── validation/      (39 GB)
│   │   ├── training_20s/    (30 GB)
│   │   ├── validation_interactive/  (38 GB)
│   │   └── training/        (425 GB)     # chain v3 의 prefetch
│   ├── processed/                         # preprocess 직접 write
│   │   ├── open_loop/{valid,train}/
│   │   └── interaction/train/
│   └── logs/                              # chain v3 + heartbeat + preprocess log
├── GameFormer/                            # git clone
└── preprocess_chain_v3.sh                 # chain script (host 에서 scp)
```

##### disk 사용 예측 + 정리 정책

| 시점 | raw | processed | total | 여유 |
|---|---|---|---|---|
| chain v3 시작 | 530 GB | 0 | 530 GB | 674 GB |
| valid 완료 | 530 | 158 | 688 GB | 516 |
| train 완료 | 530 | 158+330 | **1018 GB** | 186 |
| interaction 시작 | 530 | 488 | 1018 | 186 (interaction 870 GB 못 들어감) |

→ **train 완료 후 valid + train 을 pCloud 로 sync + container disk delete** 필요. 그 후 interaction 시작 가능.

#### CPU pod (US-IL-1, svnweu0of5 mount)

container disk 20 GB (CPU pod max). svnweu0of5 mount = `/workspace`.

```
/workspace/                                # = svnweu0of5 (위 2 참조)
/                                          # container disk 20 GB (rclone log, 임시 file)
└── /workspace/data/logs/rclone_*.log      # rclone background sync log
```

용도: svnweu0of5 의 raw / processed 를 pCloud 로 upload (chain v3 의 sync_bg 우회).

#### H100 / H200 학습 pod

container disk 200 ~ 500 GB (학습 input + output).

```
/workspace/                                # container disk
├── input/                                 # rclone sync from pCloud (학습 시작 전 1회)
│   └── processed/{open_loop,interaction}/{train,valid}/
├── output/                                # 학습 결과 (매 epoch checkpoint)
│   └── <run_name>/{checkpoints,logs}/
└── GameFormer/                            # git clone
```

##### 학습 시 sync 패턴

```
# 학습 시작 전 (pCloud → container disk)
rclone copy pcloud:06_Datasets/gameformer/processed/open_loop /workspace/input/processed/open_loop --transfers 32

# 학습 (container disk read, output write)

# 학습 끝 (container disk → pCloud)
rclone copy /workspace/output pcloud:06_Datasets/gameformer/output --transfers 32
```

mount 안 사용 — random IO (npz 1M+ file) 부적합. sync 후 local read 가 효율적.

---

### 4. host (`/home/krong/workspace/GameFormer`)

local 개발 + 3060 smoke + cloud setup 의 source.

```
/home/krong/workspace/GameFormer/
├── data -> /mnt/e/datasets/womd           # symlink → host external disk
│   ├── raw/                               # WOMD v1.2.1 (3060 smoke 용)
│   ├── processed/                         # 3060 smoke preprocess 결과
│   └── logs/                              # 3060 smoke log
├── docker/                                # Dockerfile, entrypoint
├── docs/                                  # design docs
├── scripts/                               # cloud-rclone-upload.sh, preprocess_chain_v3.sh 등
├── open_loop_planning/                    # data_process.py + train.py
├── interaction_prediction/                # data_process.py + train.py
├── model/                                 # GameFormer.py
├── utils/                                 # data_utils, train_utils
└── docker-compose.yml                     # 3060 smoke compose stack
```

#### host ↔ pod 자료 전달

- 코드: git push → pod 에서 git clone (또는 scp 변경 file)
- WOMD ADC: host ADC → pod scp (`~/.config/gcloud/application_default_credentials.json`)
- rclone config: host `~/.config/rclone/rclone.conf` → pod scp (`scripts/cloud-rclone-upload.sh` 가 자동)
- ssh key: host `~/.runpod/ssh/RunPod-Key-Go` (private), `.pub` (PUBLIC_KEY env 로 pod entrypoint 에서 authorized_keys 생성)

---

### 5. 데이터 흐름 (전체)

```
[Waymo GCS bucket]
        |
        | gsutil + ADC (host) 또는 CPU pod KAK-38
        v
[host /mnt/e/datasets/womd/raw]                    [svnweu0of5/raw (RunPod US-IL-1)]
        |                                                 |
        | (3060 smoke)                                    | (cloud preprocess input)
        v                                                 v
[host data_process.py]                            [chain v3 prefetch raw → container disk]
        |                                                 |
        v                                                 v
[host processed/]                                 [container disk processed/]
        |                                                 |
        | (검증)                                           | (rclone copy)
        v                                                 v
[3060 smoke train (KAK-9)]                        [pCloud processed/]  ←  source of truth
                                                          |
                                                          | (학습 시 rclone copy)
                                                          v
                                                  [학습 pod container disk input/]
                                                          |
                                                          v
                                                  [학습 pod output/]  →  pCloud output/
```

---

### 6. pCloud ↔ pod 연동 패턴

#### 6.1. setup (1회 / pod)

새 pod 생성 시:

```
# pod 안에서 (image 에 rclone v1.74 포함, 단 RunPod cache issue 로 v1.58 가능 — KAK-55)
which rclone || apt update && apt install -y rclone
mkdir -p /root/.config/rclone

# host 에서 (config scp)
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> ~/.config/rclone/rclone.conf root@<pod-ip>:/root/.config/rclone/

# pod 에서 검증
rclone listremotes      # pcloud: 보여야 함
rclone about pcloud:    # 7.488 TiB total 보여야 함
```

자동화: `scripts/cloud-rclone-upload.sh` 가 자동 진행 (config scp + 검증 포함).

#### 6.2. upload (pod → pCloud)

```
# 단순 (host 에서 wrapper 호출)
scripts/cloud-rclone-upload.sh <pod_id> /workspace/data/processed/X 06_Datasets/gameformer/processed/X

# 또는 pod 안 직접 (background + log)
nohup rclone copy /workspace/data/processed/X pcloud:06_Datasets/gameformer/processed/X \
    --transfers 32 --stats 60s \
    --log-file /workspace/data/logs/rclone_X.log --log-level INFO \
    > /workspace/data/logs/rclone_X.stdout 2>&1 &
```

진행 모니터링:
```
ssh <pod> 'tail -F /workspace/data/logs/rclone_X.log'
ssh <pod> 'rclone size pcloud:06_Datasets/gameformer/processed/X'
ssh <pod> 'ps -ef | grep rclone'
```

#### 6.3. download (pCloud → pod)

학습 시작 전 input prefetch:

```
# 단순 (host 에서 wrapper 호출 — 추후 cloud-rclone-download.sh 작성)

# 또는 pod 안 직접
mkdir -p /workspace/input
rclone copy pcloud:06_Datasets/gameformer/processed/open_loop /workspace/input/processed/open_loop \
    --transfers 32 --stats 60s --progress
```

#### 6.4. 검증 (size 일치)

upload/download 후 반드시 size 일치 확인:

```
# 양쪽 size 비교
ssh <pod> 'du -sh /workspace/data/processed/X'           # pod side
rclone size pcloud:06_Datasets/gameformer/processed/X    # pCloud side (host 또는 pod)

# 또는 file count 비교
ssh <pod> 'find /workspace/data/processed/X -name "*.npz" | wc -l'
rclone lsf pcloud:06_Datasets/gameformer/processed/X --recursive | grep "\.npz$" | wc -l
```

##### 검증 전 cleanup 금지 (안전 정책)

container disk 에서 sync 후 file delete 시 **반드시 size + file count 양쪽 일치 확인 후** delete. 안 그러면 데이터 손실 위험.

#### 6.5. 연동 시 함정

- **rclone mount 불가** — RunPod container 의 `/dev/fuse` device + SYS_ADMIN cap 미부여. user-level FUSE mount fail. **copy/sync 만 사용**.
- **pCloud rate limit** — 대용량 (>1 TB) sync 시 throttle 가능. 1 TB 이상은 transfers 줄여서 (16) sustained 안정성 확인.
- **chain script 안에서 aws s3 sync 금지** — pagination bug. rclone copy 로 대체.
- **소형 file 1M+ (npz)** — connection per file overhead 로 throughput 떨어질 수 있음. 그래도 mount 보다 효율적 (mount 는 random access fail).
- **RunPod image 의 rclone version mismatch** (KAK-55) — image build 의 v1.74 와 pod 의 v1.58 다를 수 있음. apt 의 v1.58 도 pCloud OAuth 호환 OK.

#### 6.6. 일반 사용 예 (KAK-41 학습 단계)

```
# 1. pCloud 에서 input prefetch
ssh <h100-pod> 'rclone copy pcloud:06_Datasets/gameformer/processed/open_loop /workspace/input --transfers 32 --progress'

# 2. 학습 (container disk read)
ssh <h100-pod> 'cd /workspace/GameFormer/open_loop_planning && python train.py --train_set /workspace/input/open_loop/train --valid_set /workspace/input/open_loop/valid --name full_b ...'

# 3. 학습 중 매 epoch checkpoint 자동 upload (train.py 후처리 또는 별도 watcher)
ssh <h100-pod> 'rclone copy /workspace/training_log pcloud:06_Datasets/gameformer/output/open_loop_full --transfers 16 --progress'

# 4. 학습 끝나면 pod destroy 가능 (모든 결과 pCloud)
runpodctl pod stop <h100-pod>
runpodctl pod remove <h100-pod>
```

---

### 7. 검증된 throughput

| 경로 | 측정값 | 비고 |
|---|---|---|
| pod ↔ pCloud (rclone copy, transfers 32) | 830 MB/s burst | 26-05-09, US-IL-1 → pCloud US |
| svnweu0of5 ↔ EU-CZ-1 pod (cross-region S3, parallel xargs) | 133 MB/s | 26-05-08 |
| svnweu0of5 ↔ US-CA-2 pod (cross-region S3, sync) | 28~39 MiB/s | 26-05-09 |
| Waymo GCS → host (gsutil) | ~50 MB/s | ADC + 인증 |
| Waymo GCS → CPU pod (KAK-38) | 110 MB/s | parallel |

---

### 8. 관련 문서

- `~/.claude/rules/kck-runpod.md` — RunPod platform 일반 (storage 정책, S3 endpoint 지원 region, rclone 사용 패턴)
- `docs/01-environment.md` — Docker / compose stack
- `docs/05-cloud_plan.md` — RunPod 진행 plan
- `docs/07-data_pipeline.md` — preprocess 의 데이터 흐름 + subset 별 상세
- `scripts/cloud-rclone-upload.sh` — pod → pCloud sync wrapper
- `scripts/preprocess_chain_v3.sh` — chain script (heartbeat + stall watchdog)
