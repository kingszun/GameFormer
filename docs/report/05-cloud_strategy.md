# 05. Cloud GPU 활용 전략

GameFormer 의 full WOMD 학습은 single 3060 으로는 불가능 (interaction prediction full set ≈ 1.7M sample, batch 16 × 30 epoch). cloud GPU 사용이 필수. 이 문서는 cloud platform 선택, region/비용 최적화, 데이터 전달 전략 등 운영 측면 결정을 정리.

## 1. Cloud platform 선택 — RunPod

선택 기준:
- 시간 단위 결제 (분 단위 정산)
- H100/H200 같은 최신 GPU 가용
- container 기반 (Docker image 직접 배포)
- 비교적 저렴

다른 옵션 (GCP / AWS / Lambda Cloud) 대비:
| 항목 | RunPod | GCP | AWS |
| --- | --- | --- | --- |
| H100 시간당 | $2.5~3 | $4~6 | $4~5 |
| H200 시간당 | $3.99 | (제한적) | (제한적) |
| 배포 단순도 | 높음 (container 이미지 push) | 보통 | 보통 |
| spot 가용 | 제한적 | yes | yes |

H100/H200 가용성 + 가격 으로 RunPod 선택.

## 2. Storage 선택 — pCloud (외부 cloud)

학습 데이터 (preprocess 결과 ~1.5 TB) 의 저장 위치:

| 옵션 | 비용 | region 자유도 | 단점 |
| --- | --- | --- | --- |
| RunPod network volume | $0.07/GB/월 | bound (region lock) | 같은 region pod 만 mount, region 별 GPU 가용성 불안정 |
| RunPod container disk | ~$0.10/GB/월 | pod local | pod destroy 시 손실, 다른 pod 와 share X |
| 외부 cloud (pCloud) | 사용자 plan | global | mount X (FUSE 권한 X), copy/sync 만 |

**pCloud 선택 이유**:
- 한 번 upload 하면 region 자유 — H200 가용 region 변경 시 lock-in 없음
- 사용자 보유 plan 활용 (RunPod 비용 절감)
- 학습 시작 시 `rclone copy` 또는 public link download 로 container disk 로 prefetch → random IO 학습은 local disk read

**제약**:
- mount 불가 → copy/sync 만. small file (npz 1MB) 의 cross-region 직접 transfer 가 latency 누적으로 느림 (10 MB/s)
- 해결: tar+split 으로 8 GB chunk 만든 후 parallel upload → 800+ MB/s burst 가능 (multi-thread cutoff 위)

## 3. 데이터 lifecycle

```
[Waymo GCS bucket] (us-central1)
        │
        │ gsutil + ADC (host) 또는 cloud CPU pod
        ▼
[host raw/{subset}] or [cloud network volume raw/{subset}]
        │
        │ data_process.py (multiprocessing)
        ▼
[processed/{subset}/{train,valid}/*.npz]   (~1.5 TB)
        │
        │ tar + split (8 GB chunk) + rclone copy
        ▼
[pCloud processed/.../{subset}_tar/*.part_aa..]   ← source of truth
        │
        │ rclone copy (학습 시작 시) or pCloud public link download
        ▼
[학습 pod container disk processed/]
        │
        │ DataLoader → batch
        ▼
[GameFormer model training]
        │
        │ 매 epoch checkpoint
        ▼
[학습 pod container disk runs/{name}/Epoch{N}.pth]
        │
        │ checkpoint-uploader.sh (polling, rclone)
        ▼
[pCloud checkpoints/{name}/Epoch{N}.pth]
```

## 4. 비용 최적화 전략

### 4.1. Multi-pod 분산 preprocess

initial preprocess 가 single H200 pod 에서 8+ 시간 (interaction full set). 비용 ≈ $32. 대신:

- **8 pod 분산**: 각 pod 가 자기 shard range 만 download + preprocess + upload → 1 시간 안 마무리. CPU pod ($0.96/h) × 8 = $7.68
- **fast-vs-slow race**: 같은 batch 를 slow pod (32 vCPU, contended host) 에 배정한 후 fast pod (128+ vCPU, idle host) 에도 race 시작. fast pod 가 5~10x 빨라 먼저 upload — slow pod 의 결과는 redundant 가 됨. 안전: rclone copy 가 atomic write + deterministic content (같은 raw + 같은 script → 같은 output)

**검증된 최적화**:
- batch_size 50 shards / pod, parallel upload transfers=32 / multi-thread-streams=4
- 100 shard batch (interaction) 처리에 contended pod 30분, idle fast pod 8분

### 4.2. Region lock-in 회피

pCloud 가 region-free → 학습 시작 시 H200 가용 region 어디든 사용 가능:
- US-IL-1 부족 → US-CA-2 / US-NC-1 / EU-CZ-1 등 대체

RunPod network volume 사용 시 lock 됨 — pCloud 우선 선택.

### 4.3. Checkpoint 자동 upload

학습 중 매 epoch 마다 `Epoch{N}.pth` (~50~200 MB) 가 생성. `scripts/cloud/transfer/checkpoint-uploader.sh` 가 폴링으로 새 .pth 발견 시 pCloud upload (학습 영향 없음). 학습 끝나면 모든 checkpoint 가 pCloud 에 보존 — pod destroy 후에도 안전.

### 4.4. CPU pod 으로 data preparation

GPU pod ($1~16/h) 대신 CPU pod ($0.96/h) 으로 download + preprocess. 학습은 GPU pod 에서. 데이터 분리:
- CPU pod: download + preprocess + tar + upload
- GPU pod: download (pCloud) + untar + train + checkpoint upload

CPU pod 은 학습이 시작되면 destroy → 비용 절감.

## 5. 운영 측면 함정 + 회피

### 5.1. Container disk resize destructive

RunPod 의 `pod update --container-disk-in-gb N` 은 disk 전체 reset (데이터 손실). 큰 disk 필요 시 새 pod create 권장. 재현 시 학습 pod 은 처음부터 큰 disk (4 TB) 로 생성.

### 5.2. multiprocessing.Pool deadlock at high worker count

worker 수 가 cgroup CPU limit 의 50% 를 초과하면 deadlock 위험. 측정값:
- nurmh (cgroup 20.4 vCPU): worker 90 → deadlock, worker 10 → 정상
- tjure (cgroup 81.6 vCPU): worker 40 → 정상

cgroup CPU 는 `cat /sys/fs/cgroup/cpu.max` (v2) 또는 `cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us` / `cpu.cfs_period_us` (v1) 로 확인.

### 5.3. AWS S3 cli pagination bug (RunPod S3 endpoint)

`aws s3 cp --recursive` 가 RunPod S3 endpoint 의 some prefix 에서 pagination bug. rclone 사용 권장.

### 5.4. Docker Hub rate limit

anonymous pull 100/6h 제한 → 동시 multi-pod 생성 시 실패 가능. 해결:
- ghcr.io (private + PAT 인증) 으로 image migrate (skopeo copy)
- 또는 pre-pulled image 의 RunPod container registry 등록

## 6. 실제 학습 구성 (재현)

데이터 + 환경 준비 후 학습 launch:

| pod | GPU | spec | 학습 | hyperparameter | ETA |
| --- | --- | --- | --- | --- | --- |
| nurmh (1×H200) | 1×H200 | 96 vCPU, 251 GB RAM, 4 TB disk | open_loop_planning | batch 128, lr 2.83e-4, epoch 20, 1 GPU | ~22h |
| tjure (4×H200) | 4×H200 | 192 vCPU, 1.5 TB RAM, 4 TB disk | interaction_prediction | batch 64/GPU × 4 = effective 256, lr 2e-4, epoch 30, DDP | ~15h |

paper hyperparameter (batch 32 lr 1e-4 / batch 16/GPU lr 1e-4) 에서 batch 4x scaling + lr sqrt scaling 적용 — VRAM 여유가 컸기 때문.

## 7. 자동화 script 정리

`scripts/cloud/` 아래 파일별 역할:

| 폴더 | 파일 | 역할 |
| --- | --- | --- |
| `cloud/preprocess/` | `pod-preprocess-pcloud-batch.sh` | pod 1개 가 자기 shard range preprocess + tar+split + pCloud upload |
| `cloud/preprocess/` | `pod-preprocess-interaction-batch.sh` | 위 와 동일 (interaction subset, single thread) |
| `cloud/preprocess/` | `pod-pipeline-h200-single.sh` | single H200 pod 에서 download wait + untar + sequential preprocess |
| `cloud/transfer/` | `pcloud-public-download.py` | pCloud public link download (auth X, parallel) |
| `cloud/transfer/` | `cloud-rclone-upload.sh` / `download.sh` | rclone (pCloud OAuth) wrapper |
| `cloud/transfer/` | `checkpoint-uploader.sh` | 학습 중 새 checkpoint 발견 → pCloud auto-upload (polling) |
| `cloud/train/` | `pod-train-from-scratch-open-loop.sh` | open_loop chain (download + untar + launch training) |
| `cloud/train/` | `pod-train-from-scratch-interaction.sh` | interaction chain (DDP) |
| `cloud/train/` | `pod-train-{open-loop,interaction}.sh` | rclone version (auth 필요) |

`scripts/cloud/train/pod-train-from-scratch-*.sh` 한 줄로 새 pod 에서 download → untar → train 까지 자동 실행:

```bash
nohup bash scripts/cloud/train/pod-train-from-scratch-open-loop.sh > /workspace/logs/launch.log 2>&1 &
```

## 8. 정리

cloud GPU 운영의 핵심은:
1. 데이터 위치를 region 자유 (pCloud) 로 → GPU 가용성 따라 region 자유 선택
2. 분산 preprocess + race 패턴으로 시간/비용 최적화
3. 큰 disk + 적절한 worker 수로 안정성 확보
4. checkpoint 자동 upload 로 데이터 손실 방지

이 전략 조합으로 full WOMD 학습 (preprocess + 학습) 을 ~$400 안에 마무리 가능 ($25 cluster preprocess + $90 open_loop + $239 interaction + $30~50 buffer).
