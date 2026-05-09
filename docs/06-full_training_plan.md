---
version: 0.3.0
status: draft
---

## 06 - 본격 학습 plan

 (1차 cloud smoke) 통과 후 본격 학습 진입을 위한 design. 단계 분해, GPU/비용 매핑, 산출물 정책, 위험 대응 정리.

### 1. 목표

- paper (ICCV'23) 의 두 task 를 cu118 stack + RunPod cloud 에서 재현
- 결과 metric 을 paper baseline 또는 official checkpoint 와 정합 검증
- 산출물 (checkpoint + log) 을 network volume 에 보존, host 회수 가능

### 2. 단계 분해

user 결정: 4090 은 동작 확인만. 본격 학습은 H100/H200. batch 도 GPU memory 적절 조정. open_loop 는 single GPU (코드 가 DDP 미지원). (b) 후보 B 확정 — batch 64 + lr 2e-4 (sqrt scaling) + paper milestones epoch 20.

| 단계 | 작업 | 데이터 | GPU | batch | lr | epoch | 시간 estimate | 비용 estimate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (b) smoke | open_loop 1 epoch sanity | training_20s 9000+1000 (preprocess 후) | 1×H200 | 64 | 2e-4 | 1 | ~2.5 분 | $0.2 |
| (b) full | open_loop full | 위 | 1×H200 | 64 | 2e-4 | 20 | ~50 분 | ~$4 |
| (c) | interaction single GPU | training entire (425 GiB) | 1×H100 | 32 | 1e-4 (paper) | 30 | ~45 시간 | ~$135 |
| (d) | interaction multi-GPU (paper 재현) | 위 + DDP 4 GPU | 4×H100 | 32×4=128 effective | 1e-4 | 30 | ~11 시간 | ~$135 |
| (d) alt | 위 | 위 | 4×H200 | 32×4=128 | 1e-4 | 30 | ~7~8 시간 | ~$140 |

(b) 의 lr scaling: paper batch 32 lr 1e-4 → batch 64 시 sqrt scaling 1.4e-4 또는 round-up 2e-4. warmup 5% step 권장 (train.py 미구현 — 별도 patch 필요한지 1 epoch 후 검토).

estimate 의 throughput 가정:
- H100 vs 3060: ~10x throughput (FP16/TF32 + memory bandwidth)
- H200 vs H100: ~1.4x (HBM3e + bandwidth)
- batch 늘리면 GPU util 100% 까지 throughput 비례 증가 (smoke batch 8 시 16% util → 적어도 batch 64 까지 선형 가능)
- 본격 진입 1 epoch 측정 후 보정

paper dataset 명세:
- open_loop: "we randomly select 10,000 20-second scenarios, where 9,000 of them are used for training and the remaining 1,000 for validation" (Sec 4.2.2)
- interaction: "trained on the entire WOMD training dataset" (Sec 4.2.1)

### 3. 단계별 spec

#### (b) open_loop_planning full (B 후보 확정)

| 항목 | 값 | 출처 |
| --- | --- | --- |
| script | `IN_POD=1 BATCH_SIZE=64 EPOCHS=20 LR=2e-4 NAME=op_full bash scripts/local/06-open_loop_train.sh` | |
| epoch | 20 | paper |
| batch_size | 64 (paper 32 의 2x — sqrt lr scaling 보정 가능 범위) | user B |
| learning_rate | 2e-4 (sqrt scaling 1.4e-4 의 round-up) | user B |
| scheduler | paper MultiStepLR milestones=[10,12,14,16,18] gamma=0.5 (train.py 그대로) | paper |
| levels | 4 | paper / default |
| seed | 3407 | default |
| dataset | training_20s 9000 train + 1000 valid scenario (paper Sec 4.2.2) | → |
| input | `/workspace/data/processed/open_loop/train` (+ valid split) | 패턴 |
| output | `/workspace/data/runs/op_full/` | patch |
| GPU | 1×H200 (us-il-1 가용성 확인 필요, fallback 1×H100) | smoke 통과 |
| 절차 | 1 epoch sanity (~2.5분) → loss 가 paper baseline (val_plannerADE ≈ 0.83) 과 합리적 거리면 19 epoch 추가 | smoke-then-full 패턴 |

epoch 시간 estimate (B 후보, H200 batch 64):
- smoke (4090, batch 8, 2 shard = 3610 sample) 1 epoch train 5분 10초 → 11.6 sample/s
- 4090 → H200 throughput 약 7x (TF32 + memory bandwidth, 외삽)
- batch 8 → 64 (8x batch ratio) — GPU util 100% 까지 선형 가정 (smoke 16% 였으니 8x 까지 fit)
- aggregate throughput: 11.6 × 7 × 8 ≈ 650 sample/s
- 280000 sample / 650 = 430 초 ≈ 7 분 / epoch (보수적 estimate)
- 20 epoch ≈ 2.4 시간 ($10)
- 위 estimate 도 추정 — 1 epoch 측정 후 보정. 실제는 2.5~7 분 사이 가능

#### (c) interaction_prediction full single GPU

| 항목 | 값 | 출처 |
| --- | --- | --- |
| script | `IN_POD=1 ... bash scripts/07-interaction_train.sh` (script 미존재 — 신규 발행 필요) | — |
| epoch | 30 | `train.py` default |
| batch_size | 16 | paper / default |
| learning_rate | 1e-4 | default |
| level | 3 (decoder reasoning) | default |
| modalities | 6 | default |
| future_len | 80 (8s @ 10Hz) | default |
| neighbors_to_predict | 1 (Waymo Joint) | default |
| seed | 3407 | default |
| dataset | training (full) — README NOTE 따라 training_20s 안 씀 | |
| input | `/workspace/data/processed/interaction/train` + `/workspace/data/processed/interaction/valid` | 패턴 확장 |
| output | `/workspace/data/runs/ip_full/` | |
| GPU 후보 | 1×H100 ($2.5~3/hr) | US-IL-1 / US-CA-2 |

epoch 시간 estimate: 3060 baseline interaction smoke 1 epoch = 13분 (8520 sample, batch 4). full training (425 GiB ≈ 8.5x training_20s sample size 추정) 에 batch 16 (smoke 의 4x), throughput H100 / 3060 ≈ 30x → 1 epoch ~ 15분. 30 epoch ~ 7.5 시간.

#### (d) interaction_prediction multi-GPU (paper 재현)

| 항목 | 값 |
| --- | --- |
| script | `IN_POD=1 NPROC_PER_NODE=4 ... bash scripts/07-interaction_train.sh` |
| GPU | 4×H100 ($10~12/hr) |
| nproc_per_node | 4 |
| effective batch | 64 (16 × 4) |
| 다른 hyperparameter | (c) 와 동일 |
| epoch 시간 | (c) / 4 — NCCL overhead 고려 = ~5분/epoch, 30 epoch ~2.5 시간 |

paper 의 4 GPU setup 그대로. (c) 와 결과 비교가 NCCL 정합성 검증 (loss curve + final metric 동일해야).

### 4. GPU 매핑 + 가용성

| GPU | mem | RunPod region | $/hr (SECURE) | 적합 단계 |
| --- | --- | --- | --- | --- |
| RTX 4090 | 24 GB | US-IL-1 | $0.69 | smoke |
| A100-40 | 40 GB | US-IL-1 / US-CA-2 | $1.5~2 | (b) backup |
| A100-80 | 80 GB | US-CA-2 | $2~2.5 | (b) (c) low-end |
| H100 | 80 GB | US-CA-2 | $2.5~3 | (b) full primary, (c) primary |
| H100 ×4 | 320 GB | US-CA-2 | $10~12 | (d) |
| H200 | 141 GB | US-CA-2 (제한) | $4~5 | (b) (c) 후속 — batch 더 크게 |
| H200 ×4 | 564 GB | US-CA-2 (제한) | $16~20 | (d) 후속 |

진입 시점 가격은 RunPod console 로 보정. region 선택은 dataset bucket (us-central1) 인접도 + GPU stock + volume region 일치 (`kingszun-storage` 가 us-il-1 → 같은 region 권장).

### 4-bis. region 충돌 검토

`kingszun-storage` (us-il-1) 와 GPU pod region 불일치 시 volume mount 가 안 됨 (RunPod network volume 은 region 고정). H100/H200 가 us-il-1 에 stock 있는지 확인 필요:

| region | 4090 | A100 | H100 | H200 |
| --- | --- | --- | --- | --- |
| US-IL-1 | High | Med | Low? | Very Low? |
| US-CA-2 | Low | High | High | Med |

us-il-1 에 H100 부족 시 옵션:
- a. volume migration 비용 (data transfer $X/GB)
- b. us-il-1 의 A100-80 으로 (b)/(c) 진행 — 상대 throughput 감소
- c. 신규 us-ca-2 volume 생성 + 데이터 재 push (S3 endpoint 으로) — second push

진입 시점 RunPod console 에서 가용성 확인 후 결정.

### 5. 데이터 prep

| subset | size (raw) | 단계 |
| --- | --- | --- |
| training_20s | 29.4 GiB | (b) |
| training | 424.6 GiB | (c)(d) |
| validation | 38.4 GiB | (b) val |
| validation_interactive | 37.8 GiB | (c)(d) val |

진행 chain:
1. GCS → host download (진행 중, ~3시간 ETA)
2. host → RunPod volume S3 endpoint push
3. preprocess: pod 안에서 `scripts/local/05-open_loop_preprocess.sh` (open_loop) + `scripts/07-interaction_preprocess.sh` (interaction, 신규 script 필요)
4. 본격 학습 진입

preprocess output 크기:
- open_loop: smoke (2 shard 164 MB → 1.8 GB processed). 비례 → 1000 shard ≈ ~50 GB
- interaction: baseline (training 2 shard 886 MB → 2.2 GB processed). 비례 → 1000 shard ≈ ~110 GB
- 합 + raw 530 GB → volume 100 GB 에 안 들어감 → volume 확장 필요

volume 확장:
- 100 GB → 800 GB ($0.07/GB/월 = $56/월). preprocess 결과까지 보존.
- 또는 preprocess 후 raw 삭제 (volume 절약, 재 preprocess 위험)

### 6. 산출물 정책

patch 적용 — 모든 산출물 volume 에 저장:

| 산출물 | path |
| --- | --- |
| raw | `/workspace/data/raw/{subset}/` |
| processed | `/workspace/data/processed/{open_loop,interaction}/{split}/` |
| training log + checkpoint | `/workspace/data/runs/{NAME}/` |
| stdout log | `/workspace/logs/{TICKET}_{cmd}.log` (host 회수 대상) |

회수: 학습 끝 시 host 로 scp (rsync `/workspace/data/runs/` → host `runs/`). volume 자체는 유지.

### 7. 진행 dependency

```
 (host download, ~3hr)
    └─> (host → volume push, ETA ~5~10hr 회선 의존)
            ├─> open_loop preprocess (volume 안에서, ~1~2hr)
            │ └─> (b) open_loop full (4090, ~12hr)
            ├─> interaction preprocess (volume 안에서, ~3~4hr)
            │ ├─> (c) interaction single GPU (H100, ~7.5hr)
            │ └─> (d) interaction multi-GPU (4×H100, ~2.5hr)
            └─> volume 확장 (100 GB → 800 GB)
```

순서:
1. 끝 → 시작
2. 끝 → preprocess sub-task 동시 (open_loop + interaction 병렬, CPU bound 라 같은 pod 가능)
3. preprocess 끝 → 학습 단계 — (b), (c), (d) 순차 또는 우선순위 결정 후

### 7-bis. paper baseline metric (재현 비교 기준)

paper section 4.2 발췌. arxiv 2303.05760 PDF Table 1, 2, 3 인용.

#### Table 1 — interaction prediction (WOMD interaction prediction benchmark)

GameFormer 두 variant + 비교 모델 (vehicle/pedestrian/cyclist 평균, 3/5/8s 평균):

| 모델 | minADE ↓ | minFDE ↓ | miss rate ↓ | mAP ↑ |
| --- | --- | --- | --- | --- |
| LSTM baseline | 1.9056 | 5.0278 | 0.7750 | 0.0524 |
| Heat | 1.4197 | 3.2595 | 0.7224 | 0.0844 |
| AIR² | 1.3165 | 2.7138 | 0.6230 | 0.0963 |
| SceneTrans | 0.9774 | 2.1892 | 0.4942 | 0.1192 |
| DenseTNT | 1.1417 | 2.4904 | 0.5350 | 0.1647 |
| M2I | 1.3506 | 2.8325 | 0.5538 | 0.1239 |
| MTR | 0.9181 | 2.0633 | 0.4411 | 0.2037 |
| GameFormer (M, M=64) | 0.9721 | 2.2146 | 0.4933 | 0.1923 |
| GameFormer (J, M=6) | 0.9161 | 1.9373 | 0.4531 | 0.1376 |

J = joint (M=6), M = marginal (M=64) + EM aggregation. 우리 (c)(d) 단계는 J variant — repo 의 default (`modalities=6`) 와 일치.

#### Table 2 — open-loop ablation (decoding levels K)

| K | Planning ADE | Collision Rate | Miss Rate | Prediction ADE |
| --- | --- | --- | --- | --- |
| 0 | 0.9458 | 0.0384 | 0.1154 | 1.0955 |
| 1 | 0.8846 | 0.0305 | 0.0994 | 0.9377 |
| 2 | 0.8529 | 0.0277 | 0.0897 | 0.8875 |
| 3 | 0.8423 | 0.0269 | 0.0816 | 0.8723 |
| 4 (paper 채택) | 0.8329 | 0.0198 | 0.0753 | 0.8527 |
| 5 | 0.8171 | 0.0245 | 0.0777 | 0.8361 |
| 6 | 0.8208 | 0.0238 | 0.0826 | 0.8355 |

우리 (b) 는 `--levels 4` (default) 로 K=4 결과 비교 기준.

#### Table 3 — open-loop final (vs baselines)

| 모델 | Collision rate (%) | Miss rate (%) | Planning err @1s | @3s | @5s | Pred ADE | Pred FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla IL | 4.25 | 15.61 | 0.216 | 1.273 | 3.175 | – | – |
| DIM | 4.96 | 17.68 | 0.483 | 1.869 | 3.683 | – | – |
| MultiPath++ | 2.86 | 8.61 | 0.146 | 0.948 | 2.719 | – | – |
| MTR-e2e | 2.32 | 8.88 | 0.141 | 0.888 | 2.698 | – | – |
| DIPP | 2.33 | 8.44 | 0.135 | 0.928 | 2.803 | 0.925 | 2.059 |
| Ours (GameFormer K=4) | 1.98 | 7.53 | 0.129 | 0.836 | 2.451 | 0.853 | 1.919 |

#### (smoke, batch 8, 1 epoch) 와 비교 (참고용 — 학습 불충분)

| metric | (1 epoch, batch 8, 2 shard subset) | paper (20 epoch, batch 32, 10000 scenario) |
| --- | --- | --- |
| val_plannerADE | 8.51 | 0.83 |
| val_predictorADE | 8.18 | 0.85 |

10x 차이 — 본격 학습 시 paper 수준 수렴 가능성 검증 필요. 1차 epoch 후 loss curve 추이 확인.

### 8. 검증 / acceptance

각 단계 acceptance:

| 단계 | metric | 기준 (paper Table) |
| --- | --- | --- |
| (b) | Planning ADE, Collision Rate, Miss Rate, Prediction ADE | Table 2 K=4: 0.8329 / 0.0198 / 0.0753 / 0.8527 ±15% (학습 random + dataset random) |
| (c) | minADE / minFDE / miss rate / mAP (vehicle/ped/cyclist 평균) | Table 1 GameFormer (J, M=6): 0.9161 / 1.9373 / 0.4531 / 0.1376 ±10% |
| (d) | (c) 와 동일 + DDP NCCL 정합 | (c) 결과와 동일해야. 다르면 NCCL sync 문제 |

추가 검증:
- loss curve overfit 확인 (val_loss 가 어느 epoch 부터 증가하는지)
- early stopping criterion 설정 (paper 명시 X 라 시각 review)
- checkpoint disk 사용량 모니터 (volume 확장 trigger)

### 9. 위험 / 대응

| 위험 | 대응 |
| --- | --- |
| volume 부족 (100 GB) | 진입 전 800 GB 확장 |
| 4090 OOM (b 의 batch 32) | 1차 batch 16 → 점진 증가. nvidia-smi mem 모니터 |
| H100 stock 부족 (US-CA-2) | A100-80 fallback 또는 region 변경 |
| NCCL 정합성 (d 단계) | single vs multi GPU 결과 비교. 다르면 sync 문제 의심 |
| dataset I/O bottleneck (volume mfs) | nvidia-smi 의 GPU util 50% 미만 시 detect. workaround: container 안 cache (큰 RAM 이용) 또는 sequential read |
| cost overrun | 학습 launch 전 estimate 확인. epoch 단위 nvidia-smi power + GPU util log → 중도 stop 가능 |
| WOMD validation set 결과 학습용 의존 | validation 도 학습에 절대 사용 X. `valid_set` arg 확인 |
| 산출물 path bug | smoke 1 epoch 으로 사전 검증 |

### 10. 후속 sub-task plan

design Done 후 다음 ticket 발행:

| ticket | 내용 | 상태 |
| --- | --- | --- |
| ~~volume 확장~~ | 100 → 1024 GB | done — 26-05-08 (kingszun-storage) |

### 11. 미결정 / 추가 검토 필요

- ~~paper 의 official baseline metric 표는 paper PDF / project page 에서 확인 후 본 문서 update~~ Done — section 7-bis Table 1, 2, 3 참조
- ~~4090 1 장으로 (b) open_loop full 20 epoch 가 9 일 estimate~~ Done — paper 의 dataset 이 9000 scenario subset 이라 4090 32 시간 ($22) 으로 충분
- epoch 시간 estimate 가 smoke (batch 8, GPU util 16%) 외삽 — 본격 학습 진입 시 1 epoch 측정 후 보정
- (b) preprocess size estimate ~140 GB — 100 GB volume 부족. 200~300 GB 확장 필요 (또는 raw 삭제 후)
- (c)(d) preprocess size estimate ~1.1 TB — volume 대폭 확장 (1.5 TB ≈ $105/월) 또는 sub-streaming 검토
- paper 의 9000 scenario 가 어느 shard 인지 미명시 — random sample 이라 우리는 first 9000 또는 random seed 동일 보장 X. metric ±15% margin 으로 흡수
- early stopping 정책 (paper 명시 X) — train.py 도 미구현
- checkpoint 정책 — 현재 train.py 가 모든 epoch 저장. (c) 30 epoch × 60 MB = 1.8 GB (작음, 유지) / (d) 동일
- mixed precision / gradient checkpointing 미사용 (paper 명시 X)
- (c) interaction 의 preprocess script 가 없음 — `scripts/07-interaction_preprocess.sh` 신규 발행 필요
- (c)(d) train script 도 없음 — `scripts/08-interaction_train.sh` 신규 (DDP wrapper)
- 본격 학습 결과 publish (Hub upload, paper figure 재현)
