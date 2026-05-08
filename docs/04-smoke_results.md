## 04 - smoke results

3060 12 GB 환경 1회 학습 결과. cloud로 가기 전 functional 검증 용도.

### environment

- GPU: NVIDIA GeForce RTX 3060 12 GB (sm_86)
- Driver: 591.86 (CUDA 13.1 reported)
- Container: `gameformer:cu118-py310-torch2.3.1` (torch 2.3.1, cuDNN 8700)
- Host: WSL2 (Ubuntu)

### GPU smoke

```
torch: 2.3.1
cuda available: True
device count: 1
device name: NVIDIA GeForce RTX 3060
cuda runtime: 11.8
cudnn: 8700
matmul ok, sum: -70273.734375
```

GPU passthrough + cu118 wheel binary 정상 동작 확인.

### open_loop_planning train (1 epoch)

| 항목 | 값 |
| --- | --- |
| dataset | training_20s 2 shards (164 MB raw, 3610 npz / 1.8 GB processed) |
| batch_size | 8 |
| epochs | 1 |
| learning_rate | 1e-4 |
| levels | 4 (default) |

결과 (`open_loop_planning/training_log/DESKTOP-D4G97VH/train_log.csv`):

| metric | value |
| --- | --- |
| train_loss | 91.05 |
| val_loss | 48.04 |
| train_plannerADE | 11.38 |
| train_plannerFDE | 23.29 |
| train_predictorADE | 8.35 |
| train_predictorFDE | 13.08 |
| val_plannerADE | 8.35 |
| val_plannerFDE | 18.43 |
| val_predictorADE | 8.20 |
| val_predictorFDE | 12.89 |

checkpoint: `predictor_1_8.3482.pth`

### interaction_prediction train (1 epoch, single-GPU DDP)

| 항목 | 값 |
| --- | --- |
| dataset | training 2 shards (886 MB raw, 8518 npz / 2.2 GB processed) |
| batch_size | 4 |
| epochs | 1 |
| workers | 2 |
| nproc_per_node | 1 (DDP launcher, single GPU) |

결과 (`interaction_prediction/training_log/smoke/train_log.csv`):

| metric | value |
| --- | --- |
| train_loss | 180.98 |
| val_loss | 109.00 |
| minADE_VEHICLE_5 | 6.96 |
| minADE_VEHICLE_15 | 17.45 |
| minADE_PEDESTRIAN_5 | 4.27 |
| minADE_CYCLIST_5 | 5.48 |
| minFDE_VEHICLE_15 | 33.80 |
| miss_rate_VEHICLE_15 | 0.997 |

수렴 곡선 (sample iter 기준):
- step 4: loss 21888
- step 100: loss 2700 (~10x 감소)
- step 1000: loss ~110
- step 8520 (epoch 끝): loss 109

checkpoint: `epochs_0.pth`

### performance baseline

| GPU | task | batch | s/sample (steady) | 1 epoch 예상 |
| --- | --- | --- | --- | --- |
| 3060 | open_loop_planning | 8 | n/a | ~10분 (3610 sample) |
| 3060 | interaction_prediction | 4 | 0.0146 | ~13분 (8520 sample) |
| H100 (예상) | interaction_prediction | 16 | ~0.003 | ~2~3분 |
| H200 (예상) | interaction_prediction | 32 | ~0.002 | ~1~2분 |

### verified path summary

| path | status | 비고 |
| --- | --- | --- |
| Docker image build (local) | ok | |
| Docker Hub push (`kingszun/gameformer`) | ok | |
| GPU passthrough (compose) | ok | |
| torch CUDA + matmul | ok | |
| WOMD tfrecord parsing (v1.2.1) | ok | patch 1 적용 후 |
| open_loop_planning preprocess | ok | |
| open_loop_planning train + checkpoint | ok | |
| interaction_prediction preprocess | ok | |
| interaction_prediction DDP train (single GPU) | ok | patch 2 적용 후 |
| waymo metrics ops (TF+torch 동거) | ok | TF_FORCE_GPU_ALLOW_GROWTH로 conflict 회피 |
| Multi-GPU NCCL all-reduce | unverified | 3060 1장으로 검증 불가, cloud 단계에서 |
| 대규모 dataset (수백 GB) iteration | unverified | cloud 단계에서 |
