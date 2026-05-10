# 03. Data pipeline — raw 부터 metric 까지

WOMD raw tfrecord 가 model input 이 되어 학습되고 metric 이 산출되기까지 과정과 각 단계의 paper 적 의미.

## 1. Dataset overview — Waymo Open Motion Dataset (WOMD)

paper 가 사용하는 WOMD v1.2.1:

| subset | scenarios | shards (1000 total) | size | 용도 |
| --- | --- | --- | --- | --- |
| `training` | ~487K | 1000 | 425 GB | interaction prediction 학습 |
| `training_20s` | ~70K (sparse subset of training) | 344 (sparse) | 30 GB | open-loop planning 학습 (긴 horizon scene) |
| `validation` | ~44K | 150 | 39 GB | open-loop planning 검증 |
| `validation_interactive` | ~44K | 150 | 38 GB | interaction prediction 검증 |

각 scenario 는:
- 시간: 9.1 sec total (1.1s 과거 + 8s 미래) for training/validation, 20s total for training_20s
- 객체: ego + 다수 neighbor (vehicle, pedestrian, cyclist) + map (lanes, crosswalks, traffic_lights, stop_signs)
- format: protobuf serialized in tfrecord

## 2. Pipeline 단계별

```
[Waymo GCS bucket]
  └─▶ download (gsutil) ─▶ raw/{subset}/*.tfrecord-NNNNN-of-01000
                              │
                              ▼
                       data_process.py (multiprocessing pool)
                              │
                              ├─ 1. tfrecord parse → scenario_pb2.Scenario
                              ├─ 2. ego/neighbor 추출 + filter
                              ├─ 3. trajectory tensor 변환 (history 11 step + future 80 step)
                              ├─ 4. map feature 추출 + 정규화 (per-agent local frame)
                              ├─ 5. quality filter (kalman, missing data 등)
                              └─ 6. .npz save (sample 단위)
                              │
                              ▼
                processed/{subset}/{scenario_id}_{ego}_{neighbor}_*.npz
                              │
                              ▼
                       Dataset class (DrivingData / DrivingData_Inter)
                              │
                              ├─ glob 으로 .npz 파일 list
                              ├─ __getitem__: np.load + np.array → tensor
                              └─ DataLoader (DistributedSampler in DDP)
                              │
                              ▼
                          batch (B, ...)
                              │
                              ▼
                       GameFormer(inputs)
                              │
                              ├─ Encoder
                              └─ Decoder (level 0..K)
                              │
                              ▼
                  predictions: dict of level_k_interactions (B, N, M, T, 4) + scores (B, N, M)
                              │
                              ▼
                       loss = Σ_k (L_traj_k + L_score_k)
                              │
                              ▼
                       optimizer.step()
                              │
                              ▼
                       valid_epoch (rank=0):
                         - inference on validation set
                         - metrics: minADE, minFDE, MR, mAP
                         - save: train_log.csv + epochs_{epoch}.pth
```

## 3. raw → processed 단계 의 paper 적 의미

### 3.1. tfrecord parse

각 tfrecord 안에 protobuf serialized `Scenario` 가 모여있음. `scenario_pb2.Scenario.FromString(record)` 으로 deserialize.

```python
data = tf.data.TFRecordDataset([file], compression_type="")
for record in data:
    parsed_data = scenario_pb2.Scenario()
    parsed_data.ParseFromString(record.numpy())
    self.process_one_scenario(parsed_data)
```

- 1 tfrecord ≈ 487 scenario (training 기준)
- scenario 의 timestamps: 91 step (10 Hz) — 11 history + 80 future

### 3.2. ego / neighbor 추출

paper Sec 4.2.1 의 interaction prediction:
- ego = `current_time_index` 에서 valid 한 trajectory (= predict 대상)
- neighbor = ego 와의 distance + valid history 기준 top 32

```python
sdc_id = scenario.sdc_track_index  # ego (self-driving car)
tracks_to_predict = scenario.tracks_to_predict  # paper 의 interesting agent pair

# 각 (ego, neighbor) pair → 1 sample
for pair in tracks_to_predict:
    ego_idx, neighbor_idx = pair.tracks_to_predict
    ...
```

`validation_interactive` 는 per-scenario 2 개의 interaction pair (ego + 1 neighbor) 를 미리 정해놨음 — 86958 = 43479 scene × 2.

### 3.3. trajectory tensor 변환

paper Sec 3.2 의 input format:
- history: `(11, 8)` — 11 timestep 의 (x, y, heading, vx, vy, w, l, h)
- future: `(80, 4)` — 80 timestep 의 (x, y, heading, valid)

각 agent 의 history + future 를 ego-centric local frame 으로 변환 (현재 timestep 기준 회전 + 평행이동).

### 3.4. map feature 추출

paper Sec 3.2 의 map representation:
- lane: `(N_lanes, 100, 16)` — 각 lane 100 point, 16 feature (3 self + 3 left + 3 right + speed_limit + 6 type/sign embedding)
- crosswalk: `(N_crosswalks, 100, 3)` — 각 crosswalk 100 point, (x, y, type)

per-agent local frame 으로 회전 — paper Sec 3.2 의 "**we transform all map elements into each agent's local frame**". 이 transformation 이 attention fusion 의 평행이동 invariance 보장.

### 3.5. quality filter

```python
# data_process.py 의 valid_check
- agent 의 history 가 적어도 3 step 이상 valid
- agent 가 일정 거리 이상 움직임 (정지 차량 제외)
- map 가 충분히 가까움 (lane 100m 이내)
- ...
```

filter 에 통과한 sample 만 .npz 로 저장. paper 가 학습에 사용한 sample 수 ≈ filter 후 scenario count.

### 3.6. .npz file 단위

각 .npz 는 **하나의 sample** = (1 scenario, 1 ego candidate). 파일명 convention:

| subset | 파일명 예 | 내용 |
| --- | --- | --- |
| open_loop train/valid | `f3f55cf64033141c_18.npz` | scenario_id + 후보 timestep |
| interaction train/valid | `cb5c992828188aca_41_10_interest.npz` | scenario_id + ego_idx + neighbor_idx + interest tag |

random shuffle 시 .npz 단위 = sample 단위. paper 의 batch 16 = 16개 .npz file load.

## 4. processed → batch 단계

### 4.1. Dataset class

```python
class DrivingData_Inter(Dataset):
    def __init__(self, data_dir):
        self.data_files = glob.glob(data_dir)  # e.g., '/path/*'
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        # return: ego_state, neighbors_state, map_lanes, map_crosswalks,
        #         ego_future, neighbors_future, ground_truth, neighbors_to_predict, ...
        return tuple(...)
```

### 4.2. DataLoader + DistributedSampler

DDP 시 각 GPU 가 dataset 의 disjoint 부분을 독립적으로 load:

```python
train_sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, batch_size=B, sampler=train_sampler, num_workers=W, pin_memory=True)
```

- effective batch = B × world_size (DDP)
- paper: B=16 per GPU × 4 GPU = 64 effective batch
- 재현: B=64 per GPU × 4 GPU = 256 effective batch (sqrt scaling lr 1e-4 → 2e-4)

## 5. batch → loss 단계 의 paper 적 의미

```python
# train_epoch
for batch in train_loader:
    inputs = {
        'ego_state': batch[0].cuda(),         # (B, 11, 9)
        'neighbors_state': batch[1].cuda(),   # (B, 32, 11, 9)
        'map_lanes': batch[2].cuda(),         # (B, 33, N_lanes, 100, 16) — per-agent
        'map_crosswalks': batch[3].cuda(),    # (B, 33, N_cw, 100, 3)
    }
    ego_future = batch[4].cuda()              # (B, 80, 4)
    neighbors_future = batch[5].cuda()        # (B, 32, 80, 4)
    
    outputs = model(inputs)
    # outputs: dict, level_k_interactions (B, N, M, T, 4), level_k_scores (B, N, M)
    
    loss = interaction_loss(outputs, ground_truth)
    loss.backward()
    optimizer.step()
```

### 5.1. Loss decomposition

```python
# pseudo-code
def interaction_loss(outputs, gt):
    total_loss = 0
    for k in range(K+1):
        pred_traj = outputs[f'level_{k}_interactions']  # (B, N, M, T, 4)
        pred_score = outputs[f'level_{k}_scores']       # (B, N, M)
        
        # winner-takes-all: 각 agent 별 GT 와 가장 가까운 mode
        distances = ...  # (B, N, M)
        best_mode = distances.argmin(dim=-1)  # (B, N)
        
        # GMM negative log-likelihood for best mode
        nll = -gaussian_log_prob(pred_traj[best_mode], gt[..., :2])
        
        # cross-entropy for mode classification
        ce = F.cross_entropy(pred_score, best_mode)
        
        total_loss = total_loss + nll + ce
    return total_loss
```

paper 의 핵심 design choice:
- **모든 level 의 loss 를 합산** → deep supervision. level 0 (initial) 부터 level K (final) 까지 모두 GT 를 직접 본다.
- **winner-takes-all** → mode collapse 방지. 단 1개 mode 만 update (gradient 가 전체 mode 에 분산되지 않음).

## 6. metric 산출 — `valid_epoch` 단계

paper Sec 4.2.1 의 evaluation:

```python
def valid_epoch(loader, model):
    model.eval()
    ade_list, fde_list, mr_list, ap_list = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            outputs = model(inputs)
            
            # 마지막 level 만 사용 (가장 정제된 prediction)
            pred = outputs[f'level_{K}_interactions']  # (B, N, M, T, 4)
            scores = outputs[f'level_{K}_scores']       # (B, N, M)
            
            # minADE: best mode 의 average distance
            ade = compute_min_ade(pred[..., :2], gt[..., :2])
            
            # minFDE: best mode 의 endpoint distance
            fde = compute_min_fde(pred[..., -1, :2], gt[..., -1, :2])
            
            # Miss Rate: endpoint 가 일정 box 안에 들어가는지
            mr = compute_miss_rate(pred[..., -1, :2], gt[..., -1, :2])
            
            # mAP: confidence-weighted Average Precision
            ap = compute_map(pred, scores, gt)
    
    return mean(ade), mean(fde), mean(mr), mean(ap)
```

각 metric 의 paper 적 의미:
- **minADE/minFDE**: model 이 multi-modal output 중 GT 에 가까운 mode 를 만들 수 있는지 (recall). Lower is better.
- **Miss Rate**: 모든 modal 의 endpoint 가 일정 box 안에 안 드는 (즉, 어떤 modal 도 정확하지 않은) 비율. Lower is better.
- **mAP**: confidence (score) 가 높은 mode 가 정확한지 (precision). Higher is better.

paper Table 1 의 Joint (M=6) baseline:

| metric | value |
| --- | --- |
| minADE | 0.9161 m |
| minFDE | 1.9373 m |
| Miss Rate | 0.4531 |
| mAP | 0.1376 |

이 4 값이 reproduction 의 비교 기준.

## 7. 학습 끝 → checkpoint → final metric

```python
# rank=0 save (DDP)
torch.save(save_state, log_path + f'epochs_{epoch}.pth')

# train_log.csv: epoch 별 train_loss, val_loss, val_ade, val_fde, val_mr, val_map
```

학습 완료 후:
1. validation set 에서 가장 낮은 val_ade 또는 val_loss 의 epoch checkpoint 선택
2. 그 checkpoint 로 inference 다시 실행 → test set 또는 leaderboard submission (paper 는 validation 에서 보고)
3. paper Table 1 과 비교 → reproduction 성공/실패 판정

이 reproduction 의 acceptance: **paper baseline 의 ±15% 이내** (KAK-34 v0.3 acceptance criteria 와 동일).

## 8. open-loop planning 의 추가 단계

interaction prediction 위에 ego trajectory 를 reference path 와 결합:

```python
# open_loop_planning/train.py
inputs = {
    'ego_state': ...,
    'neighbors_state': ...,
    'map_lanes': ...,
    'map_crosswalks': ...,
    'ref_line': batch[4].cuda(),   # reference path (lane network 따른 ideal trajectory)
}
ego_future = batch[5][:, 0]         # ego 의 GT future
neighbors_future = batch[5][:, 1:]  # neighbor 의 GT future

# loss = ego planning loss + neighbor prediction loss
```

ego 의 planning quality 를 measure 하는 추가 metric:
- **plannerADE/FDE**: planning trajectory 의 displacement error
- **predictionADE/FDE**: neighbor prediction 의 displacement error

Open-loop planning paper baseline (`open_loop_planning/train.py`의 default):
- val_plannerADE ≈ 0.83
- val_plannerFDE ≈ 1.5

## 9. 정리

| 단계 | 입력 | 출력 | paper 적 의미 |
| --- | --- | --- | --- |
| WOMD download | GCS | tfrecord | 학습 데이터 source |
| data_process | tfrecord | .npz | scenario → fixed-shape tensor (model input format) |
| Dataset / DataLoader | .npz | batch | training/inference 단위 |
| Encoder | (ego/neighbors/map) | encoding (B, N, dim) | per-agent context |
| Decoder level 0 | encoding | M modal trajectory + score | initial (marginal) prediction |
| Decoder level k | encoding + level k-1 | M modal trajectory + score | k-th level reasoning (game-theoretic) |
| Loss | level_0..K predictions + GT | scalar | multi-level deep supervision |
| metric | level K prediction + GT | minADE / minFDE / MR / mAP | paper Table 1 비교 |

각 단계가 paper 의 contribution 과 어떻게 연결되는지 위 표에서 확인 가능.
