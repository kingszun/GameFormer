## 07c - validation_interactive subset 상세 (interaction prediction valid)

doc. WOMD `validation_interactive` raw → `processed/interaction/valid/` → `interaction_prediction/train.py` 의 valid_epoch (DDP) 흐름.

전체 overview 는 [`07-data_pipeline.md`](07-data_pipeline.md) 참조.

---

### 1. raw 의 구조 — open_loop 의 validation 과의 핵심 차이

#### 1.1. 위치 / shard

- bucket: `gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation_interactive/`
- file 수: **150** shard
- 총 size: 38 GB
- 1 shard size: 평균 ~250 MB
- mtime: 2024-01-30

#### 1.2. validation_interactive 의 특성 (vs 일반 validation)

WOMD 의 `validation_interactive` subset 은 **interactive scenario** 만 모음. 즉:

- 같은 scenario 의 `objects_of_interest` field 에 **두 agent** 의 id 가 marked 됨
- 두 agent 는 서로 영향을 받는 상황 (예: 교차로 우회전 vs 좌회전, 추월 vs 양보, 보행자 횡단 vs 차량 정지)
- paper Table 2/3 의 평가 metric (Planning ADE, Collision rate, minADE/minFDE/miss rate) 는 이 subset 으로 측정

일반 validation 과의 비교:
| 항목 | validation (open_loop 용) | validation_interactive (interaction 용) |
|---|---|---|
| `objects_of_interest` | 보통 비어있음 | 두 agent id |
| `tracks_to_predict` | 1~8 agent (Waymo motion challenge) | interactive pair 위주 |
| Scene 형태 | 일반 driving | interaction 강조 (교차로, 보행자 횡단, 추월 등) |
| 목적 | open_loop planning eval | interaction prediction eval |

#### 1.3. Scenario field 의 추가 의미 (interaction 관점)

기본 schema 는 [`07a-open_loop_validation.md`](07a-open_loop_validation.md) 의 Scenario / Track / MapFeature 동일. 차이점:

- `objects_of_interest`: int[] — Waymo 가 marked 한 두 agent id (이걸 paper interactive metric 의 source 로 사용)
- `tracks_to_predict`: RequiredPrediction[] — Waymo motion challenge 가 prediction 요청한 agent + difficulty (LEVEL_1 / LEVEL_2)
- `current_time_index`: 보통 10 (1초, history 의 끝)

---

### 2. preprocess 절차 (`interaction_prediction/data_process.py`)

#### 2.1. open_loop 와의 핵심 차이

| 항목 | open_loop (07a) | interaction (07c, 07d) |
|---|---|---|
| 처리 단위 | 1 scenario × **1 timestep** | 1 scenario × **1 pair** |
| ego 수 | 1 (SDC) | **2 agent** (pair) |
| pair 종류 | N/A | `interest` (Waymo marked) + `r` (random pair augmentation) |
| timestep | 7개 (5 step interval) | **1개** (current_time_index = `hist_len-1` = 10) |
| neighbors 수 | 10 | **32** (더 많은 주변 agent) |
| future_len | 50 step | **80 step** (paper 의 8초 horizon) |
| map_lanes shape | (1+10, 6, 100, 16) | **(2, 6, 300, 17)** — 2 agent, 더 dense (300 point), 1 attribute 추가 |
| 추가 field | — | object_type, object_index, region_6, current_state |
| Pool processes | default os.cpu_count | **default 8** (script 안 default) |

#### 2.2. 함수 별 처리

##### Step 1: `build_map(map_features, dynamic_map_states)`

open_loop 의 build_map 과 거의 동일 — 단 `road_lines` 와 `road_edges` 를 묶어 `self.roads` 하나로 (open_loop 는 분리 후 union).

```python
self.lanes = {map_id → LaneCenter}
self.roads = {map_id → RoadLine|RoadEdge} # 하나로 통합
self.crosswalks = {map_id → Crosswalk}
self.stop_signs = {map_id → StopSign}
self.speed_bumps = {map_id → SpeedBump}
self.traffic_signals = dynamic_map_states
```

##### Step 2: `interactive_process(tracks_list, interesting_ids, tracks)` ★ open_loop 에 없는 단계

scenario 의 `tracks_to_predict` (id) + `objects_of_interest` (id) 를 받아서 **pair 식별**:

```python
self.sdc_ids_list = [] # 각 element: ((ego_id, neighbor_id), interesting_flag)

for ego_id in tracks_list:
    ego_state = tracks[ego_id].states[hist_len-1] # current state (timestep 10)
    ego_xy = (ego_state.center_x, ego_state.center_y)
    
    candidate_tracks = []
    cnt = 2 # 각 ego 당 최대 2 pair (interesting 1 + random N)
    
    if len(tracks_list) == 1:
        # tracks_to_predict 가 1개만 → 모든 다른 agent 를 candidate
        for i, track in enumerate(tracks):
            if i != ego_id and track.states[hist_len-1].valid:
                candidate_tracks.append((i, distance to ego_xy))
    else:
        for t in tracks_list:
            if t != ego_id:
                if t in interesting_ids and ego_id in interesting_ids:
                    # 두 ego 가 모두 interesting → interesting pair
                    self.sdc_ids_list.append(((ego_id, t), 1))
                    cnt -= 1
                    continue
                candidate_tracks.append((t, distance to ego_xy))
    
    # 가장 가까운 cnt 개 → random pair (interesting=0)
    sorted_candidate = sorted(candidate_tracks, key=distance)[:cnt]
    for can in sorted_candidate:
        self.sdc_ids_list.append(((ego_id, can[0]), 0))
```

→ 1 scenario 가 평균 **5~10 pair** 생성 (interesting 위주 + random augmentation).

**왜 random pair 도 추가하는가?**: data augmentation. interesting pair 만으로는 학습 sample 부족. random pair 도 학습해서 generalization 향상.

##### Step 3: `ego_process(sdc_ids, tracks)` — **2 agent** 의 history

open_loop 의 1 ego 와 다름. sdc_ids = (id1, id2):

```python
ego = np.zeros((2, hist_len, 9)) # 2 agent × 11 step × 9 attr
for i, sdc_id in enumerate(sdc_ids):
    states = tracks[sdc_id].states[0:hist_len] # 11 step
    for j, s in enumerate(states):
        ego[i, j] = [s.center_x, s.center_y, s.heading,
                     s.velocity_x, s.velocity_y,
                     s.length, s.width, s.height,
                     tracks[sdc_id].object_type]
return ego # (2, 11, 9)
```

**self.current_xyzh** = ego[0] 의 마지막 state (= 첫 번째 ego 의 current pose) — normalize origin.

##### Step 4: `neighbors_process(sdc_ids, tracks)` — pair 외의 32 agent

```python
neighbors = np.zeros((num_neighbors=32, hist_len=11, 9))
# pair 가 아닌 모든 valid agent → 거리 sort → top 32
sorted = sorted(other_agents, key=distance to ego[0] current)
for i, neighbor_id in enumerate(sorted[:32]):
    states = tracks[neighbor_id].states[0:11]
    neighbors[i, :] = ...
return neighbors, neighbors_id
```

##### Step 5: `map_process(traj)` — agent 별 호출 (2 agent)

open_loop 와 비슷하지만 차이:

- **6 lane × 300 point × 17 attribute** (open_loop 는 100 point × 16)
- threshold 200m → **300m** (더 멀리)
- 17번째 attribute = **stop_point (bool)** — 정지 line 위치

```python
vectorized_map = np.zeros((6, 300, 17))
vectorized_crosswalks = np.zeros((4, 100, 3))
# ... 6 lane × 300 point 추출 ...
# attribute 17개 (open_loop 의 16 + stop_point):
# 0,1,2: centerline xyh
# 3,4,5: left bdry xyh
# 6,7,8: right bdry xyh
# 9: speed_limit
# 10: lane.type
# 11: left_bdry_type
# 12: right_bdry_type
# 13: traffic_light state
# 14: interpolating
# 15: stop_sign
# 16: stop_point (NEW)
```

map_process 는 ego 2 agent 별로 호출 → map_lanes shape `(2, 6, 300, 17)`.

##### Step 6: `ground_truth_process(sdc_ids, tracks)` — **future 80 step**

open_loop 의 50 step 과 다름. paper 의 8초 (80 step × 0.1s) horizon:

```python
gt = np.zeros((2, future_len=80, 5)) # 2 agent × 80 step × 5 attr
for i, sdc_id in enumerate(sdc_ids):
    future_states = tracks[sdc_id].states[hist_len : hist_len+future_len] # 80 step
    for j, s in enumerate(future_states):
        gt[i, j] = [s.center_x, s.center_y, s.heading, s.velocity_x, s.velocity_y]
return gt # (2, 80, 5)
```

##### Step 7: `normalize_data(...)`

ego[0] 의 current state 를 origin 으로 좌표 변환:
- ego (2, 11, 9) — agent_norm
- neighbors (32, 11, 9) — agent_norm (impute=True for invalid)
- map_lanes (2, 6, 300, 17) — map_norm
- map_crosswalks (2, 4, 100, 3) — map_norm
- gt (2, 80, 5) — agent_norm

추가 output:
- **region_dict** — point_dir 가 있을 때만 (region 별 cluster point, vehicle/pedestrian/cyclist × 6/32/64 mode)
- region_dict 가 없으면 region_6 = `np.zeros((6, 2))`

##### Step 8: `np.savez(filename, ...)`

```python
inter = 'interest' if interesting==1 else 'r'
filename = f"{save_dir}/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"

np.savez(filename,
    ego=ego, # (2, 11, 9)
    neighbors=neighbors, # (32, 11, 9)
    map_lanes=map_lanes, # (2, 6, 300, 17)
    map_crosswalks=map_crosswalks, # (2, 4, 100, 3)
    object_type=object_type, # (2) — 두 agent 의 object_type
    region_6=region_6, # (6, 2) — point cluster (point_dir 없으면 zeros)
    object_index=object_index, # (2) — 두 agent 의 id (Waymo internal)
    current_state=self.current_xyzh[0], # (4) — ego[0] 의 current x, y, z, heading
    gt_future_states=gt # (2, 80, 5)
)
```

#### 2.3. multi-process pool

```python
parser.add_argument('--use_multiprocessing', action="store_true")
parser.add_argument('--processes', type=int, default=8, help='multiprocessing process num')

data_files = glob.glob(args.load_path+'/*')

if args.use_multiprocessing:
    with Pool(processes=args.processes) as p:
        p.map(multiprocessing, data_files) # 1 shard = 1 worker task
```

**default 8 worker** (open_loop 의 default `os.cpu_count` 와 다름). 명시 안 하면 8 만 사용 — 큰 vCPU 시스템에서 비효율. chain 에서 `--processes 130` 명시.

#### 2.4. skip 로직

```python
for pairs in self.sdc_ids_list:
    sdc_ids, interesting = pairs[0], pairs[1]
    inter = 'interest' if interesting==1 else 'r'
    filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
    if os.path.exists(filename):
        continue # multi-pod resume
    # process ...
    np.savez(filename, ...)
```

---

### 3. output structure (.npz file)

#### 3.1. file 명 pattern

`<scenario_id>_<sdc_id1>_<sdc_id2>_{interest|r}.npz`

예시:
- `f3f55cf64033141c_12345_67890_interest.npz` (interesting pair)
- `f3f55cf64033141c_12345_99999_r.npz` (random pair)

#### 3.2. file 1개의 9 field

| field | shape | dtype | 의미 |
|---|---|---|---|
| `ego` | (2, 11, 9) | float32 | 2 agent (pair) × 11 hist step × 9 attr |
| `neighbors` | (32, 11, 9) | float32 | 32 neighbor × 11 hist × 9 |
| `map_lanes` | (2, 6, 300, 17) | float32 | 2 agent × 6 lane × 300 point × 17 attr |
| `map_crosswalks` | (2, 4, 100, 3) | float32 | 2 agent × 4 crosswalk × 100 point × 3 |
| `object_type` | (2) | int | 2 agent 의 object_type (1=vehicle, 2=pedestrian, ...) |
| `region_6` | (6, 2) | float32 | 6-mode region anchor (point_dir 없으면 zeros) |
| `object_index` | (2) | int | 2 agent 의 Waymo internal id |
| `current_state` | (4) | float32 | ego[0] 의 current pose [x, y, z, heading] |
| `gt_future_states` | (2, 80, 5) | float32 | 2 agent × 80 future step × [x, y, h, vx, vy] |

#### 3.3. ego / neighbors 의 9 attribute (open_loop 와 동일)

[`07a-open_loop_validation.md`](07a-open_loop_validation.md) 의 3.2 참조.

#### 3.4. map_lanes 의 17 attribute (open_loop 의 16 + 1)

| index | meaning |
|---|---|
| 0,1,2 | centerline xyh |
| 3,4,5 | left bdry xyh |
| 6,7,8 | right bdry xyh |
| 9 | speed_limit (m/s) |
| 10 | lane.type |
| 11 | left_bdry_type |
| 12 | right_bdry_type |
| 13 | traffic_light state |
| 14 | interpolating |
| 15 | stop_sign |
| **16** | **stop_point (NEW vs open_loop)** |

#### 3.5. region_6 의 의미

paper 의 region anchor — model 의 multi-mode prediction 시 anchor 로 사용. 6 mode × 2 (xy).
- point_dir 없으면 zeros (point cluster 미사용)
- point_dir 있으면 vehicle/pedestrian/cyclist 별 6/32/64 mode 의 cluster center 사용 (`build_points` 에서 pickle load)

#### 3.6. file 수 — Done 측정

- 150 shard × 평균 ~6 pair/scene × 평균 ~96 scenario/shard ≈ **86,400 file**
- 실측: **86,958 file** ← 정확히 매칭

---

### 4. 학습 사용 (`interaction_prediction/train.py` 의 `valid_epoch`, DDP)

#### 4.1. DataLoader

```python
class InteractionDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*.npz")
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return (data['ego'], data['neighbors'], data['map_lanes'],
                data['map_crosswalks'], data['object_type'], data['region_6'],
                data['object_index'], data['current_state'], data['gt_future_states'])
```

DDP DataLoader: `DistributedSampler` 가 4 GPU rank 별로 batch 분배.

#### 4.2. valid_epoch logic (DDP)

```python
def valid_epoch(rank, world_size, model, valid_loader):
    model.eval
    epoch_loss = []
    epoch_metrics = []
    
    for batch in valid_loader:
        inputs = {
            'ego_state': batch[0].to(rank),
            'neighbors_state': batch[1].to(rank),
            'map_lanes': batch[2].to(rank),
            'map_crosswalks': batch[3].to(rank),
            'object_type': batch[4].to(rank),
            'region_6': batch[5].to(rank),
        }
        gt = batch[8].to(rank) # (B, 2, 80, 5)
        
        with torch.no_grad:
            outputs = model(inputs) # K-mode joint prediction
            loss, results = joint_loss(outputs, gt)
            metrics = interaction_metrics(results, gt)
        
        epoch_loss.append(loss.item)
        epoch_metrics.append(metrics)
    
    # all_reduce metric across GPUs
    ...
    return mean_loss, [planning_ade, collision_rate, minADE, minFDE, miss_rate]
```

#### 4.3. GameFormer model (interaction 변형)

기본 GameFormer + interaction 특화 head:
- **Joint encoder**: 2 agent + 32 neighbor + map → joint context
- **Level-k decoder**: 두 agent 사이의 game theoretic interaction (각 agent 가 다른 agent 의 future 예측 + 본인 plan 결정, K=4 level)
- **Multi-mode output**: K=4 mode (paper 의 K) — 각 mode 가 다른 trajectory 가능성
- output: 2 agent × K mode × 80 step × 5 attr + mode probability

#### 4.4. metric

- **Planning ADE**: 두 agent 의 평균 displacement error (over time, 모든 mode 의 weighted average)
- **Collision rate**: 두 agent 의 future trajectory 가 collide 하는 비율 (paper 의 Table 2)
- **minADE**: K mode 중 ground truth 와 가장 가까운 mode 의 ADE
- **minFDE**: 같은 mode 의 final step displacement error
- **miss rate**: minFDE > 2.0 m 인 sample 의 비율

paper baseline (Table 2 K=4):
- Planning ADE: **0.8329**
- Collision: **0.0198**

paper baseline (Table 3):
- minADE: **0.79**
- minFDE: **1.85**
- miss rate: **0.30**

acceptance: paper 값 대비 ±10~15% margin.

---

### 5. 현재 진행 상태 — **Done**

| 단계 | status |
|---|---|
| raw download | Done (150 file, 38 GB) — A5000 (이미 destroyed) 의 작업 |
| preprocess | **Done** (86,958 file, A5000 130 worker 진행 → network volume 보존) |
| output sync to S3 | Done (network volume `processed/interaction/valid/` 에 보존) |
| 학습 valid_epoch | 4×H100 학습 시 매 epoch 별 자동 |

A5000 destroy 됐지만 결과는 network volume 에 그대로. 학습 시 cross-region S3 sync 로 새 pod 에서 사용.

---

### 6. 관련

- **** (이 sub-task)
- (parent story)
- (training, 같은 preprocess script)
- (학습 ticket — interaction multi-GPU train)
- code: `interaction_prediction/data_process.py`, `interaction_prediction/train.py`, `model/GameFormer.py`, `utils/interaction_train_utils.py` (추정), `utils/data_utils.py`
- doc: [`07-data_pipeline.md`](07-data_pipeline.md), [`02-workflow.md`](02-workflow.md), [`06-full_training_plan.md`](06-full_training_plan.md)
