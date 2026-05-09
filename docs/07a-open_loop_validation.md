## 07a - validation subset 상세 (open_loop planning valid)

KAK-50 의 doc. WOMD `validation` raw → `processed/open_loop/valid/` → `open_loop_planning/train.py` 의 valid_epoch 흐름 전 과정.

전체 overview 는 [`07-data_pipeline.md`](07-data_pipeline.md) 참조.

---

### 1. raw 의 구조

#### 1.1. 위치 / shard

- bucket: `gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation/`
- file 수: **150** (`validation.tfrecord-00000-of-00150` ~ `00149-of-00150`)
- 1 shard size: 평균 ~260 MB
- 총 size: 39 GB
- mtime: 2024-01-30 (Waymo source publish 시점)

#### 1.2. 1 shard = TFRecord container

TFRecord 는 Google 의 binary container format. 각 record:
```
[length: uint64][length crc32: uint32][data: bytes][data crc32: uint32]
```

`tf.data.TFRecordDataset(filename)` 으로 read → 각 record 의 `data` = `Scenario` proto byte → `scenario_pb2.Scenario.ParseFromString(data.numpy())` 으로 deserialize.

#### 1.3. validation subset 의 특성

- **일반 driving scene** (interactive 제외) — interaction 은 별도 subset (validation_interactive)
- open_loop planning 의 evaluation set
- 1 scenario = 9.1초 (91 timestep × 0.1s)
- 1 shard 당 평균 ~280 scenario → 총 ~42,000 scenario

#### 1.4. Scenario field 별 의미

각 scenario 는 9.1초의 driving 장면 + 그 장면에 등장하는 모든 agent + 주변 map.

| field | type | 의미 |
|---|---|---|
| `scenario_id` | string | 고유 id (예: "f3f55cf64033141c") |
| `timestamps_seconds` | float[91] | timestep 별 시각 (0.0~9.0초) |
| `current_time_index` | int | 보통 10 (1초 시점 = history 10 step + future 0 step 의 경계) |
| `tracks` | Track[] | scene 의 모든 agent (vehicle/pedestrian/cyclist), 보통 ~50 agent |
| `dynamic_map_states` | DynamicMapState[91] | timestep 별 traffic light 상태 |
| `map_features` | MapFeature[] | static map (lane, road_line, road_edge, stop_sign, crosswalk, speed_bump, driveway) |
| `sdc_track_index` | int | Self-Driving Car (Waymo 차량) 의 tracks index |
| `objects_of_interest` | int[] | 관심 object id (validation 에서는 비어있을 수도) |
| `tracks_to_predict` | RequiredPrediction[] | Waymo 가 prediction 요청한 agent (보통 1~8) |

##### Track 1개 (예시 — SDC)

```
Track {
    id: 12345
    object_type: TYPE_VEHICLE = 1
    states: [
        {center_x: 100.5, center_y: 50.2, length: 4.7, width: 2.1, height: 1.6,
         heading: 0.5, velocity_x: 8.0, velocity_y: 0.1, valid: true},  # timestep 0
        {center_x: 100.6, ...},  # timestep 1
        ... (총 91 timestep)
    ]
}
```

##### MapFeature 1개 (예시 — lane)

```
MapFeature {
    id: 678
    lane: {
        type: TYPE_SURFACE_STREET = 2
        speed_limit_mph: 35.0
        polyline: [{x:100, y:50}, {x:101, y:50}, ...]  # ~50 point (~1m 간격)
        left_boundaries: [{boundary_feature_id: 999, lane_start_index: 0, lane_end_index: 30, boundary_type: BROKEN_WHITE = 1}]
        right_boundaries: [{boundary_feature_id: 1000, ...}]
        entry_lanes: [677]
        exit_lanes: [679, 680]  # 분기 가능
    }
}
```

##### DynamicMapState 1개 (timestep 10 의 신호 상태)

```
DynamicMapState {
    lane_states: [
        {lane: 678, state: GO = 6, stop_point: {x: 105, y: 50}},
        {lane: 679, state: STOP = 4, stop_point: {x: 110, y: 55}}
    ]
}
```

---

### 2. preprocess 절차 (`open_loop_planning/data_process.py`)

#### 2.1. 처리 단위

- **per scenario per timestep = 1 .npz file**
- 1 scenario 91 timestep 중 5 step 마다 1 file: `range(hist_len-1, time_len-future_len, 5)` = `range(10, 41, 5)` = `[10, 15, 20, 25, 30, 35, 40]` = **7 timestep / scenario**

##### 왜 5 step 마다인가
- 데이터 augmentation — 같은 scene 의 다른 시작 시점에서 학습 → variety 증가
- 인접 timestep 은 매우 유사 → 5 step 간격 으로 redundancy 줄임
- 1 scenario 가 7 file → disk + train batch 효율

##### 왜 timestep 10 ~ 40 만
- timestep 10 = `hist_len - 1` = 10 (history 11 step 필요, [0~10])
- timestep 40 = `time_len - future_len - 1` = 90 - 50 = 40 (future 50 step 필요, [41~90])
- 이 범위만 history + future 모두 정상

#### 2.2. 함수 별 처리 (`DataProcess` class)

##### Step 1: `build_map(map_features, dynamic_map_states)`

map_features 를 type 별 dict 로 분리 (한 번만, scene 시작 시):
```python
self.lanes = {map_id → LaneCenter}
self.road_lines = {map_id → RoadLine}
self.road_edges = {map_id → RoadEdge}
self.crosswalks = {map_id → Crosswalk}
self.stop_signs = {map_id → StopSign}
self.speed_bumps = {map_id → SpeedBump}
# WOMD v1.2.1 신규 driveway type → continue (skip, KAK patch)
self.roads = self.road_edges + self.road_lines  # boundary 처리 시 통합
self.traffic_signals = dynamic_map_states  # 91 timestep 별 신호
```

##### Step 2: `ego_process(sdc_id, timestep, tracks)`

SDC 의 history 11 step 추출:
```python
sdc_states = tracks[sdc_id].states[timestep+1-11 : timestep+1]  # 11 step
ego = np.zeros((11, 9))
for i, state in enumerate(sdc_states):
    ego[i] = [state.center_x, state.center_y, state.heading,
              state.velocity_x, state.velocity_y,
              state.length, state.width, state.height,
              ego_type=0]  # SDC type 은 항상 0 (vehicle 이지만 ego 로 marking)
self.current_xyh = (state.center_x, state.center_y, state.heading)  # last (timestep)
return ego  # (11, 9)
```

`self.current_xyh` = 마지막 (현재) state — normalize 시 origin.

##### Step 3: `route_process(sdc_id, timestep, current_xyh, tracks)`

SDC 의 ground truth trajectory 따라 reference path 생성:
```python
gt_path = tracks[sdc_id].states  # 91 step (모든 timestep)
route = find_route(gt_path, timestep, cur_pos, self.lanes, self.traffic_signals)
ref_path = np.array(route)  # (N, 5)
if N < 1000:
    ref_path = np.append(ref_path, repeat(last_waypoint, 1000-N))
return ref_path  # (1000, 5)
```

`find_route` (`utils/route_planning_utils.py`): GT 이동 경로 가까운 lane 추적 → forward direction 의 lane sequence 따라 1000 waypoint 생성. 5 attribute = `[x, y, heading, ?, ?]`.

**왜 reference path?**: GameFormer 가 plan 출력 시 reference (lane 따라가는 default plan) 와의 차이 학습 — 모델이 "lane 그대로 따라갈지 / 차선 변경할지" 학습 하기 쉬워짐.

##### Step 4: `neighbors_process(sdc_id, timestep, tracks)`

timestep 시점에 valid 한 모든 agent 중 SDC 와의 거리로 sort, top 10 추출:
```python
neighbors = {}
for i, track in enumerate(tracks):
    if i != sdc_id and track.states[timestep].valid:
        neighbors[i] = (states[timestep].center_x, states[timestep].center_y)
sorted_neighbors = sorted(neighbors, key=lambda: distance to current_xyh)
neighbors_states = np.zeros((10, 11, 9))
for i, neighbor_id in enumerate(sorted_neighbors[:10]):
    states_i = tracks[neighbor_id].states[timestep+1-11 : timestep+1]
    for j, s in enumerate(states_i):
        if s.valid:
            neighbors_states[i, j] = [s.center_x, s.center_y, s.heading,
                                      s.velocity_x, s.velocity_y,
                                      s.length, s.width, s.height,
                                      tracks[neighbor_id].object_type]
self.neighbors_id = sorted_neighbors[:10]  # ground truth 추출 시 사용
return neighbors_states, self.neighbors_id  # (10, 11, 9)
```

##### Step 5: `map_process(traj, timestep, type)` — ego + 10 neighbor 별로 11번 호출

각 agent 의 위치 기준으로 reference lane 식별 + map 추출:

```python
# 1. agent 의 trajectory 와 가장 가까운 lane 찾기
ref_lane_ids = find_reference_lanes(agent_type, traj, lane_polylines)

# 2. current lane 의 forward direction (exit_lanes) 따라 200m 까지 lane sequence
for curr_lane in ref_lane_ids:
    candidate = depth_first_search(curr_lane, lanes, threshold=200m)
    ref_lanes.extend(candidate)

# 3. (vehicle/cyclist 만) 좌/우 인접 lane 추가
if agent_type != 2:  # 보행자 아님
    neighbor_lanes = find_neighbor_lanes(...)
    forward of neighbor lanes → ref_lanes.extend
    
# 4. 중복 제거
ref_lanes = remove_overlapping_lane_seq(ref_lanes)

# 5. traffic light + stop sign 정보 lane 별로 mapping
traffic_light_lanes = {lane_id → (state, stop_x, stop_y)}
stop_sign_lanes = [lane_id, ...]

# 6. 6 lane × 100 point × 16 attribute 추출
vectorized_map = np.zeros((6, 100, 16))
for i, s_lane in enumerate(ref_lanes[:6]):
    cache_lane = np.zeros((200, 16))  # 200 point 임시 buffer
    for lane in s_lane:
        for point in lane.polyline:
            cache_lane[k, 0:3] = point  # centerline xyh
            cache_lane[k, 3:6] = nearest_point(point, left_boundary)  # left bdry xyh
            cache_lane[k, 6:9] = nearest_point(point, right_boundary)  # right bdry xyh
            cache_lane[k, 9] = speed_limit_mph / 2.237  # m/s
            cache_lane[k, 10] = lane.type
            cache_lane[k, 11] = left_boundary_type
            cache_lane[k, 12] = right_boundary_type
            cache_lane[k, 13] = traffic_light_state
            cache_lane[k, 14] = lane.interpolating
            cache_lane[k, 15] = (lane in stop_sign_lanes)
            k += 1
    vectorized_map[i] = cache_lane[::2]  # 200 → 100 point (2배 downsample)

# 7. 4 crosswalk × 100 point × 3 (xy + ?)
detection = polygon(50m × 40m, agent 앞 방향)
for crosswalk in self.crosswalks:
    if detection.intersects(crosswalk.polygon):
        polyline = polygon_completion(crosswalk.polygon)  # 폐곡선 → polyline
        polyline = polyline[linspace(0, N, num=100)]  # 100 point sample
        vectorized_crosswalks[i, :100] = polyline
```

각 point 의 16 attribute:
| index | meaning |
|---|---|
| 0,1,2 | self_point (centerline x, y, heading) |
| 3,4,5 | left_boundary point (x, y, heading) |
| 6,7,8 | right_boundary point (x, y, heading) |
| 9 | speed_limit (m/s) |
| 10 | lane.type (0~3) |
| 11 | left_boundary_type (BoundaryType + 8 if road_edge) |
| 12 | right_boundary_type |
| 13 | traffic_light state (0~9) |
| 14 | interpolating (bool) |
| 15 | stop_sign (bool) |

return: (6, 100, 16), (4, 100, 3)

##### Step 6: `ground_truth_process(sdc_id, timestep, tracks)`

SDC + 10 neighbor 의 future 50 step:
```python
gt = np.zeros((1+10, 50, 5))  # 1 SDC + 10 neighbor

# SDC future
sdc_future = tracks[sdc_id].states[timestep+1 : timestep+51]  # 50 step
for i, s in enumerate(sdc_future):
    gt[0, i] = [s.center_x, s.center_y, s.heading, s.velocity_x, s.velocity_y]

# Neighbor future (Step 4 의 self.neighbors_id 사용)
for j, neighbor_id in enumerate(self.neighbors_id):
    neighbor_future = tracks[neighbor_id].states[timestep+1 : timestep+51]
    for i, s in enumerate(neighbor_future):
        gt[1+j, i] = [s.center_x, s.center_y, s.heading, s.velocity_x, s.velocity_y]

return gt  # (11, 50, 5)
```

##### Step 7: `normalize_data(...)`

ego frame 으로 좌표 변환 — `self.current_xyh` (마지막 SDC state) 가 origin:

```python
center, angle = self.current_xyh[:2], self.current_xyh[2]

# agent normalize (history + future)
ego[:, :5] = agent_norm(ego, center, angle)
gt[0] = agent_norm(gt[0], center, angle)
for i in range(10):
    if neighbors[i, -1, 0] != 0:  # valid
        neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True)
        gt[i+1] = agent_norm(gt[i+1], center, angle)

# map normalize
for i in range(11):
    for j in range(6):
        lane = map_lanes[i, j]
        if lane[0][0] != 0:
            lane[:, :9] = map_norm(lane, center, angle)  # centerline + left/right bdry
    for k in range(4):
        crosswalk = map_crosswalks[i, k]
        if crosswalk[0][0] != 0:
            crosswalk[:, :3] = map_norm(crosswalk, center, angle)

# ref_line normalize
ref_line = ref_line_norm(ref_line, center, angle)
```

`agent_norm(traj, center, angle)`: traj 의 (x, y) 에서 center 빼고, -angle 만큼 rotate (world → ego frame).

**왜 ego frame?**: model 이 absolute position 에 의존 안 하게 — ego 가 항상 (0, 0) 이고 주변 agent / map 이 ego 기준 상대 위치. invariant 학습 (어디든 동일 model).

##### Step 8: `np.savez(filename, ...)`

```python
filename = f"{save_path}/{scenario_id}_{timestep}.npz"
np.savez(filename,
    ego=ego,                    # (11, 9)
    neighbors=neighbors,        # (10, 11, 9)
    map_lanes=map_lanes,        # (11, 6, 100, 16)
    map_crosswalks=map_crosswalks,  # (11, 4, 100, 3)
    ref_line=ref_line,          # (1000, 5)
    gt_future_states=gt         # (11, 50, 5)
)
```

**file size 항상 512,088 bytes 동일** — 모든 field 의 shape 가 fixed → numpy archive 가 fixed size. partial corruption 검출 가능.

#### 2.3. multi-process pool

```python
parser.add_argument('--use_multiprocessing', action="store_true")
parser.add_argument('--processes', type=int, default=None)  # KAK-41 patch

data_files = glob.glob(args.load_path+'/*')  # 150 shard

if args.use_multiprocessing:
    with Pool(processes=args.processes) as p:
        p.map(multiprocessing, data_files)  # 1 shard = 1 worker task
```

**worker 1개 = shard 1개 처리** (병렬 단위 = shard).

#### 2.4. skip 로직 (KAK-41 patch)

```python
for timestep in range(self.hist_len-1, time_len-self.future_len, 5):
    filename = f"{save_path}/{scenario_id}_{timestep}.npz"
    if os.path.exists(filename):
        continue  # multi-pod resume / restart 대응
    # process ...
    np.savez(filename, ...)
```

RTX 3090 chain v2 의 restart 시 활용 (이미 처리된 file 빠르게 skip).

---

### 3. output structure (.npz file)

#### 3.1. file 1개 의 6 field

| field | shape | dtype | 의미 |
|---|---|---|---|
| `ego` | (11, 9) | float32 | SDC 11 hist step × 9 attr |
| `neighbors` | (10, 11, 9) | float32 | top 10 neighbor × 11 hist × 9 attr |
| `map_lanes` | (11, 6, 100, 16) | float32 | (1 ego + 10 neighbor) × 6 ref lane × 100 point × 16 attr |
| `map_crosswalks` | (11, 4, 100, 3) | float32 | (1 ego + 10 neighbor) × 4 crosswalk × 100 point × 3 attr |
| `ref_line` | (1000, 5) | float32 | reference path 1000 waypoint × 5 attr |
| `gt_future_states` | (11, 50, 5) | float32 | (1 ego + 10 neighbor) × 50 future step × 5 attr |

#### 3.2. ego / neighbors 의 9 attribute

| index | meaning |
|---|---|
| 0 | center_x (m, ego frame) |
| 1 | center_y (m) |
| 2 | heading (rad) |
| 3 | velocity_x (m/s) |
| 4 | velocity_y (m/s) |
| 5 | length (m) |
| 6 | width (m) |
| 7 | height (m) |
| 8 | object_type (int 0~4) |

#### 3.3. map_lanes 의 16 attribute (위 Step 5 참조)

#### 3.4. map_crosswalks 의 3 attribute = (x, y, type)

#### 3.5. ref_line 의 5 attribute = (x, y, heading, ?, ?)

#### 3.6. gt_future_states 의 5 attribute

| index | meaning |
|---|---|
| 0 | center_x (m, ego frame) |
| 1 | center_y |
| 2 | heading |
| 3 | velocity_x |
| 4 | velocity_y |

#### 3.7. 추정 file 수

- 150 shard × 평균 ~280 scenario × 7 timestep ≈ **294,000 file**
- 실제 RTX 3090 chain v2 진행 중: 33분 동안 122,363 file (~42%)

---

### 4. 학습 사용 (`open_loop_planning/train.py` 의 `valid_epoch`)

#### 4.1. DataLoader

`utils/open_loop_train_utils.py` 의 `DrivingData` Dataset class:
```python
class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*.npz")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return data['ego'], data['neighbors'], data['map_lanes'], \
               data['map_crosswalks'], data['ref_line'], data['gt_future_states']
```

DataLoader 가 batch_size 만큼 stack → tuple of tensors (B, ...).

#### 4.2. valid_epoch logic

```python
def valid_epoch(data_loader, model):
    epoch_loss = []
    epoch_metrics = []
    model.eval()  # dropout, batchnorm eval mode

    with tqdm(data_loader, desc="Validation", unit="batch") as epoch:
        for batch in epoch:
            inputs = {
                'ego_state': batch[0].to(args.device),         # (B, 11, 9)
                'neighbors_state': batch[1].to(args.device),   # (B, 10, 11, 9)
                'map_lanes': batch[2].to(args.device),         # (B, 11, 6, 100, 16)
                'map_crosswalks': batch[3].to(args.device),    # (B, 11, 4, 100, 3)
                'ref_line': batch[4].to(args.device)           # (B, 1000, 5)
            }

            ego_future = batch[5][:, 0].to(args.device)        # (B, 50, 5) SDC future
            neighbors_future = batch[5][:, 1:].to(args.device)  # (B, 10, 50, 5)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)  # mask invalid

            with torch.no_grad():
                level_k_outputs = model(inputs)
                loss, results = level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)
                plan = results[:, 0]              # (B, 50, 5) SDC plan
                prediction = results[:, 1:]       # (B, 10, 50, 5) neighbor prediction

            metrics = motion_metrics(plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_loss.append(loss.item())
            epoch_metrics.append(metrics)

    # epoch metric
    plannerADE = mean(metrics[:, 0])
    plannerFDE = mean(metrics[:, 1])
    predictorADE = mean(metrics[:, 2])
    predictorFDE = mean(metrics[:, 3])
    return mean_loss, [plannerADE, plannerFDE, predictorADE, predictorFDE]
```

#### 4.3. GameFormer model forward

`model/GameFormer.py` 의 GameFormer class:
- **Agent encoder** (transformer): ego + 10 neighbor 의 11 hist step → context vector per agent
- **Map encoder** (transformer): 6 lane × 100 point per agent + 4 crosswalk → map context per agent
- **Level-k game theory decoder** (K=4):
  - level 0: 단순 prediction (no interaction)
  - level 1~K: 다른 agent 의 level-(k-1) prediction 을 input 으로 본인의 prediction (recursive)
  - SDC 는 level-K 에서 plan 결정
- output: K-step game theoretic prediction + final plan

#### 4.4. Loss + Metric

**`level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)`** (utils/open_loop_train_utils.py):
- 각 level 별 multi-modal L2 loss (plan + neighbor prediction)
- collision penalty (옵션, agent 들 사이 거리 minimum)
- mode 별 loss 계산 + best mode 선택 (winner-take-all)

**`motion_metrics(plan, prediction, ego_future, neighbors_future, neighbors_future_valid)`**:
```python
plannerADE = mean(L2(plan[:, :, :2], ego_future[:, :, :2]), dim=1)  # over time
plannerFDE = L2(plan[:, -1, :2], ego_future[:, -1, :2])  # last step
predictorADE = mean(L2(prediction[..., :2], neighbors_future[..., :2]) * valid_mask)
predictorFDE = L2(prediction[:, :, -1, :2], neighbors_future[:, :, -1, :2]) * valid_mask
return [plannerADE, plannerFDE, predictorADE, predictorFDE]
```

#### 4.5. valid_epoch 의 역할

- 학습 중 매 epoch 끝나고 (또는 매 N step) valid set 모두 forward (no gradient)
- epoch 평균 loss + metric 계산 → logging.info
- best model checkpoint (validation loss minimize) 저장
- early stopping 가능 (validation loss 가 N epoch 동안 안 줄면 중단)

valid set 은 training 동안 변화 X → epoch 별 model 의 generalization (unseen data 성능) 측정.

---

### 5. paper baseline 비교

paper Table 1 의 open_loop planning baseline (KAK-34 v0.3 의 acceptance):
- plannerADE / plannerFDE
- predictorADE / predictorFDE
- (자세한 paper 값 + acceptance margin 은 docs/06-full_training_plan.md 참조)

acceptance: paper 값 대비 **±10~15% margin**.

---

### 6. 현재 진행 상태

| 단계 | status |
|---|---|
| raw download | Done (150 file, 39 GB) — RTX 3090 의 container disk |
| preprocess | 진행 중 (RTX 3090 chain v2, 130 worker, 226 file/sec, 122K+ file) |
| output sync to S3 | preprocess 끝나면 background sync (cross-region, network volume `processed/open_loop/valid/`) |
| 학습 valid_epoch | preprocess + train preprocess 모두 끝나면 H100 (xatoyjvnespkwc) start + train.py 의 매 epoch 별 자동 |

---

### 7. 관련

- **KAK-50** (이 sub-task)
- KAK-49 (parent story)
- KAK-41 (학습 ticket — open_loop full training)
- code: `open_loop_planning/data_process.py` (preprocess), `open_loop_planning/train.py` (학습), `model/GameFormer.py`, `utils/open_loop_train_utils.py`, `utils/data_utils.py`
- doc: [`07-data_pipeline.md`](07-data_pipeline.md) overview, [`02-workflow.md`](02-workflow.md), [`06-full_training_plan.md`](06-full_training_plan.md)
