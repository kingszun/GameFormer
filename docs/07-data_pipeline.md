## 07 - data pipeline (overview)

GameFormer 의 데이터 흐름 정리 — WOMD raw 부터 학습까지 전 과정.

자세한 내용은 각 subset 별 doc 참조:
- [`07a-open_loop_validation.md`](07a-open_loop_validation.md) — validation subset (KAK-50)
- [`07b-open_loop_training_20s.md`](07b-open_loop_training_20s.md) — training_20s subset (KAK-51)
- [`07c-interaction_validation.md`](07c-interaction_validation.md) — validation_interactive subset (KAK-52)
- [`07d-interaction_training.md`](07d-interaction_training.md) — training subset (KAK-53)

관련: KAK-49 (story), KAK-1 (epic).

---

### 0. 왜 preprocess 가 필요한가

WOMD (Waymo Open Motion Dataset) 의 raw 는 **TFRecord + protobuf** 형식. 이걸 그대로 학습 시 매 batch 마다:

1. TFRecord container 에서 byte 추출
2. protobuf deserialize (`scenario_pb2.Scenario.ParseFromString`)
3. Scenario 내부에서 SDC 식별, history/future 자르기, neighbor 식별, map 추출, normalize

→ 너무 cost 큼 (deserialize 가 GPU step 보다 느려짐).

**해결 — 한 번 preprocess 후 numpy `.npz` 로 저장**:
- 학습 시 `.npz` 만 load → `dict` 로 즉시 사용
- 모든 cost (deserialize + feature 추출 + normalize) 가 preprocess 한 번에 끝남
- 단점: disk usage 증가, 한 번 preprocess 시간 (cluster 재학습 시 불필요한 재처리 가능)

이 trade-off 가 motion forecasting / planning 학습 시 표준 pattern.

---

### 1. WOMD v1.2.1 의 raw 구조

#### 1.1. bucket / shard

GCS bucket: `gs://waymo_open_dataset_motion_v_1_2_1/`

| subset | path | shard | size |
| --- | --- | --- | --- |
| validation | `uncompressed/scenario/validation/*` | 150 | 39 GB |
| training_20s | `uncompressed/scenario/training_20s/*` | **344** (description 의 252 가 아니라 실측) | 30 GB |
| validation_interactive | `uncompressed/scenario/validation_interactive/*` | 150 | 38 GB |
| training | `uncompressed/scenario/training/*` | 1000 | 425 GB |

각 shard = `<subset>.tfrecord-NNNNN-of-MMMMM` (TFRecord container).

1 shard 당 평균 **250~290 scenario** (총 scenario 수 = shard × 평균).

#### 1.2. Scenario proto schema (1 scenario = 9.1초 driving scene)

Waymo proto: `waymo_open_dataset.protos.scenario_pb2.Scenario`

```
Scenario {
    string scenario_id              # unique id (예: "f3f55cf64033141c")
    repeated float timestamps_seconds  # 91 timestep × 0.1s = 9.1초
    int32 current_time_index        # 보통 10 (1초 시점, history 끝 + future 시작 경계)
    repeated Track tracks           # scene 의 모든 agent (vehicle/pedestrian/cyclist)
    repeated DynamicMapState dynamic_map_states  # 91 timestep 별 traffic light 상태
    repeated MapFeature map_features  # static map (lane, road_line, road_edge, ...)
    int32 sdc_track_index           # Self-Driving Car (Waymo 차량) 의 tracks index
    repeated int32 objects_of_interest  # paper 가 평가하는 관심 object id
    repeated RequiredPrediction tracks_to_predict  # Waymo 가 prediction 요청한 agent + difficulty
}
```

##### Track

```
Track {
    int32 id                        # agent 고유 id (모든 timestep 공유)
    ObjectType object_type          # TYPE_UNSET=0, TYPE_VEHICLE=1, TYPE_PEDESTRIAN=2, TYPE_CYCLIST=3, TYPE_OTHER=4
    repeated ObjectState states     # 91 timestep 의 state
}
```

##### ObjectState (timestep 1개의 agent state)

```
ObjectState {
    float center_x, center_y, center_z  # world frame 위치 (m)
    float length, width, height         # bounding box 크기 (m)
    float heading                        # world frame 방향 (rad)
    float velocity_x, velocity_y         # m/s
    bool valid                           # 그 timestep 에서 추적 가능 여부
}
```

##### MapFeature

각 MapFeature 는 static map element 1개. Waymo 의 oneof field 로 type 분기:

```
MapFeature {
    int32 id
    oneof feature_data {
        LaneCenter lane            # 차선 중앙선
        RoadLine road_line         # 도로 line (점선, 실선 등)
        RoadEdge road_edge         # 도로 edge (curb)
        StopSign stop_sign         # 정지 표지
        Crosswalk crosswalk        # 횡단보도
        SpeedBump speed_bump       # 과속방지턱
        Driveway driveway          # 진입로 (WOMD v1.2.1 신규)
    }
}
```

**Lane (가장 중요)**:
```
LaneCenter {
    LaneType type                  # TYPE_UNDEFINED=0, TYPE_FREEWAY=1, TYPE_SURFACE_STREET=2, TYPE_BIKE_LANE=3
    float speed_limit_mph
    repeated MapPoint polyline     # lane 중앙선 점 (약 1m 간격)
    repeated BoundarySegment left_boundaries, right_boundaries  # 양쪽 boundary (RoadLine/RoadEdge id 와 시작/끝 index)
    repeated int32 entry_lanes, exit_lanes  # 연결된 lane id
    bool interpolating             # interpolated lane 여부
}
```

**DynamicMapState** (timestep 1개):
```
DynamicMapState {
    repeated TrafficSignalLaneState lane_states  # lane 별 신호 상태
}

TrafficSignalLaneState {
    int32 lane                     # lane id
    LaneState state                # UNKNOWN/ARROW_STOP/ARROW_CAUTION/ARROW_GO/STOP/CAUTION/GO/FLASHING_STOP/FLASHING_CAUTION
    MapPoint stop_point            # stop line 위치
}
```

**RequiredPrediction**:
```
RequiredPrediction {
    int32 track_index              # tracks[] 의 index
    Difficulty difficulty          # LEVEL_1 / LEVEL_2
}
```

##### 4 raw subset 의 차이

| subset | 특성 |
| --- | --- |
| validation | 일반 driving scene (open_loop planning eval 용) |
| training_20s | 20초 (200 step?) 길이 → 더 긴 horizon training. **단 GameFormer 는 91 step 만 사용** (script 가 처음부터 91 step만 처리) |
| validation_interactive | `objects_of_interest` 에 두 agent id 가 marked 됨 (interactive scene) |
| training | 일반 driving scene + interactive scene 모두 포함 (interaction prediction train 용) |

→ open_loop 와 interaction 의 raw 가 **따로 다름**. preprocess script 도 따로.

---

### 2. 두 preprocess pipeline

GameFormer 는 두 task 학습:

#### 2.1. open_loop planning (`open_loop_planning/data_process.py`)

**목표**: SDC 의 미래 plan + 주변 agent prediction.

**입력**: validation / training_20s raw

**처리 단위**: 1 scenario × **1 timestep** = 1 .npz file
- 각 scenario 의 91 timestep 중 5 step 마다 1 file (`range(10, 41, 5)` = 7 timestep / scenario)
- 1 scenario → ~7 file (training/validation 시 augmentation 용)

**핵심 함수** (`DataProcess` class):
1. `build_map(map_features, dynamic_map_states)` — map static + dynamic 데이터 구조 구축
2. `ego_process(sdc_id, timestep, tracks)` — SDC 의 11 step history 추출
3. `route_process(sdc_id, timestep, current_xyh, tracks)` — reference path 1000 waypoint 추출
4. `neighbors_process(sdc_id, timestep, tracks)` — top 10 neighbor agent 식별
5. `map_process(traj, timestep, type)` — 각 agent 별 6 lane × 100 point + 4 crosswalk × 100 point
6. `ground_truth_process(sdc_id, timestep, tracks)` — 50 future step
7. `normalize_data(...)` — ego frame 으로 좌표 변환
8. `np.savez(...)` — 6 field 저장

**output**: per-timestep `.npz`, 6 field, **size 512,088 bytes uniform**

#### 2.2. interaction prediction (`interaction_prediction/data_process.py`)

**목표**: 두 agent 의 joint future trajectory.

**입력**: validation_interactive / training raw

**처리 단위**: 1 scenario × **1 pair** = 1 .npz file
- pair 종류: `interest` (Waymo marked interactive pair, paper 평가 metric source) + `r` (random pair, augmentation)
- 1 scenario → 평균 ~5~10 pair file

**핵심 함수**:
1. `build_map(...)` — open_loop 와 동일
2. `interactive_process(tracks_list, interact_list, tracks)` — interesting + random pair 식별
3. `ego_process(sdc_ids, tracks)` — **2 agent** 의 11 step history (open_loop 의 1 agent 와 다름)
4. `neighbors_process(sdc_ids, tracks)` — pair 외의 다른 agent
5. `map_process(traj)` — 6 lane × **300 point** × 17 attribute (open_loop 의 100 point × 16 보다 더 dense)
6. `ground_truth_process(sdc_ids, tracks)` — 두 agent 의 future
7. `normalize_data(...)` + `np.savez(...)`

**output**: per-pair `.npz`, 9 field

---

### 3. processed data 4 종류

| 출력 path | source raw | preprocess script | 사용처 |
| --- | --- | --- | --- |
| `processed/open_loop/valid/` | validation | open_loop_planning/data_process.py | `open_loop_planning/train.py` 의 valid_epoch |
| `processed/open_loop/train/` | training_20s | open_loop_planning/data_process.py | `open_loop_planning/train.py` 의 train_epoch |
| `processed/interaction/valid/` | validation_interactive | interaction_prediction/data_process.py | `interaction_prediction/train.py` 의 valid_epoch (DDP) |
| `processed/interaction/train/` | training | interaction_prediction/data_process.py | `interaction_prediction/train.py` 의 train_epoch (DDP) |

#### file 명 pattern

- open_loop: `<scenario_id>_<timestep>.npz` (예: `f3f55cf64033141c_25.npz`)
- interaction: `<scenario_id>_<sdc_id1>_<sdc_id2>_{interest|r}.npz`

#### file size

- open_loop: 512,088 bytes uniform (모든 file 같은 size — partial corruption 검출 가능)
- interaction: 가변 (pair 의 neighbor 수, map element 따라 다름)

---

### 4. 두 학습 script

#### 4.1. open_loop_planning/train.py

**input**: `processed/open_loop/{train, valid}` (DataLoader 가 .npz 모두 load)

**model**: `model.GameFormer.GameFormer` (level-k game theory based encoder + decoder)

**loss**: `level_k_loss(level_k_outputs, ego_future, neighbors_future, neighbors_future_valid)` (utils/open_loop_train_utils.py)

**metric** (motion_metrics):
- plannerADE: SDC 의 평균 displacement error
- plannerFDE: SDC 의 마지막 step displacement error
- predictorADE: neighbor 의 평균 ADE
- predictorFDE: neighbor 의 평균 FDE

**hyperparameter** (KAK-34 v0.3, B option):
- batch 64, lr 2e-4, epoch 20
- optimizer: Adam (default in train.py)
- grad clip: 5.0
- file_descriptor sharing strategy (KAK-30 patch)

**GPU**: 1×H100

#### 4.2. interaction_prediction/train.py

**input**: `processed/interaction/{train, valid}` (DataLoader)

**model**: `model.GameFormer.GameFormer` (joint prediction head, level-k game theory)

**loss**: joint trajectory loss (interaction 특화)

**metric**:
- Planning ADE / Planning FDE
- Collision rate
- minADE / minFDE / miss rate (over K modes)

**hyperparameter** (KAK-34 v0.3, D option):
- batch 32 per GPU × 4 GPU = **128 effective**
- lr 1e-4 (paper 그대로, batch 32 per GPU 동일)
- epoch 30 (paper 그대로)
- DDP via torch.distributed.launch
- `--local_rank` + `--local-rank` 둘 다 받음 (KAK patch, torch 2.x DDP launcher 호환)

**GPU**: 4×H100 (DDP)

---

### 5. paper baseline (KAK-34 v0.3)

| 학습 | metric | paper 값 | acceptance margin |
| --- | --- | --- | --- |
| open_loop | plannerADE | Table 1 참조 | ±10~15% |
| open_loop | plannerFDE | Table 1 참조 | ±10~15% |
| interaction | Planning ADE | 0.8329 (Table 2, K=4) | ±10~15% |
| interaction | Collision rate | 0.0198 (Table 2) | ±10~15% |
| interaction | minADE | 0.79 (Table 3) | ±10~15% |
| interaction | minFDE | 1.85 (Table 3) | ±10~15% |
| interaction | miss rate | 0.30 (Table 3) | ±10~15% |

---

### 6. 현재 상태 (26-05-09 기준)

| processed data | status | 위치 |
| --- | --- | --- |
| `processed/interaction/valid/` | **Done** (86,958 file) | network volume svnweu0of5 |
| `processed/open_loop/valid/` | 진행 중 (RTX 3090 chain v2, 122K+ file) | RTX 3090 local + S3 sync |
| `processed/open_loop/train/` | pending (chain v2 step 4, prefetch 339/344) | - |
| `processed/interaction/train/` | pending (chain v2 step 7, prefetch 진행 중) | - |

---

### 7. 이번 진행에서 학습한 함정 + 검증 fact

- **partial file 검출**: open_loop file size 모두 512,088 bytes uniform → `find ... ! -size 512088c -delete` 으로 partial 폐기 가능
- **shard 수 추정 X**: training_20s 의 shard 수 = **344** (description 의 252 X). chain script verify 시 `aws s3 ls | wc -l` 으로 동적 listing
- **multi-pod chain skip**: data_process.py 의 skip patch (KAK-41/44) 활용 — 이미 file 있으면 즉시 다음 scene 으로
- **download throughput** (cross-region S3 endpoint, EU-CZ-1 ↔ US-IL-1, parallel xargs cp -P 32): **133~258 MB/s**
- **preprocess throughput** (RTX 3090 6 GPU host, cgroup 163.2 vCPU, 130 worker): **226 file/sec** (~13,580 file/min)
- **container disk resize destructive**: `runpodctl pod update --container-disk-in-gb` 시 모든 data 손실 → 새 pod 만들기 권장
- **aws s3 cp --recursive pagination bug**: `aws s3 ls + xargs cp` 로 우회

---

### 8. 관련 문서 / ticket

- [docs/02-workflow.md](02-workflow.md) — 전체 workflow (preprocess + train script 호출 패턴)
- [docs/06-full_training_plan.md](06-full_training_plan.md) — 학습 plan v0.3 (KAK-34)
- [docs/04-smoke_results.md](04-smoke_results.md) — 3060 smoke test 결과
- KAK-1 (Epic), KAK-49 (story), KAK-50/51/52/53 (sub-task), KAK-41 (open_loop full training), KAK-44 (interaction multi-GPU train)
