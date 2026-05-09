# 02c. 같은 model 의 specialization — 출력단은 동일, loss + size + index 약속만 다름

interaction_prediction 과 open_loop_planning 이 같은 `class GameFormer` 코드를 공유하면서 어떻게 task-specific behavior 가 나오는지 자세히.

## 1. Model 자체는 100% 동일 — 단 size 만 다름

같은 [`class GameFormer`](../../model/GameFormer.py) 를 두 task 가 모두 사용. 차이는 **constructor params 만**:

```python
# open_loop_planning/train.py:128
gameformer = GameFormer(
    modalities=6,              # M=6 modal trajectory per agent
    neighbors_to_predict=10,   # ego + 10 neighbor 의 trajectory
    future_len=50,             # 50 step (5 sec @ 10Hz)
    decoder_levels=args.levels # K=4 (paper)
)

# interaction_prediction/train.py:177 (defaults)
model = GameFormer(
    modalities=6,              # M=6 (paper Joint variant)
    neighbors_to_predict=1,    # ego + 1 neighbor (Waymo Joint Prediction)
    future_len=80,             # 80 step (8 sec @ 10Hz)
    decoder_levels=3           # K=3
)
```

| 항목 | open_loop | interaction | 의미 |
|---|---|---|---|
| modalities (M) | 6 | 6 | 같음 (paper Joint variant) |
| **neighbors_to_predict** | **10** | **1** | output 의 agent 수 (N=11 vs N=2) |
| **future_len** | **50** | **80** | output trajectory 의 시간 길이 |
| **decoder_levels (K)** | **4** | **3** | game-theoretic iteration 횟수 |
| encoder_layers | 6 | 6 | 같음 |

### 1.1. neighbors_to_predict = 10 vs 1 의 의미

이 param 이 **decoder 의 출력 크기** 를 정함:

```python
# Decoder.forward
N = self._neighbors + 1  # = 11 for open_loop, = 2 for interaction
results = [self.initial_stage(i, ...) for i in range(N)]
```

- **open_loop**: 11 agent (ego + 10 neighbor) trajectory 모두 출력. `(B, 11, M=6, T=50, 4)`
- **interaction**: 2 agent (ego + 1 neighbor pair) trajectory 출력. `(B, 2, M=6, T=80, 4)`

interaction 의 N=2 는 Waymo Joint Prediction Challenge 의 표준 — *2 차량 pair* 만 예측.

### 1.2. future_len 50 vs 80 의 의미

prediction horizon (예측 길이):

- **open_loop**: 50 step = **5 초** (planning 은 짧은 horizon)
- **interaction**: 80 step = **8 초** (Waymo challenge 표준)

`GMMPredictor` 의 출력 shape 결정:

```python
self.gaussian = nn.Linear(512, future_len*4)  # 50*4 vs 80*4
```

### 1.3. decoder_levels K=4 vs K=3 의 의미

game-theoretic iteration 횟수 (자세한 의미는 [02d-game_theoretic_interpretation](02d-game_theoretic_interpretation.md) 참조):

- **open_loop**: K=4 (planning 은 더 깊은 reasoning 필요)
- **interaction**: K=3 (challenge 표준)

각 level 마다 `InteractionDecoder` 1 개 추가 → model 의 parameter 수 다름.

## 2. Encoder + Decoder 구조 동일

```
                      ┌────────────────────────┐
                      │     Encoder            │
                      │                        │
input dict ──────────▶│ - AgentEncoder (LSTM)  │──┐
{                     │ - LaneEncoder (PointNet)│  │
  ego_state,          │ - CrosswalkEncoder      │  │
  neighbors_state,    │ - TransformerEncoder    │  │
  map_lanes,          │   (6 layer fusion)      │  │
  map_crosswalks,     └────────────────────────┘  │
  ref_line  ←─ open_loop 만 있지만 무시됨            │
}                                                  │
                                                   ▼
                      ┌────────────────────────┐
                      │     Decoder            │
                      │                        │
                      │  ┌──────────────────┐  │
                      │  │ InitialDecoder   │  │ (level 0)
                      │  │  for each agent  │  │
                      │  └──────────────────┘  │
                      │           ↓             │
                      │  ┌──────────────────┐  │
                      │  │ InteractionDecoder│  │ (level 1)
                      │  └──────────────────┘  │
                      │           ↓             │
                      │  ┌──────────────────┐  │
                      │  │ InteractionDecoder│  │ (level 2)
                      │  └──────────────────┘  │
                      │           ↓             │
                      │     ... (K levels)      │
                      └─────────────┬──────────┘
                                    │
                                    ▼
output dict
{
  level_0_interactions: (B, N, M, T, 4),
  level_0_scores: (B, N, M),
  level_1_interactions: ...,
  ...
  level_K_interactions: ...,
  level_K_scores: ...
}
```

**같은 코드 path**, 같은 layer types, 같은 attention block. parameter 개수만 다름.

## 3. 출력단 (head) 도 100% 동일

`InitialDecoder` 와 `InteractionDecoder` 모두 마지막에 `GMMPredictor` 사용:

```python
# model/modules.py:123
class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        self.gaussian = nn.Sequential(
            nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(512, future_len*4)  # mu_x, mu_y, log_sig_x, log_sig_y
        )
        self.score = nn.Sequential(
            nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(64, 1)  # mode confidence
        )
```

**ego 와 neighbor 둘 다 같은 GMMPredictor 사용** — 출력단에 ego/neighbor 차별 없음. trajectory shape 도 모두 같음 `(M, T, 4)`.

즉 model 입장에서는 **agent 가 ego 인지 neighbor 인지 모름** — 그냥 N agent 의 trajectory 를 모두 출력.

## 4. 그럼 어디서 ego 와 neighbor 가 구분되나?

3 군데:

### 4.1. Index 약속 (data 차원)

코드 약속: **agent index 0 = ego**, 1.. = neighbors.

```python
# Decoder.forward
results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
#                              ↑
#                     i=0 이면 ego, i=1..N-1 이면 neighbor
```

model 은 i 의 의미를 모름. 학습할 때 loss 가 i=0 을 ego 로 처리한다.

### 4.2. Loss function 에서 ego 차별

[`utils/open_loop_train_utils.py`](../../utils/open_loop_train_utils.py) 의 `level_k_loss`:

```python
def level_k_loss(outputs, ego_future, neighbors_future, neighbors_future_valid):
    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions']  # (B, N=11, M=6, T=50, 4)
        plan = trajectories[:, :1]                          # (B, 1, M=6, T=50, 4)  ← ego 의 planning
        predictions = trajectories[:, 1:] * neighbors_future_valid[..., None]  # (B, 10, M=6, T=50, 4) ← neighbor prediction
        trajectories = torch.cat([plan, predictions], dim=1)
        
        gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)
        il_loss, future = imitation_loss(trajectories, scores, gt_future)
        ...
```

여기서:
- `trajectories[:, :1]` = ego (index 0) 의 planning trajectory
- `trajectories[:, 1:]` = neighbor (index 1..) 의 prediction
- 둘 다 같은 `imitation_loss` 통과하지만 ego 는 winner-takes-all 로 1 mode 만 update

### 4.3. Data preprocessing 의 ref_line normalization

`open_loop_planning/data_process.py:319` 의 `normalize_data`:

```python
def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ref_line, ground_truth, viz=False):
    center = ego[10, :2]                    # ego 의 현재 위치
    angle = ego[10, 2]                      # ego 의 현재 heading
    
    # 모든 trajectory 와 map 을 ego-centric local frame 으로 회전 + 평행이동
    ego = ego_norm(ego, center, angle)
    neighbors = neighbors_norm(neighbors, center, angle)
    map_lanes = map_lanes_norm(map_lanes, center, angle)
    map_crosswalks = map_crosswalks_norm(map_crosswalks, center, angle)
    ref_line = ref_line_norm(ref_line, center, angle).astype(np.float32)
    ground_truth = ground_truth_norm(ground_truth, center, angle)
```

**ref_line 은 input dict 에 들어가지만 model 은 사용 안 함**. 단 normalization 시 frame 결정에 사용됨 — 즉 모든 agent 의 trajectory + map 이 *ego 의 ref_line 기준 frame* 으로 회전됨.

## 5. ref_line 의 진짜 역할

```
Raw WOMD scenario (world coordinates)
           │
           ▼
data_process.py:
  - ego 현재 위치/heading 으로 모든 데이터 회전
  - ref_line 도 같이 normalize
  - npz 로 save
           │
           ▼
DataLoader: load .npz
           │
           ▼
train.py:
  inputs['ref_line'] = batch[4]   ← input dict 에 들어가지만
           │
           ▼
model(inputs)
  - Encoder 가 ego_state, neighbors_state, map_* 만 사용
  - ref_line 은 무시
           │
           ▼
output: trajectories
           │
           ▼
imitation_loss(trajectories, gt) ← ref_line 은 loss 에도 사용 안 됨
```

**놀라운 점**: open_loop 에서 ref_line 은 npz file + input dict 에 있지만 model 도 loss 도 직접 사용 X.

ref_line 의 진짜 역할:
1. **data_process 에서 ego 의 미래 trajectory (ground_truth) 를 ref_line 따라가도록 sampling/normalize**
2. open_loop 은 GT 자체가 "reference path 따라가는 ego trajectory" 라서 model 이 GT 만 보고 학습하면 자동으로 ref_line aligned

즉 ref_line 은 *학습 데이터의 GT 를 정의하는 implicit constraint*. model 이 이 GT 를 따라하도록 학습하면 결과적으로 ref_line aligned planning 이 나옴.

## 6. 두 task 가 진짜로 다른 점 정리

| 항목 | open_loop | interaction | 같음/다름 |
|---|---|---|---|
| `class GameFormer` 코드 | 동일 file 에서 import | 동일 | 같음 |
| Encoder + Decoder 구조 | 동일 layer types | 동일 | 같음 |
| GMMPredictor (출력단) | 동일 | 동일 | 같음 |
| modalities (M) | 6 | 6 | 같음 |
| **neighbors_to_predict** | **10** | **1** | 다름 (size) |
| **future_len** | **50** | **80** | 다름 (output T) |
| **decoder_levels (K)** | **4** | **3** | 다름 (depth) |
| Total trainable params | (size 차이로) 다름 | — | 다름 |
| **Loss function** | `level_k_loss` (planning + prediction + interaction consistency) | `interaction_loss` (pure GMM NLL) | 다름 |
| **dataset** | `training_20s` | `training` | 다름 |
| **ego/neighbor 의 처리** | loss 에서 ego (idx=0) 따로 처리 | 모든 agent 평등 | 다름 (loss 차이) |
| ref_line 입력 | input dict 에 있지만 model 사용 X | 없음 | 다름 (data 차이) |
| ref_line 의 실제 영향 | data_process 에서 GT trajectory 가 ref_line aligned | — | 다름 (data 차이) |

## 7. 정리

같은 모델, 출력단도 같음. 다른 건 4 가지:

1. **Size (constructor params)**
   - neighbors_to_predict: 10 vs 1
   - future_len: 50 vs 80
   - decoder_levels: 4 vs 3
   - → Total parameter 수 와 forward pass 의 tensor shape 다름

2. **Loss function**
   - open_loop: `level_k_loss` — ego 의 planning 따로 + neighbor prediction + level 간 consistency
   - interaction: `interaction_loss` — 모든 agent 평등한 GMM NLL

3. **Dataset 과 GT 정의**
   - open_loop: `training_20s` (긴 horizon, ego 의 reference path 가 사전에 결정된 scenario)
   - interaction: `training` (general scene, ego/neighbor 평등)

4. **Index 약속**
   - open_loop: idx=0 = ego planning, idx=1.. = neighbor prediction (loss 가 분리 처리)
   - interaction: 모든 idx 가 평등한 prediction

**model 자체에는 ego head / neighbor head 같은 분리 없음.** 학습 시 loss 가 "ego 는 첫 idx" 라는 약속 으로 ego 를 차별 학습하게 만듦. 즉 *model 은 모르고 loss 가 안다*.

이 design 의 장점:
- 모델 코드 재사용 (engineering 단순화)
- task-specific behavior 가 loss + data 에 격리 → 새 task 추가 시 model 변경 없이 loss + data 만 정의
