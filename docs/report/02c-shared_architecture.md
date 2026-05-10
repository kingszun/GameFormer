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

## 8. Loss 의 정확한 차이 — 무엇이 다른지

같은 이름의 `level_k_loss` 함수가 두 file 에 별개로 정의되어 있고 내부 구성이 다름.

### 8.1. open_loop ([utils/open_loop_train_utils.py:115](../../utils/open_loop_train_utils.py#L115))

```
per level k = 0 ~ K-1:
    loss += imitation_loss(trajectories, scores, gt_future)
           = GMM NLL (모든 agent batch+agent 평균)
           + GMM NLL (ego index 만 batch 평균)     ← ego 한 번 더
           + score CE (label_smoothing=0.2)
    if k >= 1:
        loss += 0.1 * interaction_loss            ← physical proximity penalty
```

`interaction_loss` 는 ego ↔ neighbor + neighbor ↔ neighbor 의 거리 inverse (3m 이내 활성) — potential field penalty. ego 와 neighbor 가 가까운 trajectory 시나리오를 회피하도록 model 을 push.

### 8.2. interaction ([utils/inter_pred_utils.py:101](../../utils/inter_pred_utils.py#L101))

```
per level k = 0 ~ K:
    loss += gmm_loss(trajectories, convs, probs, gt_future)
           = GMM NLL (ego + 1 partner 평등)
           + 추가 weight on [t=29, 49, 79]         ← Waymo metric step
           + 2 * score CE                           ← score weight 2배
```

physical proximity penalty 없음. timestep weighting 으로 Waymo eval (3s, 5s, 8s) 와 align.

### 8.3. 직접 비교

| 항목 | open_loop | interaction |
|---|---|---|
| modeled agent 수 | ego + N neighbor (N up to 10) | ego + 1 partner |
| ego weighting | 2x (`gmm + gmm[:, 0]`) | 평등 |
| score loss weight | 1x | 2x |
| timestep weighting | uniform (5 step 마다) | uniform + [3s, 5s, 8s] 추가 |
| log_std clip range | (-2, 2) | (0, 5) |
| interaction_loss (proximity penalty) | k≥1 weight 0.1 | 없음 |
| level loop | `range(K)` (K=4 → 4회) | `range(K+1)` (K=3 → 4회 = init + 3 refine) |

## 9. Loss 차이가 만드는 trade-off — neighbor prediction 의 경우

### 9.1. open_loop 의 neighbor 예측은 정확도가 떨어진다

세 가지 이유:

1. ego 12x weight — `loss = mean(all) + mean(ego)` 에서 ego effective weight ≈ 1.09, 각 neighbor ≈ 1/N ≈ 0.09. ego 가 각 neighbor 대비 약 12배 weighted → gradient 가 ego 정확도에 집중

2. capacity dilution — open_loop 는 ego + 10 neighbor 를 한 model 로 학습 (per-agent capacity ≈ 1/11). interaction 은 ego + 1 partner (per-agent ≈ 1/2) → interaction 이 partner 에 더 specialize 가능

3. proximity penalty bias — `interaction_loss` 가 GT 보다 더 멀리 떨어진 trajectory 를 선호하도록 push → neighbor 예측이 현실보다 conservative 해짐

### 9.2. neighbor 예측의 방향성도 다름

```
실제 GT:        neighbor 가 ego 에 1.5m 까지 접근 (실제 도로의 normal)
open_loop 예측: neighbor 가 ego 에 3m+ 거리 유지 (penalty 회피로 conservative)
interaction 예측: neighbor 가 ego 에 1.5m (GT 매칭, pure)
```

→ open_loop 의 neighbor 예측은 "인간 운전자의 평균 + 안전거리 bias" — 정확한 prediction 이 아니라 "planning 의 input 으로 쓸 만한 conservative 가정". interaction 은 "인간 운전자의 honest joint distribution" — Waymo Challenge 를 위한 unbiased 추정.

### 9.3. 이 trade-off 가 ego planning 에 주는 영향

| 측면 | 효과 |
|---|---|
| ego trajectory 자유도 | ↑ — neighbor 가 더 멀리 떨어진다고 가정하므로 ego 의 movable space 확장 |
| safety-critical scenario 의 underestimate | ↑ — 실제 neighbor 가 1.5m 까지 들어올 수 있는데 model 은 3m 가정 → close-proximity 시나리오 학습 X |
| closed-loop deploy | ↓ — 실제 도로의 close interaction 시 model 의 가정 (neighbor 멀리) 이 깨지면서 OOD |

paper 의 의도: open_loop 는 open-loop (한 step) prediction 이고 control stack 은 별도 — proximity penalty 가 만든 "안전한 candidate trajectory" 만 잘 나오면 OK. closed-loop 의 OOD 문제는 별도 paper (DIPP 등) 가 해결.

### 9.4. 요약

paper 는 두 task 의 loss 를 다르게 design 함으로써:
- open_loop: "ego planning 정확도 + system 의 collision-safe 시나리오 prediction" 을 동시에 학습. neighbor 예측은 정확하지 않아도 ego 의 input 으로 쓸 만하면 OK
- interaction: "ego + partner 의 honest joint distribution" 학습. Waymo metric (mAP, MR) 직접 최적화

같은 GameFormer architecture 를 두 가지 다른 evaluation strategy 로 학습 → 두 specialized model 이 나옴. 그러나 이 design 의 cost 는 open_loop 의 neighbor prediction 자체는 standalone 으로 신뢰할 수 없음 — closed-loop 또는 honest neighbor prediction 이 필요한 task 에는 interaction model 이 더 적합.
