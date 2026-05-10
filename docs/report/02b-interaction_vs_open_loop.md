# 02b. interaction vs open_loop — 차이와 game-theoretic 설계

GameFormer paper 의 두 task — `interaction_prediction` 과 `open_loop_planning` — 의 관계와 차이를 자세히 정리.

## 1. 자율주행 stack 에서의 위치

전통적인 자율주행 modular pipeline:

```
                                                ego 의 input ─┐
                                                              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Perception   │→│ Prediction   │→│ Planning     │→│ Control      │
│              │  │              │  │              │  │              │
│ "주변에 뭐가  │  │ "다른 agent  │  │ "ego 가 어떻 │  │ "throttle /  │
│  있는가?"     │  │  가 다음에   │  │  게 trajec-  │  │  brake / steer│
│              │  │  뭐 할 것인가?"│  │  tory 만들까?"│  │  어떻게?"    │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                          ↑                  ↑
                          │                  │
                  ┌───────┴────────┐ ┌──────┴──────────┐
                  │ paper 의       │ │ paper 의        │
                  │ interaction_   │ │ open_loop_      │
                  │ prediction     │ │ planning        │
                  └────────────────┘ └─────────────────┘
```

paper 의 두 task 는 위 stack 의 **prediction** 과 **planning** 단계에 각각 대응.

## 2. 자연스러운 의문 — 왜 별도?

직관적으로:
- **planning 은 prediction 결과가 있어야 가능** (다른 agent 가 뭐 할지 알아야 ego 가 뭐 할지 결정)
- 그럼 cascade (`prediction → planning`) 가 자연스럽지 않나?

전통적 cascade:

```
                                ┌─────────────────────────────────────┐
                                │ neighbor prediction model           │
ego state ─────────────────────▶│  input: agent + map                 │
neighbor states ───────────────▶│  output: 32 neighbor 의 trajectory │
map ────────────────────────────▶│                                     │
                                └─────────────────────────────────────┘
                                                  │
                                                  ▼ (predicted neighbors)
                                ┌─────────────────────────────────────┐
                                │ ego planning model                  │
ego state ─────────────────────▶│  input: ego + (predicted neighbors) │
ref_line ───────────────────────▶│         + map + ref_line            │
map ────────────────────────────▶│  output: ego trajectory             │
                                └─────────────────────────────────────┘
                                                  │
                                                  ▼
                                          ego 의 plan
```

**cascade 의 문제**: prediction 이 *ego 의 미래 행동을 모름* → ego 의 행동이 neighbor reaction 을 유발하는 dynamic 무시.

예시 (4-way 교차로):
- ego 가 좌회전 vs 직진 결정에 따라 oncoming car 가 다르게 reaction
- cascade model: prediction 이 ego 를 무시 → "oncoming car 가 50% 확률로 직진, 50% 확률로 우회전" 같이 단순 marginal
- 실제: ego 가 좌회전하면 oncoming car 는 yield 대기, ego 가 직진하면 oncoming car 가 동시에 통과
- → cascade 는 이 dependency 를 못 잡음

## 3. GameFormer 의 해결 — Game-theoretic joint model

paper 가 제안: **ego planning + neighbor prediction 을 한 model 의 한 forward pass 에서 동시 생성** + **level-k reasoning** 으로 서로의 reaction 을 iteration 으로 근사.

```
                            ┌─────────────────────────────────────────────┐
                            │ GameFormer (open_loop_planning)              │
                            │                                              │
ego state ─────────────────▶│  ┌────────┐                                  │
neighbor states (5) ───────▶│  │Encoder │                                  │
map ────────────────────────▶│  │        │                                  │
ref_line ───────────────────▶│  └────┬───┘                                  │
                            │       │                                      │
                            │       ▼                                      │
                            │  ┌─────────────────────────────────────┐    │
                            │  │ Decoder (level-k reasoning)         │    │
                            │  │                                     │    │
                            │  │ level 0 → level 1 → ... → level K   │    │
                            │  │ (4 iterations of game theory)       │    │
                            │  └────┬────────────────────────────────┘    │
                            │       │                                      │
                            └───────┼──────────────────────────────────────┘
                                    │
                                    ▼
                  {ego trajectory (planning),
                   neighbor[1..5] trajectories (prediction)}
                  모두 multi-modal (M=6)
```

**핵심 insight**:
- ego 와 neighbor 가 **같은 decoder loop** 안에서 서로를 보며 iteration
- level-k 가 끝나면 **수렴된 fixed point** — game-theoretic equilibrium 의 근사
- "prediction 따로 → planning 따로" 가 아니라 **prediction 과 planning 이 서로 영향**

## 4. Level-k reasoning 의 game-theoretic iteration

decoder 의 4 단계 (open_loop 의 K=4) 가 어떻게 동작하는지:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Initial State (encoder output)                                              │
│  - ego 의 context encoding                                                   │
│  - 5 neighbor 의 context encoding                                            │
│  - map encoding                                                              │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Level 0 (Initial guess — marginal prediction)                               │
│                                                                             │
│  각 agent 가 다른 agent 무시한 채 자기 trajectory 추측 :                     │
│  - ego: 6 가능 trajectory (multi-modal)                                      │
│  - neighbor 1: 6 가능 trajectory                                             │
│  - ... neighbor 5: 6 가능 trajectory                                         │
│                                                                             │
│  → 각자 marginal best                                                        │
│  → score (각 modal 의 likelihood)                                            │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Level 1 (1st-order reasoning)                                               │
│                                                                             │
│  각 agent 가 다른 agent 의 level 0 prediction 을 보고 best response :        │
│                                                                             │
│  ego 의 perspective:                                                         │
│    "neighbor 1 이 level 0 에서 직진 확률 60%, 좌회전 40% 라네.               │
│     그럼 나는 어떻게 가는 게 best?"                                          │
│    → ego 의 새 6 trajectory (neighbor 의 행동 고려)                          │
│                                                                             │
│  neighbor 1 의 perspective:                                                  │
│    "ego 가 level 0 에서 좌회전 70%, 직진 30% 라네.                           │
│     그럼 나는 ego 가 좌회전 했을 때 yield, 직진 시 동시 통과"                │
│    → neighbor 1 의 새 6 trajectory                                           │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Level 2 (2nd-order reasoning)                                               │
│                                                                             │
│  각 agent 가 level 1 의 result 를 보고 또 best response :                    │
│                                                                             │
│  ego: "neighbor 1 이 level 1 에서 yield 비중 60% 로 갱신됐네.                │
│         이 정보로 내 trajectory 다시 계산"                                   │
│  neighbor: "ego 가 level 1 에서 좌회전 비중 80% 로 올라갔네.                 │
│              그럼 나는 yield 더 확실히"                                      │
│                                                                             │
│  → 양쪽 prediction 이 서로 의존하면서 점진적으로 정제                        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼ (iteration K times)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Level K (수렴)                                                              │
│                                                                             │
│  level k 가 충분하면 (paper K=4), agent 의 prediction 이 서로의 reaction 을  │
│  반영한 stable equilibrium 에 도달.                                          │
│                                                                             │
│  → ego 의 planning trajectory + neighbor prediction 이 일관성 있게 출력      │
│                                                                             │
│  Loss: 모든 level (0, 1, ..., K) 의 GT 와 비교 (deep supervision)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

각 level 의 transition 은 [`model/modules.py`](model/modules.py) 의 `class InteractionDecoder` 한 forward pass:

```python
def forward(self, id, current_states, actors, scores, last_content, encoding, mask):
    # 1. 다른 agent 들의 prev-level prediction 을 weighted mean 으로 압축
    multi_futures = self.future_encoder(actors[..., :2], current_states)
    futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)
    
    # 2. self-attention 으로 agent 간 interaction 추론 ("level k thinking" 의 핵심)
    interaction = self.interaction_encoder(futures, mask[:, :N])
    
    # 3. 자기 자신의 prev prediction 은 mask out (다른 agent 의 행동만 본다)
    encoding = torch.cat([interaction, encoding], dim=1)
    mask[:, id] = True
    
    # 4. cross-attention + GMM prediction (level k 의 새 trajectory)
    query = last_content + multi_futures[:, id]
    query_content = self.query_encoder(query, encoding, encoding, mask)
    trajectories, scores = self.decoder(query_content)
    
    return query_content, trajectories, scores
```

## 5. 두 model 의 동작 차이

같은 architecture 지만 input/output/loss 가 다름.

### 5.1. interaction_prediction model

```
input:                                  output:
─────────────                          ─────────────
ego_state (1, 11, 9) ─────┐
neighbors_state (32,11,9)─┤            level_0..K_interactions:
map_lanes (33, 100, 16) ──┼─▶ Encoder ─▶ Decoder ─▶  (B, N=33, M=6, T=80, 4)
map_crosswalks (33, 100,3)┤              (K=3)         (= 모든 33 agent 의 6 modal × 80 step)
                          │
                          │            level_0..K_scores:
                          │              (B, N=33, M=6)
                          │              (= 각 modal 의 confidence)
                          ┘

                                       Loss:
                                       ─────
                                       모든 agent (ego + 32 neighbor) 의 GMM NLL
                                       + score classification
                                       (모두 평등한 prediction)
```

### 5.2. open_loop_planning model

```
input:                                  output:
─────────────                          ─────────────
ego_state (1, 11, 9) ─────┐
neighbors_state (5, 11, 9)┤            level_0..K_interactions:
map_lanes (6, 100, 16) ───┼            ego: planning trajectory (6 modal)
map_crosswalks (6, 100,3) │            neighbor 1..5: prediction (6 modal each)
ref_line ─────────────────┼─▶ Encoder ─▶ Decoder ─▶
(ego 가 가야할 reference  │              (K=4)        ── 모두 (B, N=6, M=6, T, 4)
 path — lane centerline)  │
                          │            level_0..K_scores:
                          │              (B, N=6, M=6)
                          ┘

                                       Loss:
                                       ─────
                                       1. Ego planning loss (ref_line 따라가는 정도)
                                       2. Neighbor prediction loss (GMM NLL)
                                       3. Speed/comfort loss (가속도 / jerk smoothness)
                                       4. Mode selection loss (ego 의 best mode)
```

**핵심 차이**:
- interaction: 모든 agent prediction 에 평등한 loss
- open_loop: **ego 만 특별 처리** — ref_line 따라가야 + 부드러운 acceleration + 안전 (충돌 X)

## 6. 학습 setup 차이

| 항목 | interaction | open_loop |
|---|---|---|
| dataset (train) | `training` (full WOMD, 1000 shards, 487K scene) | `training_20s` (sparse subset, 9000개 scenario sampling) |
| dataset (valid) | `validation_interactive` (87K scene) | `validation` (308K scene) |
| neighbors 수 | 32 (multi-agent prediction) | 5 (ego 주변 가까운 차량만) |
| levels (k) | 3 | 4 |
| modalities (M) | 6 | 6 |
| future len | 80 (8 sec @ 10Hz) | 80 |
| batch_size (per GPU) | 16 | 32 |
| GPU | 4 GPU DDP | 1 GPU |
| epoch | 30 | 20 |
| learning_rate | 1e-4 | 1e-4 |
| 추가 input | — | `ref_line` |
| 추가 loss | — | comfort, planning |

## 7. 평가 metric 차이

### 7.1. interaction_prediction (Waymo motion prediction challenge 표준)

```
type 별 (vehicle / pedestrian / cyclist) × horizon 별 (5s / 9s / 15s) × metric

minADE — 모든 modal 중 GT 와 가장 가까운 trajectory 의 average displacement
minFDE — 같은 trajectory 의 endpoint distance
miss_rate — 모든 modal 의 endpoint 가 box 안에 못 들어가는 비율
overlap_rate — modal 간 overlap (mode collapse 측정)
mAP — confidence weighted Average Precision
```

paper Table 1 (Joint M=6, validation_interactive):
- minADE 0.9161, minFDE 1.9373, MR 0.4531, mAP 0.1376

### 7.2. open_loop_planning

```
plannerADE — ego 의 planned trajectory 와 GT trajectory 의 average displacement
plannerFDE — ego 의 endpoint distance
predictorADE — 5 neighbor 의 average prediction error
predictorFDE — 5 neighbor 의 endpoint prediction error
```

paper baseline: val_plannerADE ≈ 0.83.

## 8. Use case 별 model 선택

| 시나리오 | 사용할 model | 이유 |
|---|---|---|
| 자율주행 simulator 의 prediction module 만 교체 | `interaction` | 다른 planner 와 plug-and-play 가능 |
| Waymo Motion Prediction Challenge submission | `interaction` | challenge metric (minADE/FDE/MR/mAP) 가 interaction 학습 직접 측정 |
| 자율주행 stack 의 standalone prediction 모듈 | `interaction` | 32 neighbor 처리 가능 |
| 간단한 open-loop 자율주행 demo (planner 까지 한 model 로) | `open_loop` | ref_line 만 주면 ego planning 출력 |
| trajectory generation 학술 비교 (다른 motion prediction paper 와) | `interaction` | 표준 metric 로 비교 |
| 실제 차량 deploy (closed-loop control) | 둘 다 부적합 | paper 는 open-loop 만 — closed-loop 은 [DIPP](https://github.com/MCZhi/DIPP) 등 별도 |

## 9. User 의 직관에 답

> *interaction 을 알아야 openloop 을 알수있는거 아니야?*

**개념적으로 맞음.** 하지만 paper 의 design 이 영리:

1. **cascade 가 아닌 joint** — ego planning + neighbor prediction 이 같은 model 의 같은 forward pass 에서 game-theoretic level-k reasoning 으로 동시 생성. cascade 의 dependency 무시 문제 해결.

2. **별도 학습** — 같은 architecture 지만:
   - interaction 의 weight 가 open_loop 학습에 transfer 안 됨 (각자 random init 으로 학습)
   - dataset / loss / 추가 input (ref_line) 다르므로 학습 setup 분리
   - 결과: 각 task 의 SOTA (interaction = challenge baseline, open_loop = planning baseline)

3. **사용 시 분리** — 자율주행 stack 에 interaction 을 prediction module 로, 또는 open_loop 을 standalone planner 로 사용 가능.

## 10. 비유로 이해

**전통적 cascade (잘못된 접근)**:
> 의사가 진단 (prediction) 결과를 적은 후 다른 방으로 가서 치료 처방 (planning) 결정. 두 행위가 분리.

**GameFormer 의 joint reasoning**:
> 의사 한 명이 진찰실에서 환자를 보며 "이 환자가 어떤 약 처방하면 → 어떻게 반응 → 그럼 다음 처방을 어떻게" 를 동시에 생각하면서 (level-k iteration) 최종 진단 + 처방 동시 결정.

paper 의 contribution 은 **prediction 과 planning 을 한 thought process 로 묶은 model design**.

## 11. 정리

| 질문 | 답 |
|---|---|
| 같은 architecture? | yes (`model/GameFormer.py` 의 같은 `class GameFormer`) |
| 같은 학습? | no (별도 dataset, 별도 loss, 별도 weight) |
| interaction → open_loop cascade? | no (각각 standalone model) |
| open_loop 안에 interaction 있나? | no (open_loop 은 5 neighbor 만, interaction 은 32 neighbor) |
| 둘 다 prediction 함? | yes (open_loop 도 5 neighbor prediction 함) |
| 차이의 핵심? | open_loop 은 ego trajectory 가 ref_line 따라가야 + 부드러워야 (특별 loss) |
| user 의 직관 (cascade) 와의 관계? | concept 적으로는 맞음. 하지만 paper 는 joint reasoning 으로 우회 |
