# 02d. Game-theoretic 해석 — "내가 뭐 할까" 보다 "모두가 합리적이라면 시스템이 어떻게 흐를까"

GameFormer 의 model 이 정말로 답하는 질문 — *이는 단순히 ego 의 planning 이 아닌 시스템 전체 prediction*. 그리고 그 prediction 이 **모두가 합리적이라는 가정** 하에 jointly consistent 하게 만들어지는 game-theoretic 의미.

## 1. 핵심 통찰

> **"내가 어떻게 움직여야 한다"** 보다 **"모두가 함께 합리적으로 이동해야 한다면 시스템이 어떻게 흘러갈까"** 를 추론.

이 framing 이 paper 의 contribution 핵심. ego planning 이 아니라 **시스템 전체 prediction** 을 만들고, 그 prediction 의 ego slice 가 곧 ego 의 best plan.

```
일반적인 사고:
  "나는 어떻게 움직여야 안전하고 빨리 갈까?"  → ego planning 중심

paper 의 사고:
  "모두가 합리적으로 행동한다고 가정하면 다음 5초간 시스템이 어떻게 흘러갈까?"
                                              ↓
                              그 시스템 prediction 의 ego trajectory 가 곧 ego 의 best plan
```

## 2. Cooperative vs Non-cooperative game

Game theory 에 두 종류:

### 2.1. Cooperative game (협력 게임)

```
중앙 coordinator (또는 모두가 합심) 가 시스템 전체 utility 최대화

예: 4-way 교차로
  - 모두가 "전체 통행시간 최소화" 목표 공유
  - 누가 yield 하고 누가 가는지 합의
  - social welfare 최적화
```

- 모두가 같은 utility 공유 (or 누군가 할당)
- 합리적이면 모두 win-win
- 자율주행 vehicle-to-vehicle (V2V) communication 있는 미래에 가능

### 2.2. Non-cooperative game (비협력 게임) ← GameFormer

```
각 agent 가 독립적으로 자기 utility 최대화
단, 다른 agent 도 같은 식으로 행동한다는 가정

예: 4-way 교차로
  - 각 차량이 "내 시간 / 안전 / 편안함" 만 신경
  - 다른 차량이 어떻게 행동할지 예측 후 best response
  - 다른 차량도 같은 식으로 예측 → iteration
  - 결국 Nash equilibrium 에 수렴 (아무도 자기 행동 바꿀 incentive 없는 state)
```

- 각자 **독립 결정** (V2V 없이도 가능)
- "다른 agent 도 합리적" 가정만 공유
- 결과: jointly consistent equilibrium

GameFormer 는 **non-cooperative**:
- ego 가 neighbor 와 communication 안 함
- ego 가 *"neighbor 가 어떻게 reasoning 할지"* 를 모델링 (mental model)
- neighbor 도 *"ego 가 어떻게 reasoning 할지"* 를 모델링
- iteration 으로 양쪽 mental model 이 수렴

## 3. Level-k = "다른 agent 도 같은 reasoning" 가정

paper 의 핵심 trick — **"다른 agent 가 나와 같은 식으로 reasoning 한다"** 라는 가정을 K depth 로 unrolled:

```
┌─────────────────────────────────────────────────────────┐
│ Level 0 — "다른 agent 는 단순 (반응적)"                 │
│                                                          │
│  각 agent X: "다른 agent 들은 그냥 자기 trajectory 따라  │
│              직선 / context 기반 단순 prediction"        │
│              → 그 가정 하에 X 의 marginal best          │
└─────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────┐
│ Level 1 — "다른 agent 는 level-0 reasoning"             │
│                                                          │
│  각 agent X: "다른 agent 들도 자기 입장에서 level-0      │
│              best response 함 (= 단순 prediction 한 후    │
│              그에 맞춰 행동)"                            │
│              → 그 가정 하에 X 의 새 best response       │
└─────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────┐
│ Level 2 — "다른 agent 는 level-1 reasoning"             │
│                                                          │
│  각 agent X: "다른 agent 도 'X 가 level-0 행동한다고     │
│              가정한 best response' 함"                   │
│              → X 도 한 단계 더 깊이 추론                │
└─────────────────────────────────────────────────────────┘
                                ↓
                          ... K levels ...
                                ↓
┌─────────────────────────────────────────────────────────┐
│ Level K — 수렴 (Nash-like equilibrium)                  │
│                                                          │
│  모든 agent 의 prediction 이 "다른 agent 도 K-level      │
│  reasoning 한다" 가정 하에 mutually consistent           │
│  → 누구도 자기 행동 바꿀 incentive 없음 (equilibrium)    │
└─────────────────────────────────────────────────────────┘
```

## 4. "내가 뭐 해야 하나" 는 시스템 prediction 의 부분집합

```
GameFormer model
       ↓
시스템 전체 prediction (Nash equilibrium)
{
  agent 0 (ego):      6 modal trajectory + scores
  agent 1 (neighbor): 6 modal trajectory + scores
  agent 2 (neighbor): 6 modal trajectory + scores
  ...
  agent 10:           6 modal trajectory + scores
}
       ↓
       ↓ (use case 별로 slice)
       ↓
 ┌─────────────────────┬─────────────────────┐
 │ 자율주행 control    │ 자율주행 prediction  │
 │ → ego trajectory    │ → 모든 agent         │
 │   (agent 0) 만 추출 │   trajectory 사용    │
 │                     │                      │
 │ "내가 뭐 해야 하나" │ "주변에서 뭐가       │
 │ 의 답               │  일어날까" 의 답     │
 └─────────────────────┴─────────────────────┘
```

즉:
- **model 의 직접 output**: *"모두가 합리적이라 가정하면 시스템이 어떻게 흘러갈까"* (전체 prediction)
- **"내가 뭐 해야 하나" 의 답**: 그 전체 prediction 에서 **ego 의 trajectory** 만 추출

## 5. Loss 가 강조하는 부분의 차이 (interaction vs open_loop)

| | interaction | open_loop |
|---|---|---|
| Model output | 모든 agent 의 jointly consistent prediction | 모든 agent 의 jointly consistent prediction |
| **Loss 가 보는 것** | 모든 agent 평등 — "전체 시스템 prediction 정확도" | ego 우선 + neighbor 보조 — "ego trajectory 가 GT planning 과 얼마나 맞는지 + neighbor prediction 정확도" |
| 사용 의도 | "다음 8 초 시스템 prediction" 자체가 목적 | "ego 가 어떻게 plan 할지" 가 목적 — 시스템 prediction 은 도구 |

**둘 다 game-theoretic system prediction 을 output**. 차이는:
- interaction: 그 prediction 자체가 evaluation target (모두 평등하게)
- open_loop: 그 prediction 의 ego 부분 만 evaluation target (ego 정확도가 핵심)

## 6. 4-way 교차로 시나리오로 비유

전통 cascade vs GameFormer 의 차이:

### 6.1. 전통적 cascade (잘못된 가정)

```
ego 의 prediction module:
  "oncoming car 가 50% 직진, 50% 우회전"
  (ego 의 행동 무관 prediction)
       ↓
ego 의 planning module:
  "50% 직진 / 50% 우회전 prediction 받았으니 안전하게 yield"
       ↓
실제 결과:
  - oncoming car 도 ego 가 yield 한 걸 보고 yield → 둘 다 정지 (deadlock)
  - 또는 ego 가 yield 했더니 oncoming 이 직진 → 안전 (다행)
  → 모두가 잘못된 이유: prediction 이 ego 의 의사결정 무관하게 만들어졌음
```

### 6.2. GameFormer 의 game-theoretic reasoning

```
Level 0:
  ego: "oncoming 은 50% 직진 / 50% 우회전 (단순 추측)"
  oncoming: "ego 는 50% 직진 / 50% 우회전 (단순 추측)"

Level 1:
  ego: "oncoming 이 level-0 추측한대로 행동하면 어떻게?
        → 내가 좌회전하면 oncoming 이 yield (왜냐하면 oncoming 도 나의 좌회전 의도를 봄)
        → 내가 직진하면 oncoming 도 직진 (둘 다 같은 priority)"
        
  oncoming: 같은 식으로 ego 의 행동에 대한 reaction 추론

Level 2:
  ego: "oncoming 이 level-1 처럼 reasoning 하면, 내가 어떤 action 을 보일 때
        oncoming 이 어떻게 반응할지 더 세밀하게 예측"
  
  oncoming: 같은 식

Level 3, 4: 수렴
  ego 의 trajectory + oncoming 의 trajectory 가 jointly consistent.
  예: ego 좌회전 trajectory + oncoming yield trajectory (한 mode)
       또는 ego 직진 trajectory + oncoming 직진 trajectory (다른 mode)

각 mode 가 "모두가 합리적일 때 가능한 시스템 진화" 의 하나.
```

**핵심**: GameFormer 는 "내가 뭐 해야" 만 답하지 않고, **"모두가 게임 이론대로 행동할 때 시나리오들"** 을 multi-modal 로 출력. 그 중 ego 의 trajectory 가 ego 의 plan.

## 7. paper 에서 명시한 이 framing

paper 의 contribution 다시 읽으면:

> *"An agent's optimal action depends on the predicted actions of others, and those predictions depend recursively on what others believe this agent will do."*

= "다른 agent 도 나처럼 reasoning 한다는 가정 하에 모두 jointly 합리적으로 행동" = User 의 직관

paper 가 한 일:
1. 이 가정을 **수학적으로 명시** (level-k recursion)
2. 그걸 **end-to-end model 로 학습** (not hand-coded)
3. 학습 결과 자율주행 prediction / planning baseline 보다 정확

## 8. 비유로 정확히

**나쁜 가정 (cascade / single-shot)**:
> "다른 차량은 내 행동과 무관하게 움직인다. 그 prediction 을 기반으로 내 plan 결정"

**좋은 가정 (GameFormer / level-k)**:
> "다른 차량도 나처럼 reasoning 한다. 그러면 다음 시나리오가 가장 그럴듯:
>   - 내가 좌회전하면 → oncoming 은 yield 할 것
>   - 내가 직진하면 → oncoming 도 직진할 것
> → 둘 다 가능하지만 oncoming 이 'ego 가 좌회전 vs 직진' 을 어떻게 추측할지 도 고려해서 시나리오 확률 계산"

→ 결과 시나리오 = "모두가 reasoning 했을 때 jointly consistent 한 미래"

## 9. Multi-modal 의 game-theoretic 의미

paper 가 M=6 modal trajectory 를 출력하는 이유:

- Nash equilibrium 이 **여러 개** 일 수 있음 (multi-equilibria)
- 4-way 교차로 예: (ego 좌회전 + oncoming yield), (ego 직진 + oncoming yield), (ego yield + oncoming 직진) 등
- 각 mode = 하나의 가능한 equilibrium scenario
- score = 각 equilibrium 의 likelihood

multi-modal output 이 **"이 시나리오 모두 합리적으로 가능"** 의 표현. 자율주행 control 시 가장 likely mode (또는 가장 안전 mode) 선택.

## 10. 자율주행 system 에서의 활용

GameFormer 의 output 이 어떻게 사용되는지:

```
자율주행 stack
       │
       ▼
┌─────────────────────────────────────────────────┐
│ Perception (sensor → ego_state, neighbors_state)│
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│ GameFormer (시스템 prediction)                  │
│   output: 11 agent × 6 mode × 50 step trajectory│
└─────────────────────────────────────────────────┘
       │
       ├─ ego trajectory (6 mode) ──▶ Planning module
       │                              "이 6 mode 중 가장 안전 + 빠른 mode 선택"
       │                                          │
       │                                          ▼
       │                                    ┌──────────┐
       │                                    │ Control  │
       │                                    │ throttle │
       │                                    │ brake    │
       │                                    │ steer    │
       │                                    └──────────┘
       │
       └─ neighbor predictions ─────▶ Risk assessment / safety check
                                       "어느 mode 가 충돌 위험?"
```

GameFormer 의 ego trajectory 는 **planner 가 받는 candidate trajectory**. 최종 control 결정은 planner 가 mode selection + safety check 후 결정.

## 11. "planner" 는 모델이 아니라 downstream interpretation

이 framing 의 가장 sharp 한 함의 — **open_loop_planning 의 model 도 엄밀히 말하면 "planner" 가 아니다**. 모델은 "현재 scene 에서 모두가 합리적으로 (level-k 가정 하에) 움직인다면 시스템이 어떻게 진화할까" 를 추론하는 **system rational evolution predictor**. "ego 의 plan" 은 그 system prediction 의 한 index slice 일 뿐.

### 11.1. 코드 레벨 evidence

`open_loop_planning/train.py` 의 forward + loss:

```
prediction, plans, scores, ego_plan = model(inputs)
   prediction:  [B, N-1, M, T, 5]   ← neighbor 들의 multi-modal trajectory
   plans:       [B, M, T, 5]        ← ego 의 multi-modal trajectory
   scores:      [B, M]              ← M mode 별 likelihood
   ego_plan:    [B, T, 3]           ← top-1 mode 의 ego trajectory

planning_loss = MSE(ego_plan, gt_plan)               ← ego index 만 사용
prediction_loss = GMM_NLL(prediction, gt_neighbors)  ← neighbor index 들 사용
```

→ model 의 forward 는 **모든 agent 의 joint multi-modal trajectory** 를 한번에 만들고, loss 단계에서 "ego index → planning loss, neighbor index → prediction loss" 로 분기. model 본체는 ego/neighbor 구분 없이 jointly reasoning.

`model/GameFormer.py` 의 GameDecoder:

```
# k 번째 level decoder — 모든 agent 가 동시에 다른 모든 agent 의 예측을 받아 best response
agents_query: [B, N, D]              ← N 개 agent 의 query 가 평등하게 self-attention
predictions:  [B, N, M, T, 5]        ← N 개 agent 의 multi-modal output 동시 산출
```

→ ego 와 neighbor 는 같은 self-attention 을 거쳐 같은 output head 로 평등하게 출력. "ego" 라는 special handling 은 model 안에 없음. (ego 에게 추가로 주는 route conditioning 은 input 에서만 — output 추출 단계에선 단지 index 0)

### 11.2. open_loop 와 interaction 의 진짜 차이

| | open_loop | interaction |
|---|---|---|
| **input 차이** | ego 에게 route/goal feature 추가 conditioning | 모든 agent 대칭 input |
| **모델 본체** | system rational evolution prediction | system rational evolution prediction |
| **output 추출** | ego index → planning loss + 모든 neighbor index → prediction loss | 모든 modeled agent index → joint prediction loss |
| **사용 의도** | "ego 가 어떻게 plan 할지" — 시스템 prediction 은 도구 | "다음 8 초 시스템 prediction" 자체가 목적 |
| **metric** | ego trajectory 의 ADE/FDE | 모든 modeled agent 의 minADE/minFDE/MR/mAP |

핵심: 두 task 의 모델 forward 는 **본질적으로 같은 일** — system rational evolution prediction. 차이는 (1) ego 에게 route conditioning 을 줄지 (2) loss 가 어느 index 를 어떻게 평가할지 뿐.

### 11.3. 그래서 "planner" 라는 단어는?

paper 가 "planning" 이라고 부르는 이유:

- 자율주행 산업 관행 — ego trajectory 를 만드는 module = "planner"
- 학습 supervision 의 핵심 target 이 ego trajectory (planning loss 의 weight 가 prediction loss 보다 큼)
- downstream control stack 이 받는 입력 = ego trajectory

근데 모델 자체는 planner 가 아니라 **conditional joint distribution sampler** — `p(future trajectories of all agents | current scene, ego goal)` 를 model. 그 분포에서 ego index marginal 을 뽑아 "plan" 으로 해석.

### 11.4. 왜 이 sharpening 이 중요한가

이 framing 이 정확하면 다음이 자연스럽게 따라옴:

1. **multi-modal output 의 의미** — ego planning 의 6 modal 이 아니라 **6 가지 가능한 jointly consistent 시스템 진화 시나리오**. 각 mode 가 하나의 가능한 Nash equilibrium scenario.

2. **mode 간 consistency** — 같은 mode 안에서 ego trajectory 와 neighbor trajectory 가 jointly consistent (서로의 행동을 가정하고 best response). 다른 mode 는 다른 equilibrium.

3. **closed-loop vs open-loop** — open-loop 는 한 step 의 system prediction. closed-loop (DIPP 등) 은 매 step 마다 system prediction 다시 → 한 mode 선택 → 한 step 실행 → 다시 prediction 의 closed loop.

4. **safety / interpretability** — system prediction 이 명시적이라 "이 mode 를 선택하면 neighbor 가 어떻게 행동할 것" 을 interpretable 하게 visualize 가능. cascade 보다 explanation 강함.

5. **paper 의 기여 위치** — paper 의 novelty 가 "더 나은 planner 만들기" 가 아니라 **"planner 라는 framing 자체를 system prediction 의 slice 로 재정의"**. 그래서 prediction 과 planning 을 같은 architecture 로 통합 가능 (interaction + open_loop 가 같은 model class).

## 12. 정리

paper 의 핵심 framing:

| 일반적 사고 | GameFormer 사고 |
|---|---|
| ego 의 planning 이 핵심 | 시스템 전체 prediction 이 핵심 |
| neighbor 는 "예측 대상" | neighbor 는 "동등한 reasoning agent" |
| cascade (predict → plan) | joint reasoning (level-k iteration) |
| "나는 어떻게 가야" | "모두가 합리적이면 시스템이 어떻게 흐를" |

User 의 직관 = paper 의 contribution 정확:
- model 의 직접 output = 시스템 prediction
- ego 의 plan = 그 prediction 의 ego slice
- "모두가 함께 합리적으로" = level-k iteration 으로 수렴한 Nash equilibrium

이 측면에서:
- **interaction model** = 시스템 prediction 자체가 산출물 (모든 agent 평등)
- **open_loop model** = 같은 시스템 prediction 인데 ego 부분만 강조 (ego plan 이 evaluation 의 핵심)

근데 둘 다 *"내가 뭐 해야 하나"* 보다는 **"모두가 합리적이면 시스템이 어떻게 흘러갈까"** 의 답을 내는 model. 자율주행에서 그 답의 **ego slice 가 곧 ego 의 best plan**.

가장 sharp 한 표현: GameFormer 의 모델은 planner 가 아니라 "conditional joint trajectory distribution sampler". "planner" 는 그 sampler 의 ego marginal 을 뽑아서 control stack 에 넘기는 downstream 해석. 모델 본체는 ego 의 의사결정을 하는게 아니라 모두가 합리적이라는 가정 하의 시스템 진화 시나리오 들을 multi-modal 로 출력.

## 13. 한계 — 진짜 game-theoretic 추론인가, WOMD 모방인가

지금까지 본 framing 은 GameFormer 가 "모두가 합리적이라는 가정 하의 시스템 prediction" 을 한다고 했지만, 이 "rationality" 가 어디서 오는지 정직하게 보면 한계가 명확.

### 13.1. "game-theoretic" 은 architecture 의 inductive bias 일 뿐

paper 의 level-k decoder 가 강제하는 것:

```
Level 0:  ego query + map + history          → initial trajectory
Level 1:  ego query + (level-0 의 모든 agent prediction) attention → refined
Level 2:  ego query + (level-1 의 모든 agent prediction) attention → refined
...
```

이 구조가 모델에게 주는 capacity:
- 다른 agent 의 trajectory 를 보고 본인 trajectory 조정 가능 (cross-attention)
- "다른 agent 도 같은 식으로 reasoning 한다" 패턴을 학습할 수 있는 architectural prior

이 구조가 강제하지 않는 것:
- Nash equilibrium 의 mathematical convergence (no fixed-point iteration, no equilibrium check)
- 명시적 utility function (각 agent 의 목적 함수가 없음)
- reward 또는 cost — model 은 그저 GT trajectory 와 매칭만 함

즉 "game-theoretic" 은 architectural inductive bias — "이런 구조라면 game-theoretic reasoning 을 학습할 수 있는 capacity 가 있다" 는 의미. 실제 학습 후 그 capacity 가 game-theoretic 으로 사용되는지는 GT 가 어떤 패턴을 보여주느냐에 따라 결정.

### 13.2. supervision 은 GT trajectory 매칭 (= behavior cloning)

```
Loss = imitation_loss(prediction, GT_human_trajectory)
     = "WOMD 인간 운전자의 다음 N초간 실제 trajectory" 를 매칭
```

따라서 모델이 학습한 "rationality" = "WOMD 인간 운전자가 어떻게 행동했는가". 이게 진짜 합리적인지 (게임 이론적 optimal 인지) 와는 별개.

paper 의 prediction 은 사실상:

> "WOMD 분포 안에서, 이 scene 과 비슷한 상황에서 인간 운전자들이 어떻게 행동했는가"

→ 이걸 "합리적 행동" 으로 가정하고 학습. 합리성의 정의가 데이터 분포에 의해 implicitly 결정됨.

### 13.3. 따라오는 fundamental limitation

| 한계 | 의미 |
|---|---|
| distribution shift | WOMD 분포 밖의 시나리오 (눈길, 사고, 응급차) 는 학습 안 됨 — 일반화 X |
| cultural / regional bias | WOMD = 미국 6 도시 (Phoenix, Mountain View 등). 한국 / 인도 / 유럽 운전 패턴과 다름 |
| covariate shift (closed-loop drift) | open-loop 에서 학습한 model 이 closed-loop deploy 시 본인 action 의 결과를 본 적 없음 → 점점 OOD 로 drift. DIPP 가 별도로 만들어진 이유 |
| rare event underrepresent | 사고 / 충돌 / 교통 위반 = WOMD 에 거의 없음 → safety-critical 시나리오 학습 안 됨 |
| no safety guarantee | first-principles 수학적 보장 X. "GT 매칭이 잘 되면 안전할 거다" 의 가정 |
| suboptimal demonstrations | GT 인간 운전자 자체가 sub-optimal — 졸음, 짜증, 부주의 운전 모두 GT 에 섞임. 그것을 "rational" 로 학습 |

### 13.4. 진짜 first-principles game-theoretic 접근들과의 비교

| 접근 | rationality 의 source | 한계 |
|---|---|---|
| MPC + game theory (Wang, Schwager 등) | 명시적 utility function + iterative best-response 또는 Nash solver | 계산 비싸고 utility 정의 어려움 |
| MARL (multi-agent RL) | reward function + simulator 에서 self-play 학습 | reward shaping 어려움, sim2real gap |
| RL with human feedback | 인간 평가자의 실시간 피드백 → reward signal | cost ↑, scalability 한계 |
| GameFormer (BC + game-theoretic prior) | 데이터 분포 (WOMD) | distribution shift, no safety guarantee |

GameFormer 의 위치: "learned model 의 expressiveness + game-theoretic architectural prior" 의 hybrid. 둘 다 어느 정도만 — 진짜 game theory 도 아니고 순수 BC 도 아닌 중간.

### 13.5. paper 가 실제로 평가한 것

paper 의 metric (minADE, minFDE, MR, mAP) 모두 WOMD validation set 안에서 측정:
- distribution shift 평가 없음
- closed-loop 평가 없음
- safety-critical scenario 평가 없음
- regional / cultural transfer 평가 없음

이게 ICCV academic paper 의 typical limitation — academic dataset 에서 SOTA 측정. 산업체 deploy 의 fundamental 문제는 별도.

### 13.6. paper 의 contribution 의 정확한 위치

paper 가 정말 한 일:
> WOMD 분포 안의 prediction + planning 의 quality 를 level-k joint reasoning architecture 로 개선. distribution 안에서 SOTA. distribution 밖은 future work.

이 framing 으로 보면:
- paper 가 자율주행의 fundamental 해결 X
- 한 분야 (joint prediction + planning architecture) 의 incremental improvement
- "game-theoretic" 은 architectural prior 의 명명일 뿐, 진짜 game theory 의 mathematical guarantee X

진짜 자율주행 fundamental 해결은 별도 paradigm:
- safer paradigm (RL with safety constraints, formal verification, MPC + safety filter)
- end-to-end perception (sensor cascade error 제거)
- closed-loop training (DAgger, RL, sim2real)
- safety-critical dataset (rare events, edge cases)

GameFormer 는 그 중 어느 것도 직접 다루지 않음.

## 14. 또 하나의 가정 — perception 의 oracle

GameFormer 의 input 을 다시 보면 자율주행 stack 에서의 위치가 명확:

### 14.1. GameFormer 의 input 은 vectorized scene representation

```python
# utils/inter_pred_utils.py 의 DrivingData
ego:           [21 step × 7 feature]   # x, y, heading, vx, vy, ...  state history
neighbor:      [10 × 21 × 7]           # 10 neighbor 의 state history
map_lanes:     [L × P × F]              # vectorized lane polylines
map_crosswalks:[C × P × F]              # vectorized crosswalk polygons
```

모두 vectorized scene representation. 즉 perception (camera/lidar → object detection + tracking + map) 이 끝난 후의 산출물. raw image 나 lidar point cloud 안 받음.

### 14.2. AD stack 에서의 위치

```
┌──────────────────────────────────────────────────────┐
│ Sensor (camera, lidar, radar)                        │
└─────────────────────┬────────────────────────────────┘
                      │ raw signals
                      ▼
┌──────────────────────────────────────────────────────┐
│ Perception (object detection + tracking + map)       │
│ - 별도 module (e.g., CenterPoint, MapTR)             │
└─────────────────────┬────────────────────────────────┘
                      │ vectorized objects + map
                      ▼
┌──────────────────────────────────────────────────────┐
│ GameFormer (prediction + planning)  ← 여기            │
│ - input: ego/neighbor states + map vector            │
│ - output: multi-modal trajectory                     │
└─────────────────────┬────────────────────────────────┘
                      │ ego trajectory candidate
                      ▼
┌──────────────────────────────────────────────────────┐
│ Control (throttle, brake, steer)                     │
└──────────────────────────────────────────────────────┘
```

GameFormer 는 modular AD stack 의 prediction + planning module. perception 은 oracle 가정.

### 14.3. perception oracle 가정의 함의

- detection error (false positive / negative) → cascade
- tracking ID switch → ego/neighbor 매핑 불안정
- map error (lane geometry 부정확) → 학습 분포와 mismatch
- weather / lighting condition → perception 직접 영향 (GameFormer 는 모름)

paper 는 이 cascade error 의 정량적 분석 없음 — 모든 실험은 perception 이 oracle 이라 가정.

### 14.4. End-to-end alternatives

이미지 / sensor 직접 받는 모델은 별도 계열:
- UniAD (CVPR'23 best paper): camera → BEV → prediction + planning end-to-end
- VAD (ICCV'23): vectorized scene 학습 + planning end-to-end
- Tesla FSD v12 (산업체): camera-only end-to-end neural net
- Wayve (영국 startup): end-to-end RL + vision

이 계열의 의도: perception 의 cascade error 회피 + raw scene 정보 활용. 단점: compute 비용 + interpretability 어려움.

GameFormer 는 그 trade-off 의 다른 쪽 — modular + interpretable + perception quality 가정. perception 이 정확하면 deploy 쉬움, 부정확하면 cascade error.

## 15. 두 한계의 종합

GameFormer 의 두 fundamental 가정:

1. rationality = WOMD 분포 — imitation learning 의 한계 (BC paradigm 의 typical 문제)
2. perception = oracle — modular stack 의 cascade error 가정

두 가정 모두 paper 의 평가 범위 내에서는 문제 안 됨 (WOMD validation set 에서 perception 이 정확). 하지만 산업체 deploy 시:

- distribution shift → unfamiliar scenario 에서 fail
- closed-loop drift → 시간 지나면서 OOD 로 drift
- perception cascade → detection 부정확 시 trajectory 부정확
- safety-critical event → 학습 분포에 없으니 적절 대응 불가

paper 의 contribution 을 정직히 위치시키면:
- WOMD 같은 academic dataset 안에서 prediction + planning quality 의 incremental improvement
- 자율주행의 fundamental 한 해결책 (safety, generalization, robust deploy) 은 직접 다루지 않음
- "game-theoretic" 은 architectural inductive bias 의 명명 — 진짜 game theory 의 mathematical guarantee 는 없음

이 한계가 GameFormer 의 결함은 아님. 모든 ICCV-tier academic paper 의 typical scope. 다만 paper 의 framing ("game-theoretic reasoning for autonomous driving") 이 fundamental 해결처럼 들리는 부분을 정직히 분리해서 봐야 함.
