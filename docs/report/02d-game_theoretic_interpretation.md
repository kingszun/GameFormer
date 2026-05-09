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

## 11. 정리

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
