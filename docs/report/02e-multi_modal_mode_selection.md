# 02e. Multi-modal output + winner-takes-all 학습 + 학습-평가-deploy mismatch

GameFormer 의 M=6 multi-modal output 의 의미, winner-takes-all (WTA) 학습 mechanism, 그리고 학습-평가-deploy 의 mode selection 이 모두 다른 fundamental mismatch 분석.

## 1. modality 가 무엇인가

modality (mode) = **"가능한 미래 trajectory 가설 1 개"** 를 표현하는 단위. multi-modal = "여러 가능한 미래를 동시에 출력".

### 1.1. 왜 multi-modal 이 필요한가

인간 운전 행동은 본질적으로 uncertain — 같은 past observation 에서도 미래는 여러 갈래.

```
교차로에 다가오는 차량:
  ┌─ 직진 (확률 0.5)
  ├─ 좌회전 (확률 0.3)
  └─ 우회전 (확률 0.2)

deterministic model: "직진" 하나만 출력 → 좌/우회전 가능성 무시
multi-modal model:  3 mode 모두 출력 (각 score 와 함께) → 안전한 planning 가능
```

### 1.2. GameFormer 의 M=6

각 agent 마다 6 개의 가능한 미래 trajectory 를 score 와 함께 출력:

```
agent 1 (ego):
  mode 0: [t=0~5s 의 trajectory] + score 0.40 ← 직진
  mode 1: [trajectory]            + score 0.25 ← 좌회전
  mode 2: [trajectory]            + score 0.15 ← 우회전
  mode 3: [trajectory]            + score 0.10 ← 감속
  mode 4: [trajectory]            + score 0.07 ← 차선 변경
  mode 5: [trajectory]            + score 0.03 ← 정지

agent 2 (neighbor):
  mode 0~5: 각자의 6 가능한 미래
...
```

output tensor shape: `[batch, N_agent, M=6, T_steps, 5]`
- 5 = (x, y, log_std_x, log_std_y, ...) — Gaussian mixture parameters per timestep

### 1.3. J (Joint) vs M (Marginal) variant — paper 의 두 design

paper Table 1 의 두 variant:

| variant | M (modalities) | mode 의 의미 |
|---|---|---|
| J (Joint) | 6 | 각 mode = 모든 agent 의 jointly consistent 한 시나리오 1 개 |
| M (Marginal) | 64 | 각 agent 가 독립적으로 64 개 trajectory + EM aggregation |

#### J variant (M=6)

```
mode 0 = "ego 직진 + neighbor1 yield + neighbor2 직진"     ← 한 시나리오
mode 1 = "ego 좌회전 + neighbor1 직진 + neighbor2 yield"   ← 다른 시나리오
...
```

→ 6 개의 system-level scenario. 각 mode 안에서 ego + neighbor 의 행동이 서로 일치 (game-theoretic).

#### M variant (M=64)

```
ego:       64 개 가능한 trajectory (다른 agent 와 무관)
neighbor1: 64 개 가능한 trajectory
neighbor2: 64 개 가능한 trajectory
→ 후처리 EM 으로 jointly consistent set 추출
```

→ 더 다양한 가능성 (64 > 6) but joint consistency 가 보장 안 됨 → EM 으로 후처리 필요. 코드 미공개.

### 1.4. paper Table 1 의 J vs M trade-off

| variant | minADE | minFDE | Miss Rate | mAP |
|---|---|---|---|---|
| J (M=6) | 0.9161 | 1.9373 | 0.4531 | 0.1376 |
| M (M=64) | 0.9721 | 2.2146 | 0.4933 | 0.1923 |

**M=64 가 best-of-K metric (minADE/FDE/MR) 에서 나쁨에도 불구하고 mAP 는 더 좋음**.

이게 의미하는 바: 단순히 mode 수 늘리면 minADE 만 좋아지지 않음 — score 의 정확도 (mode 어느 것이 진짜 likely 한지) 가 실제 metric 에 큰 영향. 이 분석은 section 4-5 에서 자세히.

## 2. Winner-takes-all (WTA) 학습 mechanism

GameFormer 의 학습은 6 mode 중 **GT 와 가장 가까운 mode 1 개만 supervision** — winner-takes-all.

### 2.1. 학습 코드 ([utils/inter_pred_utils.py:74](../../utils/inter_pred_utils.py#L74))

```python
def gmm_loss(trajectories, convs, probs, ground_truth):
    # trajectories: [B, N, M=6, T, 2]
    # 각 mode 와 GT 사이 distance 계산
    distance = norm(trajectories - ground_truth)
    ndistance = distance.mean(-1) + distance[metric].sum(-1)

    # ← GT 와 가장 가까운 mode 의 index (oracle best)
    best_mode = argmin(ndistance.mean(1), dim=-1)

    # 그 mode 의 trajectory 만 추출
    best_mode_future = trajectories[..., best_mode, ...]

    # 그 mode 만 GMM NLL supervision
    loss = GMM_NLL(best_mode_future, ground_truth)

    # score head 가 그 best_mode index 를 맞추도록 cross-entropy
    prob_loss = cross_entropy(probs, best_mode, label_smoothing=0.2)
    loss = loss + 2 * prob_loss
```

### 2.2. WTA 의 의미 — 6 mode 중 1 mode 만 학습

```
                    forward
GameFormer ────────────────────────► output:
                                       trajectories: [B, N, 6, T, 5]   ← 6 mode trajectory
                                       scores:       [B, N, 6]         ← 6 mode score
                                                          │
                                                          ▼
              ┌──────────────────── argmin(distance to GT) ─────────────┐
              ▼                                                          │
       best_mode = idx                                                   │
              │                                                          │
              ├─ trajectory loss: GMM_NLL(traj[best_mode], GT)           │
              │  (그 mode 만 supervision — WTA)                           │
              │                                                          │
              └─ score loss: CE(scores, best_mode)                       │
                 (score head 가 어느 mode 가 GT 가까운지 맞춤)             │
                                                                         │
              ┌──────────────────────────────────────────────────────────┘
              ▼
    학습 결과 model 의 행동:
      - 6 mode 중 "best 가능한 1 mode" 가 GT 잘 따라감
      - 나머지 5 mode 는 free-floating (어떤 데이터에서 best 였는지에 따라)
      - score head 는 "이 scene 에서 6 mode 중 어느 것이 GT 같을까" 학습
```

### 2.3. WTA 의 부작용

학습 시 6 mode 중 1 mode 만 supervision 받음. 다른 5 mode 의 trajectory loss = 0:

- mode 다양성이 자연스럽게 emerge 만 함 — 명시적 diversity loss 없음
- 다른 5 mode 의 trajectory 가 reasonable 한지 보장 X
- 결과: 학습된 6 mode 중 1~3 mode 만 quality 보장, 나머지 mode 는 "혹시 다른 batch 에서 best 였던 잔재" 의 free-floating

→ "M=6 mode 학습" 해도 **사실상 winner-takes-all 학습 → 실질 mode 다양성 학습 약함**.

paper 의 J variant 는 이를 game-theoretic structure (level-k decoder) 로 일부 보완 — 각 level 의 prediction 이 다른 mode 와 cross-attention 받아 다양성 유도. 하지만 명시적 "6 mode 모두 학습" loss 는 없음.

## 3. 학습 / 평가 / deploy 의 mode selection mismatch

### 3.1. 3 단계의 mode selection 이 모두 다름

| 단계 | mode selection | basis | GT 사용 |
|---|---|---|---|
| 학습 (trajectory) | argmin(distance to GT) | oracle | yes |
| 학습 (score) | classify GT-best mode | GT 가까운 mode 를 score 1로 | yes |
| 평가 (minADE) | argmin(distance to GT) | oracle | yes |
| 평가 (mAP) | argmax(score) | model 자신감 | no |
| deploy | argmax(score) | model 자신감 | no |

### 3.2. 학습 — oracle best (GT 알고 있음)

학습 시 GT 가 있으니 "GT 가장 가까운 mode" 직접 계산 가능. argmin 으로 best mode 선택 후 그 mode 만 학습.

### 3.3. 평가 minADE — 학습과 같은 oracle 평가

```
distance[i] = ADE(trajectories[i], GT)  for i in 1..6
minADE = min(distance)
```

→ "6 mode 중 GT 와 가장 가까운 1 개의 ADE". GT 알고 있어야 계산 가능 = oracle metric.

이게 사용자 직관과 정확히 일치: **"6 번 찍기 중 가장 잘 맞춘 것" 만 평가** — K 가 클수록 무조건 유리한 metric.

### 3.4. 평가 mAP — top-1 by score (deploy 와 align)

```
predicted_top_modes = sort_by_score(modes, descending=True)
match_GT(predicted_top_modes[0])  # 가장 score 높은 mode 가 GT 와 맞는지
```

→ "score 가 가장 높다고 model 이 확신한 mode" 가 실제 GT 와 가까운지. score 정확도까지 평가.

### 3.5. Deploy — top-1 by score (또는 safety filter)

```
predictions = model(scene)              # 6 mode + scores
top_mode = argmax(scores)               # ← 이게 deploy 의 selection
ego_trajectory = predictions[top_mode]
control_stack(ego_trajectory)
```

또는 safety-aware:
```
filtered = [m for m in modes if collision_check(m) < threshold]
best = argmax(filtered, key=score)      # safety filter 후 score top-1
```

→ deploy 는 score 만 신뢰. GT 없으니 oracle best 를 모름.

### 3.6. mismatch 정리

```
학습:    "GT 가장 가까운 mode" 학습 → 그 mode trajectory + 그 mode score 강화
평가 minADE: "GT 가장 가까운 mode" 평가 → oracle, K 클수록 유리한 metric
평가 mAP:    "score 가장 높은 mode" 평가 → deploy 와 align
deploy:  "score 가장 높은 mode" 사용

→ 학습은 "best mode trajectory" 학습 + score 는 "best mode index" 학습
→ deploy 는 "score top mode" 사용 — 이게 진짜 best 인지 보장 X
→ minADE 는 학습과 같은 oracle 평가 → "학습 시 잘 했다" 만 측정
→ mAP 만이 학습-평가-deploy 의 align
```

### 3.7. 평가 metric 의 deploy 적합도

| metric | mode selection 기준 | oracle GT 필요 | deploy 적합도 |
|---|---|---|---|
| minADE@6 | argmin(ADE to GT) — oracle | yes | 낮음 (학습과 동일 평가) |
| minFDE@6 | argmin(FDE to GT) — oracle | yes | 낮음 |
| Miss Rate | 6 mode 모두 miss → fail | yes | 중간 |
| mAP | argmax(score) — model 자신감 | no | 높음 |
| Brier-FDE (Waymo Sim Agents) | top-1 by score weighted | no | 가장 높음 |

paper 가 minADE/minFDE 를 highlight 하지만 mAP 가 진짜 deploy 평가.

## 4. 본 reproduction 의 결과 재해석

### 4.1. tjure (interaction) epoch 7 결과 비교

| metric | tjure | paper J M=6 | 차이 | 의미 |
|---|---|---|---|---|
| minADE (oracle best) | 1.050 | 0.9161 | +14.6% | "학습 시 best mode" 의 trajectory 정확도 |
| minFDE (oracle best) | 2.290 | 1.9373 | +18.2% | 동일 |
| Miss Rate | 0.559 | 0.4531 | +23.3% | 6 mode 가 GT cover 율 |
| mAP (top-1 by score) | 0.103 | 0.1376 | -25.2% | "deploy 에서 사용할 mode" 의 정확도 |

### 4.2. mAP 가 가장 큰 gap — 의미하는 바

학습 epoch 7 의 typical pattern:
- "trajectory regression" 은 어느 정도 됐음 (minADE 14% 차이)
- "score classification" 은 아직 학습 부족 (mAP 25% 차이) → score head 늦게 수렴

WTA 학습의 typical 양상 — trajectory head 와 score head 의 학습 속도 차이. 23 epoch 더 학습으로 score head 수렴 기대.

### 4.3. paper Table 1 의 J vs M variant 의 reinterpret

| variant | minADE | mAP | 해석 |
|---|---|---|---|
| J (M=6) | 0.9161 (좋음) | 0.1376 (나쁨) | 적은 mode 의 each 가 잘 학습됨, but mode 가 적어 score 변별력 약함 |
| M (M=64) | 0.9721 (나쁨) | 0.1923 (좋음) | 많은 mode 라 best-of-K 의 best 가 사실 noise — but score 변별력 강함 |

→ 두 variant 가 다른 trade-off: J 는 "각 mode 의 quality", M 은 "score 의 variance & accuracy" 우위.

진짜 deploy quality 는 mAP — M variant 가 우위. 그런데도 J variant 만 코드 공개됨 — paper 의 marketing 문제.

## 5. 후속 paper 들이 이 한계를 어떻게 다루는가

### 5.1. WTA 의 fundamental 한계 정리

- 6 mode 중 1 mode 만 trajectory loss
- 다른 5 mode 의 quality 보장 X
- score head 가 늦게 수렴 (deploy 에서 critical)
- distribution diversity 자체가 학습 objective 가 아님

### 5.2. 후속 paradigm 의 해결 방식

[09-traffic_sim_paradigms.md](09-traffic_sim_paradigms.md) 의 paradigm 들의 실제 해결:

| paradigm | WTA / fixed-M 우회 방식 |
|---|---|
| MultiPath++ (concurrent work) | score loss + Gaussian mixture full likelihood — 모든 mode 학습 |
| MTR (concurrent work) | dense future loss + best-of-K mixture |
| Trajeglish (token AR) | next-token cross-entropy — discrete distribution 의 모든 token 학습. 사실상 unlimited mode |
| SMART (token AR) | 동일 + scaling. WaymoSim Agents 2024 우승 |
| GAIA-1 (world model) | unsupervised next-token — sample 다양성 자체가 학습 objective |
| CtRL-Sim (return-conditioned) | exponential tilting 으로 mode coverage control. score 가 명시적 reward 와 align |

### 5.3. token AR paradigm 의 fundamental advantage

Trajeglish / SMART 의 핵심 insight:
- Multi-modal trajectory 의 "M=6 mode + WTA loss" 자체가 ill-posed
- discrete sequence 로 token 화하면 → next-token cross-entropy 가 모든 가능 sequence 의 likelihood 학습
- "단 1 mode 만 supervision" 의 한계 우회

```
GameFormer:        6 mode argmin(GT) → WTA → 1 mode supervision
MultiPath++/MTR:   6 mode 의 mixture likelihood → 모든 mode supervision (개선)
Trajeglish/SMART:  vocab 384/2048 의 next-token CE → 모든 가능 trajectory 의 likelihood (근본 해결)
```

### 5.4. Waymo Sim Agents 의 metric 재정의

paper 시점 (ICCV 2023) 의 standard metric (minADE/minFDE) 자체가 best-of-K 라는 fundamental 비판. Waymo 가 2023 에 새 challenge benchmark 도입:

- realism meta metric (distribution match)
- interactive metric (multi-agent consistency)
- kinematic metric (physics feasibility)
- map metric (drivable area)

→ best-of-K 의 oracle 의존도 ↓. SMART-101M 이 2024 challenge 우승 (realism 0.7614).

GameFormer 미참가 — paper 시점 (ICCV 2023.10) 에 Waymo Sim Agents (NeurIPS 2023.05) 와 동시기지만 paper 가 fixed-M paradigm 의 마지막 세대.

## 6. 정리

GameFormer 의 multi-modal output 의 핵심:

1. **modality M=6** = 6 가지 가능한 미래 trajectory 가설을 score 와 함께 출력
2. **학습은 WTA** — GT 가장 가까운 mode 1 개만 trajectory loss + score head 가 그 idx 맞춤
3. **평가 minADE** = "6 번 찍기 중 가장 잘 맞춘 것" — oracle best, K 클수록 유리한 metric
4. **평가 mAP** = "score top-1 mode 가 GT 와 맞는지" — deploy 평가
5. **deploy** = score top-1 mode 사용 (또는 safety filter)
6. **3 단계 mismatch** — 학습 (oracle best) ≠ deploy (top-1 by score). 이게 mAP 가 minADE 보다 늦게 수렴하는 이유

후속 paradigm:
- MultiPath++ / MTR — mixture likelihood 로 WTA 우회
- Trajeglish / SMART — token AR 로 fixed-M 자체 우회. 학계 SOTA
- Waymo Sim Agents — best-of-K 의존도 ↓ benchmark
- GameFormer 는 fixed-M paradigm 의 마지막 세대
