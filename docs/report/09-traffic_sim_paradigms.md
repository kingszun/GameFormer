# 09. GameFormer 의 한계를 극복하는 traffic simulation paradigm

GameFormer (ICCV'23) 의 두 fundamental 한계 — (1) pure imitation learning (= "WOMD 인간 = rational" 가정), (2) open-loop training (= covariate shift 미해결) — 를 직접 다룬 후속 paradigm 의 deep-dive 분석. 4 가지 paradigm × 12 paper 의 정량 사실 + GameFormer 한계와의 매핑.

본 문서의 모든 정량 수치는 paper table 직접 인용. 추측한 부분은 [추측] 으로 명시.

## 1. GameFormer 의 한계 다시 정리

[02d-game_theoretic_interpretation](02d-game_theoretic_interpretation.md) 에서 분석한 한계:

| 한계 | 의미 |
|---|---|
| pure GT imitation | supervision = WOMD trajectory 매칭. "rationality" 의 정의가 데이터 분포에 implicit |
| sub-optimal demonstration 학습 | 인간 운전자의 졸음/부주의/위반 도 "rational" 로 학습 |
| open-loop training | 1 step prediction loss. self-induced state distribution 학습 X |
| covariate shift | deploy 시 본인 action 의 결과 본 적 없음 → drift |
| safety-critical scenario 부재 | WOMD 의 99.9% 가 normal — collision/near-miss 학습 X |
| fixed M=6 mode | multi-modal 의 mode 수가 적고 데이터 분포 안 |
| no controllability | aggressive vs cautious 행동 deploy time 변경 X |
| perception oracle 가정 | vectorized scene 입력 — perception cascade error 미고려 |

후속 paradigm 들이 이 한계 중 어느 것을 어떻게 다루는지 정리.

## 2. Paradigm 분류

```
┌───────────────────────────────────────────────────────────────┐
│ GameFormer (open-loop BC + game-theoretic prior, ICCV 2023)    │
└───────────────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────┼──────────────────────────────────┐
   ▼                          ▼                          ▼       ▼
┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Closed-  │  │ RL with    │  │ Adversarial  │  │ World model +    │
│ loop     │  │ reward     │  │ scenario     │  │ token autoregr   │
│ reactive │  │            │  │ generation   │  │                  │
├──────────┤  ├────────────┤  ├──────────────┤  ├──────────────────┤
│ TrafficSim│  │ CtRL-Sim   │  │ STRIVE       │  │ GAIA-1           │
│ Symphony  │  │ GUMP       │  │ KING         │  │ DriveDreamer (1/2)│
│ TrafficBots│  │ BC-SAC    │  │ CAT          │  │ Trajeglish        │
│           │  │ (Waymo)    │  │              │  │ SMART             │
└──────────┘  └────────────┘  └──────────────┘  └──────────────────┘
   │              │                  │                  │
   ▼              ▼                  ▼                  ▼
covariate     "WOMD = rational"  long-tail safety   fixed M-mode +
shift 해결    가정 극복           scenario 학습      distribution diversity
```

각 paradigm 이 한 측면씩 해결. GameFormer 의 모든 한계를 한 모델로 해결한 single solution 은 아직 없음.

---

## 3. Paradigm 1 — Closed-loop Reactive Simulation

### 3.1. 핵심 아이디어

문제: open-loop BC 는 학습 시 GT state 만 봄 → deploy 시 model 의 self-induced state distribution 에서 drift. GameFormer 가 정확히 이 한계.

해결: 학습 루프 안에서 model 의 prediction 을 다시 sim state 로 넣어 self-induced distribution 학습. 핵심 mechanism 은 backprop through simulation (BPTT) 또는 tree search distillation.

### 3.2. TrafficSim (Suo et al., CVPR 2021)

- arxiv: [2101.06557](https://arxiv.org/abs/2101.06557)
- Affiliation: Uber ATG + University of Toronto + University of Waterloo
- Architecture: rasterized HD map + GRU + GNN-based scene-level CVAE. 모든 actor 의 $T_{plan}$ step 미래를 한 번에 출력
- Loss: ELBO (reconstruction + KL) + collision (5-circle approximation) + time-varying $\lambda(t)$ weighting
- Covariate shift 해결: closed-loop unrolling + fully differentiable simulation. $t \le T_{label}$ 에는 posterior sample, $t > T_{label}$ 에는 prior sample 로 unroll. backprop 이 simulation step 통과
- Dataset: ATG4D (Uber in-house, WOMD 아님). 1M+ frame, 64-beam LiDAR, 6500 snippet × 25s

Benchmark (Table 1, 12s rollout, 비교 baseline: IDM, MTP, ILVM, AdversarialIL, DataAug):

| Model | SCR (%) | TRV (%) | minSADE (m) | meanSADE (m) |
|---|---|---|---|---|
| IDM | 1.19 | 0.25 | 3.03 | 3.48 |
| ILVM | 2.90 | 4.37 | 1.33 | 1.50 |
| TrafficSim | 0.50 | 2.77 | 0.57 | 0.85 |

Ablation (Table 2) 가 covariate shift 해결의 핵심 증거: open-loop BC SCR 5.92% → closed-loop unroll 0.60% → +collision loss 0.50%.

GameFormer 한계 매핑:
- covariate shift / drift: directly 해결 (closed-loop differentiable unroll)
- pure GT matching distribution shift: $\beta$-VAE ELBO + prior rollout 으로 GT 너머의 state 분포 학습

자체 한계: differentiable dynamics 가정 (복잡 vehicle dynamics 호환 어려움), in-house dataset 만 검증 (WOMD/nuPlan benchmark 와 직접 비교 X).

### 3.3. Symphony (Igl et al., ICRA 2022)

- arxiv: [2205.03195](https://arxiv.org/abs/2205.03195)
- Affiliation: Waymo Research
- Architecture: per-object-type MLP encoder + max-pool. BC 또는 MGAIL (model-based GAIL) base + parallel beam search + hierarchical goal policy
- Training: 4 가지 조합 (BC, BC+H, BC+TS, BC+TS+H, MGAIL 도 동일 4 조합)
- Parallel Beam Search (TS): training $S=4$ / inference $S=16$ branch. 2초 마다 discriminator score 로 하위 절반 prune. 결과 rollout 을 BC 의 추가 학습 데이터로 distill
- Hierarchical Policy (H): goal generating policy (route as lane segment seq, episode 시작 1번 sample) + goal-conditioned policy. 200 feasible route 의 softmax

Benchmark (Table I, WOMD 10s rollout):

| Method | Collision (%) | Off-road (%) | minSADE | Curv. JSD |
|---|---|---|---|---|
| BC | 24.65 | 2.75 | 1.76 | 1.32 |
| BC+TS | 4.94 | — | — | — |
| BC+TS+H | 4.86 | 1.30 | 1.66 | 2.82 |
| MGAIL | 9.48 | 1.91 | 3.13 | 4.14 |
| MGAIL+TS+H | 4.89 | 1.65 | 2.22 | 2.80 |

Proprietary Waymo dataset 도 동일 패턴: BC collision 16.65% → BC+TS 1.84% → BC+TS+H 1.80%. tree search 가 covariate shift 해결의 dominant effect, hierarchy 가 mode collapse 회피.

GameFormer 한계 매핑:
- covariate shift: parallel beam search prune + GAIL state distribution matching
- sub-optimal demonstration: discriminator 가 reward 학습 → reward design 불요
- mode collapse: hierarchical goal 로 diversity 보존

자체 한계: inference time computation 증가 ($S=16$ branch), goal=route 가정으로 비정형 환경 (parking lot 등) 부적합, MGAIL 의 differentiable simulator 가정.

### 3.4. TrafficBots (Zhang et al., ICRA 2023)

- arxiv: [2303.04116](https://arxiv.org/abs/2303.04116)
- Affiliation: ETH Zürich + MPI for Informatics + KU Leuven
- Architecture: vectorized Transformer (raster X). novel angular encoding (AE, eq. 3) 으로 yaw 의 $2\pi$ 주기 overload 회피. less than 3M params (hidden 128) — TrafficSim/Symphony 보다 magnitude 작음
- Training: BC + closed-loop BPTT 80 step. 두 conditioning:
  - Destination $g$: 각 agent 의 next goal lane segment (multi-class CE)
  - Personality $z$: 16-dim Gaussian latent CVAE (time-invariant — TrafficSim 의 time-varying 과 대비)
- Loss: smoothed L1 reconstruction + KL (free nats clipping) + destination CE
- 속도: 2080Ti single GPU 에서 16 simulation parallel, rollout step ~10ms

Benchmark (Table I, WOMD test marginal motion prediction leaderboard):

| Method | mAP ↑ | minADE | minFDE | overlap |
|---|---|---|---|---|
| DenseTNT | 0.328 | 1.039 | 1.551 | 0.178 |
| SceneTransformer | 0.279 | 0.612 | 1.212 | 0.147 |
| TrafficBots (a priori) | 0.212 | 1.313 | 3.102 | 0.145 |

저자가 명시: TrafficBots 의 minADE/minFDE 가 open-loop method 보다 worse — "auto-regressive policy rollout 의 compounding error 때문" — closed-loop method 의 알려진 trade-off. 하지만 overlap rate 0.145 로 가장 낮음 (가장 collision-free).

Table II ablation (WOMD valid): vs SimNet (BC w/o personality+destination) vehicle col 13.6% → TrafficBots 11.5%. vs TrafficSim re-impl (w/o dynamics) col 48.0% → TrafficBots 11.5% — destination + personality + world-model training 의 명백한 개선.

GameFormer 한계 매핑:
- covariate shift: closed-loop BPTT 80 step
- multi-modality / sub-optimal: time-invariant personality CVAE
- controllability: destination/personality sim-time tunable (GameFormer 가 갖지 못한 capacity)

자체 한계: minADE/minFDE 가 open-loop method 보다 worse (compounding error trade-off), player agent (planner) training 미제공, personality 의 time-invariant 가정.

---

## 4. Paradigm 2 — RL with Explicit Reward

### 4.1. 핵심 아이디어

문제: BC 는 "WOMD 인간이 했다 = 합리적이다" 의 implicit 가정. sub-optimal 인간 행동 (졸음, 부주의, 위반) 까지 "rational" 로 학습. GameFormer 의 supervision 한계.

해결: reward function (collision avoidance, kinematic feasibility, lane adherence, ride comfort 등) 을 명시 → RL 또는 return-conditioned 로 그 reward 직접 최대화. 인간 행동을 prior 로 쓰되 안 따라할 수 있음.

### 4.2. CtRL-Sim (Rowe et al., CoRL 2024)

- arxiv: [2403.19918](https://arxiv.org/abs/2403.19918)
- Affiliation: Mila + Université de Montréal + Princeton + Torc Robotics
- Code: [github.com/montrealrobotics/ctrl-sim](https://github.com/montrealrobotics/ctrl-sim)
- Architecture: encoder-decoder Transformer (Decision Transformer 계열). agent-major + timestep-major autoregressive token sequence: `(state, goal), (returns-to-go for C reward components), action` 반복
- Reward 3 component factorization:
  - $R_{goal}$: 1 m 이내 goal 도달 binary
  - $R_{veh-veh}$: vehicle 간 collision binary
  - $R_{veh-edge}$: road edge collision binary
- Training: return-conditioned offline RL with exponential tilting
  - Sampling rule: $G'_t \sim p_\theta(G_t|s_t,s_G) \cdot \exp(\kappa \cdot G_t)$
  - $\kappa > 0$: dataset 평균보다 우수, $\kappa < 0$: 평균보다 나쁜 (adversarial). 검증된 $\kappa$ 범위 -25 ~ 25
- Sub-optimal demo 처리: dataset 그대로 학습하지만 sampling 시 $\kappa > 0$ 로 "평균보다 우수" 영역만 mode 로 끌어올림. mathematically principled
- Dataset: WOMD (134K train / 9.7K val / 2.5K test scene). Nocturne + Box2D physics simulator

Benchmark (WOMD test, 1000 scene):

| Metric | CtRL-Sim ($\kappa>0$) | IL baseline | DT (max return) |
|---|---|---|---|
| FDE (m) | 2.04 | 1.95 | 3.07 |
| Collision rate (%) | 5.3 | 5.8 | 5.3 |
| Offroad rate (%) | 11.0 | 12.1 | 11.0 |
| JSD ($\times 10^{-2}$) | 7.9 | 8.3 | 8.4 |

CTG++ (diffusion baseline) 대비 5.4x 빠름. Negative tilting 만으로 추가 collision 238 건/1000 scene 자동 생성 (CAT-style adversarial finetuning 호환).

Controllability: per-component $\kappa$ 변경으로 deploy time 행동 조절. aggressive: $\kappa_{veh-veh} < 0, \kappa_{goal} > 0$. cautious: $\kappa_{veh-veh} > 0, \kappa_{veh-edge} > 0$.

GameFormer 한계 매핑:
- "WOMD = rational" 가정 극복: positive exponential tilting 으로 mode 를 "평균보다 우수" 영역에 집중
- sub-optimal bad pattern 회피: principled mechanism (mathematical guarantee)
- safety-critical scenario 자동 생성: negative tilting (GameFormer 에는 이 mode 자체가 없음)
- controllability: $\kappa$ deploy-time tunable

자체 한계: 3 dim binary reward 만 (comfort, traffic rule 미포함), goal reward 의 1m hard threshold, generation time 8.2s/scene 이 large-scale RL 학습에 prohibitive (저자 인정).

### 4.3. GUMP (Hu et al., ECCV 2024)

- arxiv: [2407.02797](https://arxiv.org/abs/2407.02797)
- Affiliation: Horizon Robotics
- 약어: Generative Unified Model for motion Planning (사용자 질문의 "Generalizable Unified Model-based Planner" 와 다름)
- Architecture:
  1. Static encoder (2D conv): raster map + route + static obstacle
  2. Dynamic tokenizer: object 별 (ID + category) key, state (x, y, heading, vx, vy, w, l) value
  3. Multimodal Causal Transformer (GPT-2 backbone + Gated Cross Attention)
  4. RNN decoder (stacked GRU): discrete state autoregressive
- Operating mode: full AR (scene 생성) + partial AR / NAR (frame 내 agent 병렬 decode)
- Variant: small / base / medium (LLM scaling law 검증). 정확한 size 미공개 [추측 base ~100M]
- Training: next-token prediction 만 (RL 직접 학습 X)
  - $P(s_0|c)$: scene generation, $P(s_t|s_{t-1},...,s_0,c)$: extrapolation
- Compounding error 완화: prediction chunking + temporal aggregation (decay rate $\gamma$)
- RL training environment 으로 활용: SAC in GUMP env 0.626 vs SAC in logsim 0.569

Benchmark:
- WOD scene generation: speed/size error 42.1% / 26.6% 감소 vs prior SOTA
- Waymo Sim Agents: minADE 1.590 (가장 낮음)
- nuPlan open-loop validation: 88.6 (CKS-1.5b 86.6 대비 우위)
- nuPlan Hard-14 closed-loop test: 77.77 (IDM env), 73.60 (GUMP env) — PDM baseline 상회

GameFormer 한계 매핑:
- "인간 = rational" 자체는 극복 X (imitation only objective)
- closed-loop covariate shift: temporal aggregation 으로 compounding error 완화
- RL training environment 으로 외부 reward 주입 가능 — 다음 단계 enabler

자체 한계: imitation only objective (GameFormer 와 같은 sub-optimal 학습 문제 잔존), raster map input (vectorized 대비 정보 손실, 저자도 vectorized 전환 언급), reward function design 회피했으나 결국 외부 정의 필요.

### 4.4. BC-SAC / "Imitation Is Not Enough" (Lu et al., Waymo 2022)

- arxiv: [2212.11419](https://arxiv.org/abs/2212.11419)
- Affiliation: Waymo Research + Google + UC Berkeley
- Architecture: dual actor-critic (TD3/SAC). Transformer observation encoder (vehicle state + roadgraph point + traffic signal + route goal). Actor: tanh-squashed diagonal Gaussian
- Reward (additive sum):
  - $R_{collision} = \min(d_{collision} - 1.0, 0)$ — 1m offset 미만 페널티
  - $R_{off-road} = \text{clip}(-1.0 - d_{to-edge}, -2.0, 0.0)$ — 1m 안쪽 페널티, -2 cap
- BC-SAC objective: $\mathbb{E}_{s,a \sim \pi}[Q(s,a) + H(\pi)] + \lambda \mathbb{E}_{s,a \sim D}[\log \pi(a|s)]$
  - SAC 의 Q + entropy 에 BC log-likelihood term 을 $\lambda$ 로 가중. in-distribution 에선 BC 안정화, OOD 에선 SAC 가 reward gradient 로 가이드
- Dataset: 100K+ miles real San Francisco urban driving, 6.4M training / 10K test segment, 10s @ 15Hz. Difficulty stratification: Top1% / Top10% / Top50%

Benchmark:
- Top1 (가장 어려운 시나리오): failure rate 38% 감소 vs IL-only baseline
- Pure RL (SAC) 대비 40% 개선
- Baseline: BC, MGAIL, SAC

GameFormer 한계 매핑:
- "WOMD = rational" 극복: BC term + 명시적 collision/off-road reward 가 sub-optimal 충돌 행동을 critic 으로 negative signal 처리
- safety-critical reward 명시: collision distance / road edge distance 가 explicit reward — GameFormer 에 없는 mechanism
- closed-loop training: SAC 자체가 covariate shift 해소 (저자 명시: "closed-loop training establishes causal relationship between observation, action, outcome")

자체 한계: non-reactive simulated agent (log replay NPC), $\lambda$ tuning 의 heuristic, traffic rule (신호, 우선순위) 미포함, evaluation dataset 비공개 (외부 재현/비교 어려움).

---

## 5. Paradigm 3 — Adversarial Scenario Generation

### 5.1. 핵심 아이디어

문제: WOMD 의 99.9% 는 normal 상황 — 사고 / 충돌 / 위반 거의 없음. safety-critical scenario 학습 불가능. GameFormer 는 이런 OOD 상황의 rationality 추론 capacity 없음.

해결: GT 분포 안에서 학습한 model 에 adversarial perturbation 가해 collision 직전 시나리오 생성 → 그 OOD 시나리오에 robust 한 policy 학습.

### 5.2. STRIVE (Rempe et al., CVPR 2022)

- arxiv: [2112.05077](https://arxiv.org/abs/2112.05077)
- Affiliation: NVIDIA Toronto AI Lab + Stanford + University of Toronto
- Code: [github.com/nv-tlabs/STRIVE](https://github.com/nv-tlabs/STRIVE)
- Architecture: graph-based CVAE (inter-agent graph network + autoregressive decoder). decoder 내부에 kinematic bicycle model 박혀 physical plausibility 강제
- Adversarial perturbation: latent space optimization. agent 별 latent $z$ + planner internal latent $z_{plan}$ optimize. trajectory space 가 아닌 latent space → traffic prior 가 자연스럽게 plausibility 강제
- Realism constraint: $L_{prior}$ (NLL of $z$ under prior) + $L_{init}$ (stay-close-to-init) + bicycle model + environment-collision penalty
- Generation: $\gamma$-weighted $\exp(-\text{distance})$ 로 collision attractive term. adversary 가 planner 와 가까울수록 weight 커지고 그 agent 가 planner 와 collide 하도록 gradient
- Planner interaction: black-box 처리 (learned planner-node decoder 가 differentiable approximation 역할)
- 2-stage: (1) adversarial optimization 으로 collision 생성, (2) solution optimization — 회피 가능한 ego trajectory 존재 확인. 없으면 unsolvable 폐기

Benchmark (Table 1, nuScenes 1200 scene):

| Planner | Collision Regular | Collision Challenging | Solution rate |
|---|---|---|---|
| Replay | 0% | 43.7% | 82.4% |
| Rule-based | 1.2% | 27.4% | 86.8% |

Table 2 (vs Bicycle baseline = AdvSim trajectory-space attack): STRIVE accel 0.98 m/s² (Bicycle 2.00), Env Coll 10.8% (Bicycle 16.5%), NN Dist 0.72m, NLL 323.4 (Bicycle 962.9). Bicycle baseline ~40x 느림.

Table 3 (rule-based planner improvement): None (regular-tuned) Coll 4.6 / 68.6%; +Challenging data 6.0 / 51.4%; +Extra learned mode 4.6 / 54.3% — 14.3pp 감소 + regular 유지.

GameFormer 한계 매핑:
- WOMD long-tail 부재: nuScenes collision-free regular scene 을 perturb → collision scene 합성 (다른 dataset 에서 동일 문제 풀이)
- safety-critical reasoning: rule-based planner 의 fundamental design limitation 노출 → architectural improvement 도출
- distribution shift robustness: solution optimization 으로 useful OOD 만 학습

자체 한계: perfect perception 가정 (planner only attack), vehicle 만 collide 대상 (pedestrian/cyclist X), regular vs challenging tuning balance 어려움.

### 5.3. KING (Hanselmann et al., ECCV 2022)

- arxiv: [2204.13683](https://arxiv.org/abs/2204.13683)
- Affiliation: Tübingen + MPI Intelligent Systems + Mercedes-Benz AG R&D
- Code: [github.com/autonomousvision/king](https://github.com/autonomousvision/king)
- Architecture: 학습된 prior 없음 (STRIVE/CAT 와 가장 큰 차이). kinematic bicycle model $\kappa$ 를 differentiable proxy
- Search space: 각 adversarial agent 의 timestep 별 (steering, acceleration) action sequence
- Adversarial perturbation: gradient-based optimization through bicycle proxy. simulator + sensor render $R$ non-differentiable 이지만 "direct path" (state → next state via $\kappa$) 만으로 충분
- Realism: collision attractive + adversary-adversary repulsive + road boundary repulsive. "human-like" 가 아닌 "물리적으로 가능한"
- Planner interaction: white-box (driving policy forward + backward 필요)

Benchmark (Table 2, CARLA Town03-06, 80 route × 3 density, 180s budget):

| Method | Overall CR (%) | t50% (s) | s/it |
|---|---|---|---|
| Random Search | 66.67 | 9.66 | 1.38 |
| CMA-ES | 68.33 | 8.17 | 1.40 |
| KING | 82.50 | 7.78 | 1.90 |

Table 3 (AIM-BEV fine-tuning):
- No Fine-tuning: held-out KING CR 100%, Town10 CR 17.48%, DS 86.74
- $D_{crit} \cup D_{reg}$: 28.57% / 8.13% / 90.20 — 100→28.57% (KING) + 17.48→8.13% (Town10), DS +3.46

GameFormer 한계 매핑:
- long-tail 부재: hand-crafted scenario 자동화 ("scenarios have to be manually re-tuned to each driving agent" 문제 해결)
- safety-critical reasoning: t-bone, merge cut-in, unprotected turn 등 specific failure mode 에 fine-tune
- distribution shift robustness: D_crit 추가로 100→28.57% 감소

자체 한계: 학습 prior 없음 (human-like plausibility 보장 약함, kinematic feasibility 만), adversary trajectory fixed after optimization, CARLA 만 (sim2real gap), pedestrian/cyclist 미포함.

### 5.4. CAT (Zhang et al., CoRL 2023)

- arxiv: [2310.12432](https://arxiv.org/abs/2310.12432)
- Affiliation: Tsinghua + UCLA + Edinburgh (Bolei Zhou lab)
- Code: [github.com/metadriverse/cat](https://github.com/metadriverse/cat)
- Architecture: DenseTNT (off-the-shelf motion forecasting) 를 traffic prior
- Adversarial perturbation: probabilistic factorization + resampling (gradient X — KING/STRIVE 와 가장 큰 차이). $M=32$ opponent trajectory candidate sampling → $N=5$ ego rollout queue 와 collision likelihood $\alpha^k$ ($\alpha=0.99$, $k$=collision step) → posterior maximize
- Realism: candidate 가 motion prior sample → 자동으로 plausible. resampling 만 → in-distribution 성 강함
- Generation: NHTSA pre-crash typology 9 종 (Right Turn, Left Turn, U-Turn, Rear-End, Emergent Brake, Lane Change, Cross Paths, Run-Off-Road, Opposite Direction)
- Planner interaction: black-box (forward 만, backward 불필요). RL/IL/HF 모두 호환
- Closed-loop integration: TD3 RL agent scratch 학습. 매 episode 마다 현재 policy 의 rollout 으로 새 adversarial scenario 생성 → 학습 environment (진정한 closed-loop)

Benchmark (Table 1, attack success rate, 100 test scene):

| Method | Replay | IDM | Pretrained ego | time (s) |
|---|---|---|---|---|
| Raw | 0% | 34% | 14% | — |
| M2I (adv) | 47% | 41% | 19% | 0.41 |
| STRIVE | 85% | 82% | 66% | 153.10 |
| CAT (N=1) | 91% | 71% | 62% | 0.66 |
| CAT (N=5) | 91% | 86% | 69% | 3.34 |

CAT vs STRIVE: competitive attack success + ~46x faster.

Table 2 (TD3 driving policy, held-out test):
- No Adv (replay): RC 72.91%, Crash 19.89% (log) / 63.48%, 43.33% (safety-critical)
- Closed-loop Adv (CAT): RC 72.47% / 67.62%, Crash 13.43% / 28.15%
- 효과: log crash -6.46pp, safety-critical crash -15.18pp vs No Adv. RC 거의 유지 → "in-dist 성능 안 깎음"

GameFormer 한계 매핑 (가장 직접적):
- WOMD long-tail 부재: CAT 가 정확히 WOMD 를 base 사용. raw collision rate 0% → CAT 로 91%. GameFormer 학습 distribution 위에 직접 adversarial layer
- safety-critical reasoning: closed-loop online — 학습 진행에 따라 policy weakness 변화 → adversarial 도 재생성. 1-shot offline (KING/STRIVE) 보다 long-horizon robustness ↑
- distribution shift robustness: in-dist 성능 유지하면서 OOD robustness ↑. Rule-based Adv 의 over-conservative 문제 회피
- 비용: 3.34s/scene → million-episode RL 학습 안에서 closed-loop 사용 가능 (KING/STRIVE 는 비용 때문에 사실상 offline)

자체 한계: vehicle adversary 만, 500 scene 제한, TD3 만 검증, MetaDrive sim 만 평가 (real-world 미검증), motion prior (DenseTNT) 의 mode coverage 한계 → diversity 가 prior 에 묶임.

---

## 6. Paradigm 4 — World Model + Token Autoregressive

### 6.1. 핵심 아이디어

문제: BC 의 deterministic (또는 fixed M-mode) prediction 은 진짜로 가능한 시나리오 분포가 아님. multi-modal 도 mode 수가 적고 학습 분포 안. GameFormer 의 M=6 이 그 대표.

해결: world model 또는 token autoregressive 로 next-state distribution 자체 modeling → 다양한 plausible future sampling. discrete tokenization + next-token prediction 이 fixed M-mode 의 본질적 한계 (mode collapse, mode 부족) 를 vocabulary size 와 sampling temperature 로 우회.

### 6.2. GAIA-1 (Wayve, 2023 tech report)

- arxiv: [2309.17080](https://arxiv.org/abs/2309.17080)
- Affiliation: Wayve (UK 자율주행 startup)
- Architecture: 6.5B autoregressive transformer (world model) + 2.6B video diffusion decoder = 9B+ params
- Tokenization: vector quantization 으로 video frame → discrete image token
- Input: 3 modality (video frame + free-form text + ego action [steering/speed/curvature])
- Output: single front-camera video (multi-camera X — Wayve acknowledged limitation)
- Pipeline: encoders → shared token space → autoregressive transformer next image token → diffusion decoder pixel
- Training: next-token prediction (LM 동일) + diffusion denoising loss
- Dataset: Wayve proprietary 4,700 hours London UK driving (2019~2023)

Generative capability:
- "long, diverse driving scenes entirely from imagination" (Wayve blog) — sample 마다 다른 multi-modal future
- Controllability: text prompt (weather/lighting/traffic), action conditioning (ego speed/steering), video prompt (seed scene)
- Distribution coverage: rare scenario (pedestrian abrupt crossing, atypical weather) 도 prompt 로 sampling

Benchmark: WaymoSim Agents 등 정량 trajectory benchmark 미참가 (image-space generative 라 metric 자체 다름). 자체 qualitative evaluation 위주.

GameFormer 한계 매핑:
- fixed M=6 극복: stochastic next-token sampling 으로 $M=\infty$
- distribution diversity: text + action prompt 로 long-tail scenario 직접 sampling
- "WOMD 모방" BC 한계: unsupervised sequence modeling 으로 distribution 자체 학습 (LM "다음 단어" 와 동일)

자체 한계: 9B params 학습에 large-scale infra, autoregressive inference time 큼, single front-camera 만, safety guarantee 없음 (hallucination 가능), 실제 deploy = research/PoC stage.

### 6.3. DriveDreamer / DriveDreamer-2 (Wang et al., ECCV 2024 / AAAI 2025)

- DriveDreamer arxiv: [2309.09777](https://arxiv.org/abs/2309.09777), ECCV 2024
- DriveDreamer-2 arxiv: [2403.06845](https://arxiv.org/abs/2403.06845), AAAI 2025
- Affiliation: GigaAI + Tsinghua [추측]
- Architecture:
  - DriveDreamer (v1): diffusion-based video generation (Stable Diffusion lineage) + ActionFormer (latent space future structural feature) + Auto-DM (future driving video gen). input: reference frame + HDMap + 3D bbox + text + historical action
  - DriveDreamer-2: + LLM interface (user query → agent trajectory → traffic-rule-compliant HDMap 합성) + Unified Multi-View Model (multi-camera temporal-spatial coherence)
- Tokenization: discrete token X (latent diffusion 기반)
- Parameter [추측 ~1B Stable Diffusion backbone]
- Training: diffusion denoising loss + structural constraint loss. 2-stage (structured constraint → future state anticipation)
- Dataset: nuScenes (1000 scene, 20s, Boston/Singapore)

Generative capability:
- diffusion sampling per-sample diversity
- Controllability multi-modal: text ("rainy night"), action (ego control), HDMap, 3D bbox
- DriveDreamer-2 highlight: "vehicles abruptly cut in" 같은 uncommon scenario user query 직접 생성 — long-tail safety scenario 합성

Benchmark (nuScenes video generation):
- DriveDreamer-2: FID 11.2, FVD 55.7 (이전 SOTA 대비 30% / 50% 개선)
- generated video 로 perception model 학습 시 downstream 성능 향상
- WaymoSim Agents 미참가 (motion forecasting benchmark 가 아님)

GameFormer 한계 매핑:
- distribution diversity: LLM query 로 명시적 long-tail scenario 합성
- fixed M-mode 극복: diffusion sampling stochasticity
- "WOMD 모방" 한계: BC 가 아닌 generative — 학습 시 못 본 weather/time/agent layout 합성

자체 한계: diffusion inference cost 큼 (real-time 부적합), nuScenes scale (1000 scene) 만 검증, LLM trajectory 합성의 physics-realism 평가 metric 부족, closed-loop control 미지원, safety guarantee 없음.

### 6.4. Trajeglish (Philion et al., ICLR 2024)

- arxiv: [2312.04535](https://arxiv.org/abs/2312.04535)
- Affiliation: NVIDIA Toronto AI Lab + University of Toronto
- Architecture: GPT-style encoder-decoder, autoregressive in time. 2 encoder + 6 decoder layer, hidden 512 (small model — language model 대비 미니어처)
- Tokenization: k-disks 알고리즘 — trajectory action → discrete vocabulary
  - Vocabulary size: 384
  - Discretization error: expected 1 cm, average corner distance 1.18 cm
  - 0.5초 간격 action token
- Encoder: VectorNet (map encoding) + Latent Query Attention (scene init)
- Capacity: agent up to 24, map object up to 96, range 60m
- Intra-timestep agent interaction (single-pass 가 아닌 agent-by-agent autoregressive)
- Training: dataset WOMD (~1.5B token), loss next-token cross-entropy. closed-loop reactive 10Hz, privileged future info 없음

Benchmark — WaymoSim Agents 2023:
- Realism meta 0.5339 (paper claim 시점 prior SOTA 대비 +3.3%)
- Interactive 0.5811 (+9.9%)
- Kinematic 0.4019, Map 0.6667, minADE 1.872m
- 비교: Wayformer, MultiPath++, MTR, MTR++, Joint-MultiPath++, MVTA, MVTE — 모두 fixed M-mode trajectory regression. Trajeglish 가 처음으로 discrete sequence modeling 적용

주의: Trajeglish 의 2023 benchmark 1위 claim 은 paper 시점. 실제 Sim Agents Challenge 2023 우승은 MVTA / MVTE (closed-loop training 으로 우위).

GameFormer 한계 매핑:
- fixed M=6 → vocabulary 384 next-token sampling 으로 사실상 무한 mode
- joint multi-agent: GameFormer 의 marginal+game refinement 보다 native joint AR 로 intra-timestep interaction → interaction metric +9.9% (가장 큰 gain)
- BC 한계: 여전히 supervised next-token (BC 의 sequence version) 이지만 distribution match metric 우위 (point-wise L2 가 아닌 distribution-level loss)

자체 한계: "severely data-constrained" (저자 본인 명시) — 동일 param LM 대비 데이터 부족, scaling law 미충족. kinematic realism 이 discretization 으로 손해. text controllability 없음. safety guarantee 없음.

### 6.5. SMART (Wu et al., NeurIPS 2024) — WaymoSim Agents Challenge 2024 우승

- arxiv: [2405.15677](https://arxiv.org/abs/2405.15677)
- Affiliation: SenseTime Research + Tsinghua (Wei Wu)
- Code: [github.com/rainmaker22/SMART](https://github.com/rainmaker22/SMART)
- Architecture: decoder-only transformer (GPT-style, encoder-decoder X — Trajeglish 와 차이). 3 variant 7M / 26M / 101M
- Tokenization:
  - Motion vocabulary 512~2048 (모델 크기에 따라)
  - Road vocabulary 1024 (모든 변형 동일)
  - k-disks clustering (Trajeglish 와 유사 but vocab size 더 큼)
  - 0.5초 간격
- Input: vectorized map (road token) + agent trajectory (motion token) joint sequence
- Output: 모든 agent 의 next motion token (joint multi-agent)
- Training: WOMD 0.18B + NuPlan 0.13B + proprietary 0.68B = ~1B motion token (GPT-style scaling 입증). single pretraining (fine-tuning X)

Benchmark — WaymoSim Agents 2024 leaderboard 1위:

| Variant | realism meta | kinematic | interactive | map | minADE | inference (ms/step) |
|---|---|---|---|---|---|---|
| SMART-7M | 0.7591 | — | — | — | — | 17.21 |
| SMART-101M | 0.7614 | 0.4786 | 0.8066 | 0.8648 | 1.3728 | 10~46 |

- SMART-7M (0.7591) vs Trajeglish 2023 (0.5339 paper claim) 약 +11pp
- WaymoSim Agents Challenge 2024 공식 우승 (CVPR 2024 Workshop)
- Zero-shot generalization: NuPlan train → WOMD test 0.7210 realism (cross-dataset transfer 입증)

GameFormer 한계 매핑:
- fixed M=6 → vocab 512~2048 stochastic sampling
- distribution diversity: realism 0.7614 (fixed-M model 도달 불가능 영역)
- joint multi-agent native: GameFormer 의 marginal+game refinement 보다 단순/강력
- BC 한계: cross-dataset zero-shot 0.7210 (WOMD 만 학습한 GameFormer 는 cross-dataset evaluation 자체 불가)
- scaling law: 7M → 101M 키울수록 일관된 개선 (GameFormer 류 fixed-M 은 scaling 효과 saturated)

자체 한계: text/natural language controllability 없음, pixel-space scene generation 안 함 (vectorized motion 만), safety guarantee 없음, discretization 의 kinematic realism 한계, proprietary 0.68B 비공개 (open replication 시 0.31B 만, scale 재현 어려움).

---

## 7. 종합 비교 — GameFormer 한계와 paradigm 매핑

### 7.1. 한계별 paradigm 매핑

| GameFormer 한계 | 가장 직접 해결 paradigm | 대표 paper |
|---|---|---|
| pure GT imitation | RL with reward | CtRL-Sim, BC-SAC |
| sub-optimal demonstration 학습 | RL with reward (exponential tilting) | CtRL-Sim |
| open-loop covariate shift | closed-loop reactive | TrafficSim, Symphony, TrafficBots |
| safety-critical scenario 부재 | adversarial generation | CAT (WOMD direct), STRIVE, KING |
| fixed M=6 mode | world model + token AR | Trajeglish, SMART |
| no controllability | RL with reward / world model | CtRL-Sim ($\kappa$), GAIA-1 (text) |
| perception oracle 가정 | world model end-to-end | GAIA-1, DriveDreamer |

### 7.2. 정량 비교 — distribution match vs collision rate

WOMD-direct evaluation (가능한 paper 만):

| Paper | Paradigm | Dataset | metric | 수치 |
|---|---|---|---|---|
| GameFormer | open-loop BC + game | WOMD | minADE / minFDE | 0.92 / 1.94 (paper Joint M=6) |
| TrafficBots | closed-loop BPTT | WOMD test | mAP / overlap | 0.212 / 0.145 |
| CAT | adversarial closed-loop | WOMD 500 scene | TD3 crash 감소 | log -6.46pp / safety-critical -15.18pp |
| CtRL-Sim | RL exp-tilt | WOMD 1000 test | collision / FDE | 5.3% / 2.04 |
| Trajeglish | token AR | WOMD | realism meta | 0.5339 |
| SMART-101M | token AR | WOMD | realism meta | 0.7614 (Sim Agents 2024 1위) |

GameFormer 직접 비교는 어려움 (각 paper 의 baseline 이 다른 fixed-M model). 다만 SMART-7M 0.7591 vs Trajeglish 0.5339 (+11pp) 의 paradigm shift 효과는 명확. fixed-M 류는 token AR 의 이 metric 영역 도달 불가.

### 7.3. WOMD-direct follow-up 가장 가까운 4 paper

GameFormer 와 동일 WOMD dataset 사용 + 직접 적용 가능한 후속 paradigm:

1. **CAT** (CoRL 2023) — WOMD 500 scene + closed-loop adversarial. GameFormer 의 학습된 prediction prior 자체를 motion prior 로 재활용 가능 [추측]. 가장 직접적 follow-up.
2. **TrafficBots** (ICRA 2023) — WOMD only + closed-loop BPTT + destination/personality conditioning. controllability 추가.
3. **GUMP** (ECCV 2024) — WOD + nuPlan. closed-loop simulation environment 제공 → GameFormer 위에 RL 학습 layer 추가 가능.
4. **SMART** (NeurIPS 2024) — WOMD + NuPlan + proprietary. paradigm 자체를 token AR 로 변경. fixed-M 의 scaling saturation 회피.

### 7.4. Timeline — paradigm shift 의 흐름

```
2020 이전:   open-loop fixed-M BC
              (Scene Transformer, MultiPath++, Wayformer, MTR — GameFormer 의 직접 predecessors)

2021:        TrafficSim — closed-loop differentiable sim 의 시초

2022:        Symphony (Waymo) — beam search distillation
             TrafficBots 에 영향
             STRIVE — adversarial latent perturbation
             KING — gradient-based collision attack

2023:        TrafficBots — vectorized + BPTT + personality
             CAT — closed-loop adversarial WOMD
             BC-SAC (Waymo) — IL+RL hybrid
             GAIA-1 (Wayve) — 9B world model

2023.10:     GameFormer (ICCV) — 발표 시점에 이미 closed-loop / adversarial / world model paradigm 으로 field 이동 중

2023~2024:   Trajeglish (ICLR) — 처음으로 discrete token AR 적용
             SMART (NeurIPS) — token AR scaling, Sim Agents 2024 우승

2024:        CtRL-Sim (CoRL) — return-conditioned controllable
             GUMP (ECCV) — generative + RL env
             DriveDreamer-2 (AAAI 2025) — LLM-controlled video generation
```

GameFormer 는 사실상 "open-loop BC + game-theoretic architectural prior" paradigm 의 마지막 세대. 발표 동시점에 후속 paradigm 들이 이미 존재했고, 1~2 년 만에 Sim Agents leaderboard 등에서 token AR + closed-loop paradigm 에 밀림.

## 8. 공통 unsolved 영역

4 paradigm 모두 다루지 못한 부분:

1. **Reward function design 자동화** — RL paradigm 도 reward 정의는 hand-crafted. specification 의 어려움
2. **End-to-end perception 통합** — GAIA-1 / DriveDreamer 가 일부 다루지만 closed-loop control 까진 X
3. **Sim2real gap** — 모든 simulator 평가는 real-world 와 갭 존재
4. **Multi-objective reward** — comfort, traffic rule (신호, 우선순위, lane discipline) 등의 통합
5. **Pedestrian / cyclist 통합 modeling** — vehicle 만 다루는 paper 다수
6. **Safety / collision-free guarantee** — generative paradigm 모두 hallucination 가능, formal verification 없음
7. **Scaling law 의 open replication** — proprietary dataset (Wayve, Waymo, SenseTime) 비공개
8. **Long-horizon stability** — closed-loop 라도 30s+ rollout 시 drift

## 9. 결론 — GameFormer 의 위치

GameFormer 의 contribution 을 정직히 위치시키면:

- **paradigm 시점**: open-loop BC + game-theoretic prior — 발표 시점에 이미 outdated paradigm
- **architectural contribution**: level-k joint reasoning architecture — 그 자체는 흥미로운 inductive bias 지만 supervision paradigm 의 한계 안에서만 효과
- **academic value**: WOMD validation set 안에서 SOTA — 단 closed-loop / adversarial / world model 평가 없음
- **industrial value**: 후속 paradigm (CAT, TrafficBots, SMART) 의 building block 으로 활용 가능 — game-theoretic prior 자체는 reusable

진짜 자율주행 fundamental 해결을 위해서는:
- **closed-loop training** (TrafficSim, Symphony, TrafficBots) — covariate shift 해결
- **explicit reward** (CtRL-Sim, BC-SAC) — sub-optimal demonstration 회피
- **adversarial scenario** (CAT, STRIVE, KING) — long-tail safety-critical 학습
- **world model / token AR** (GAIA-1, Trajeglish, SMART) — distribution diversity + scaling

이 4 가지 paradigm 의 통합이 다음 단계. 개별 paradigm 은 단점 있지만 합쳐지면 GameFormer 의 8 가지 한계 대부분을 다룰 수 있음. 산업체 deploy (Waymo, Wayve, Tesla 등) 가 이 통합 방향으로 진행 중.

## 10. 참고 자료

### Closed-loop reactive
- TrafficSim: [arxiv 2101.06557](https://arxiv.org/abs/2101.06557), [CVPR 2021 PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.pdf)
- Symphony: [arxiv 2205.03195](https://arxiv.org/abs/2205.03195), [Waymo Research](https://waymo.com/research/symphony-learning-realistic-and-diverse-agents-for-autonomous-driving-simulation/)
- TrafficBots: [arxiv 2303.04116](https://arxiv.org/abs/2303.04116), [project](https://zhejz.github.io/trafficbots), [code](https://github.com/SysCV/TrafficBots)

### RL with reward
- CtRL-Sim: [arxiv 2403.19918](https://arxiv.org/abs/2403.19918), [project](http://montrealrobotics.ca/ctrlsim/), [code](https://github.com/montrealrobotics/ctrl-sim)
- GUMP: [arxiv 2407.02797](https://arxiv.org/abs/2407.02797), [code](https://github.com/HorizonRobotics/GUMP/)
- BC-SAC: [arxiv 2212.11419](https://arxiv.org/abs/2212.11419), [Waymo](https://waymo.com/research/imitation-is-not-enough-robustifying-imitation-with-reinforcement-learning/)

### Adversarial
- STRIVE: [arxiv 2112.05077](https://arxiv.org/abs/2112.05077), [project](https://research.nvidia.com/labs/toronto-ai/STRIVE/), [code](https://github.com/nv-tlabs/STRIVE)
- KING: [arxiv 2204.13683](https://arxiv.org/abs/2204.13683), [code](https://github.com/autonomousvision/king)
- CAT: [arxiv 2310.12432](https://arxiv.org/abs/2310.12432), [project](https://metadriverse.github.io/cat/), [code](https://github.com/metadriverse/cat)

### World model + token AR
- GAIA-1: [arxiv 2309.17080](https://arxiv.org/abs/2309.17080), [Wayve](https://wayve.ai/science/gaia/)
- DriveDreamer: [arxiv 2309.09777](https://arxiv.org/abs/2309.09777), [project](https://drivedreamer.github.io/)
- DriveDreamer-2: [arxiv 2403.06845](https://arxiv.org/abs/2403.06845)
- Trajeglish: [arxiv 2312.04535](https://arxiv.org/abs/2312.04535), [NVIDIA project](https://research.nvidia.com/labs/toronto-ai/trajeglish/)
- SMART: [arxiv 2405.15677](https://arxiv.org/abs/2405.15677), [code](https://github.com/rainmaker22/SMART)

### Benchmark
- Waymo Open Sim Agents Challenge: [arxiv 2305.12032](https://arxiv.org/abs/2305.12032), [2024 challenges](https://waymo.com/blog/2024/03/2024-waymo-open-dataset-challenges/)
