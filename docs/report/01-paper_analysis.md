# 01. Paper 분석 — GameFormer

원본 논문:
- Huang, Liu, Lv. *"GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving."* ICCV 2023. [[arXiv:2303.05760]](https://arxiv.org/abs/2303.05760)
- 저자: Nanyang Technological University, AutoMan Research Lab

## 1. 문제 정의

자율주행 차량의 motion prediction + planning 은 다른 agent (차량, 보행자, 자전거) 와의 **상호작용** 을 고려해야 한다. 기존 방법론의 한계:

- **Marginal prediction**: 각 agent를 독립적으로 예측 → 실제 도로에서 차량들이 서로의 의도를 보고 행동을 조정한다는 점을 무시. mode collapse + 비현실적 trajectory.
- **Single-shot interaction**: 한 번의 attention 으로 상호작용 모델링 → 양쪽이 동시에 의사결정한다는 가정. 인간 운전자의 실제 추론 (상대가 어떻게 반응할지 예측 → 내 행동 결정 → 상대도 같은 추론) 을 반영하지 못함.

## 2. Contribution

1. **Game-theoretic k-level reasoning**: 차량 간 상호작용을 *level-k thinking* (행동경제학) 으로 모델링. level 0 은 다른 agent 무시한 초기 추측, level k 는 "level k-1 에서 다른 agent 가 그렇게 예측했을 때 내 최선의 반응" 을 점진적으로 refining.
2. **Joint multi-modal prediction**: 한 model 이 ego + neighbors 의 미래 trajectory를 모두 multi-modal (M가지 가능성) 로 동시 예측 — agent 간 consistency 보장.
3. **Transformer-based architecture**: agent encoder (LSTM) + map encoder (PointNet) + fusion (Transformer encoder) + 계층적 decoder (InitialDecoder + InteractionDecoder × K).
4. **Open-loop planning**: prediction 을 그대로 planning 입력으로 사용하는 baseline 시스템 — paper 가 prediction quality 가 planning 의 핵심 라는 가설 검증.

## 3. Architecture 개요

```
                ┌────────────────────────────────────────────────┐
                │ Encoder                                        │
                │                                                │
   ego state ──▶│  AgentEncoder (LSTM) ──┐                       │
                │                        ├─▶ TransformerEncoder  │──┐
   neighbors ──▶│  AgentEncoder (LSTM) ──┤   (6 layers)          │  │
                │                        │                       │  │
   map lanes ──▶│  LaneEncoder           │                       │  │
   crosswalks ─▶│  CrosswalkEncoder ─────┘                       │  │
                └────────────────────────────────────────────────┘  │
                                                                    ▼
                ┌────────────────────────────────────────────────────┐
                │ Decoder (level k reasoning)                        │
                │                                                    │
                │  level 0:  InitialDecoder                          │
                │            (multi-modal query, no interaction)     │
                │                ▼                                   │
                │            (M trajectories per agent + scores)     │
                │                                                    │
                │  level 1..K: InteractionDecoder × K levels         │
                │              ▼                                     │
                │     - encode prev-level futures (FutureEncoder)    │
                │     - self-attention 으로 agent 간 interaction      │
                │     - cross-attention 으로 context 와 결합           │
                │     - GMM predictor → trajectories + scores       │
                │                                                    │
                └────────────────────────────────────────────────────┘
                                                                    ▼
              level_0_interactions, level_0_scores,
              level_1_interactions, level_1_scores,
              ...
              level_K_interactions, level_K_scores      (loss 는 모든 level 합산)
```

`model/GameFormer.py:114` 의 `class GameFormer` 가 위 구조 — Encoder + Decoder 두 단계.

## 4. Level-k reasoning 의 의미

Paper 의 핵심 가설:

> "*An agent's optimal action depends on the predicted actions of others, and those predictions depend recursively on what others believe this agent will do.*"

이를 수렴식 추론 (k-level) 으로 근사:
- **level 0**: 다른 agent 의 행동 예측 없이 각자 trajectory 가능성 추측. (= marginal prediction)
- **level 1**: level 0 에서 다른 agent 가 어떻게 행동할지 (M개의 weighted average) 보고, 그에 대한 best response 로 trajectory 재계산.
- **level k**: level k-1 의 결과를 기반으로 again refining.

`model/GameFormer.py:100-109` 의 loop:
```python
for k in range(1, self._levels+1):
    interaction_decoder = self.interaction_stage[k-1]
    results = [interaction_decoder(i, ...) for i in range(N)]
    ...
    decoder_outputs[f'level_{k}_interactions'] = last_level
    decoder_outputs[f'level_{k}_scores'] = last_scores
```

각 level 의 출력이 모두 보존됨 → loss 계산 시 모든 level 의 prediction 에 GT trajectory 와의 loss 를 부과 (deep supervision). 이 통해 level k 가 점진적으로 refining 하도록 훈련.

paper 에서 reasoning level K=2 (interaction prediction) 또는 K=4 (open-loop planning) 사용. K 가 늘면 정확도 ↑ 이지만 compute 도 ↑.

## 5. Joint vs Marginal model

paper Table 1 에서 두 variant 비교:

| variant | M (modes per agent) | minADE | minFDE | Miss Rate | mAP |
| --- | --- | --- | --- | --- | --- |
| Joint | 6 | 0.9161 | 1.9373 | 0.4531 | 0.1376 |
| Marginal (EM ensemble) | 64 | 0.9721 | 2.2146 | 0.4933 | 0.1923 |

- **Joint (M=6)**: 각 agent 의 6 modal trajectory 를 다른 agent 의 6 modal 과 jointly 예측. mode 가 적지만 agent 간 consistency 좋음 → 낮은 ADE/FDE/MR.
- **Marginal (M=64)**: 각 agent 의 64 modal trajectory 를 독립 예측 → mode 다양성 ↑ → 높은 mAP (각 mode 의 confidence weighted accuracy). 그러나 agent 간 일치성 부족 → ADE/FDE 더 나쁨.

원본 repo 는 **Joint model 만 공개**. Marginal 모델 + EM ensemble 은 미공개 (재현 불가).

## 6. Open-loop planning component

interaction prediction 위에 ego 차량의 trajectory 를 reference path 와 결합하여 planning 출력 생성. open-loop = 환경의 reaction 없이 한 번에 trajectory 를 생성.

paper Sec 4.2.2:
> "We randomly select 10,000 20-second scenarios from the training set, where 9,000 of them are used for training and the remaining 1,000 for validation."

`open_loop_planning/data_process.py` 가 위 sampling 을 구현 (특정 scene 에서 ego 의 long-horizon trajectory 추출).

## 7. Evaluation metrics (Waymo Motion Prediction Challenge 기준)

- **minADE**: 모든 modal 중 GT 와 가장 가까운 trajectory 의 average displacement error (전 timestep 평균 거리, m)
- **minFDE**: 같은 trajectory 의 final displacement error (마지막 timestep 거리, m)
- **Miss Rate**: 모든 modal 의 endpoint 가 lateral 1m / longitudinal 2m 안에 들지 못한 비율
- **mAP**: confidence weighted Average Precision (Mean Average Precision over recall thresholds)

이 4 metric 은 모두 **lower is better** (mAP 만 higher is better). validation set 에서 측정.

## 8. 재현 의의

이 reproduction 의 목적은 paper 의 학습 path 를 현대 환경 (torch 2.x, cu118) 에서 그대로 재현하여 baseline metric 이 일관되게 나오는지 확인하는 것. 모델 구조 / 학습 logic 변경 없음 — 환경 호환성과 cloud GPU 운영 측면만 다룸.
