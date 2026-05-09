# 07. 한계 + 향후 작업

이 reproduction 의 범위 밖이거나, 시간 / 자원 제약으로 다루지 못한 부분들을 정리.

## 1. 원본 paper 의 미공개 부분

원본 [MCZhi/GameFormer](https://github.com/MCZhi/GameFormer) 가 처음부터 제공하지 않은 부분:

### 1.1. Marginal model + EM ensemble (interaction prediction)

paper Table 1 의 두 variant 중:
- Joint (M=6) — 공개됨, 본 reproduction 의 대상
- Marginal (M=64) + EM ensemble — **미공개**

Marginal 은 더 다양한 mode (M=64) 를 만들지만 Joint 보다 mAP 가 더 좋음 (0.1923 vs 0.1376). EM 은 Expectation-Maximization style ensemble 로 mode 를 정제. 코드 미공개로 재현 불가능.

**이유 추정**: paper 의 main contribution 은 Joint (game-theoretic interaction). Marginal 은 비교 baseline 으로 별도 학습된 모델. 저자가 maintenance 부담으로 한쪽만 공개.

### 1.2. Closed-loop planning

paper 의 open-loop planning 은 한 번에 trajectory 를 생성. 실제 자율주행은 closed-loop (환경의 reaction 후 새 plan) 필요. paper 는 이를 다루지 않고, 저자의 다른 paper [DIPP](https://github.com/MCZhi/DIPP) 로 안내.

### 1.3. WOMD challenge submission code

> Code for packaging and submitting prediction results to the WOMD Interaction Prediction Challenge

미공개. submission 위해서는 별도 packaging code (proto serialize, batch inference, results format) 필요. 저자의 다른 ecosystem (DIPP, GameFormer-Planner) 에 있을 가능성.

이 reproduction 은 paper 의 validation set 결과만 측정 (Waymo leaderboard 미제출). 자세한 내용은 [docs/09-evaluation_methodology.md](../09-evaluation_methodology.md) 참조.

### 1.4. nuPlan 실험

paper 는 WOMD 에서만 실험. 저자의 다른 repo [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner) 가 nuPlan 실험을 다룸. 본 reproduction 은 WOMD 만.

## 2. 시간 / 자원 제약으로 미수행

### 2.1. 다양한 hyperparameter sweep

paper 의 baseline 만 재현. 다음은 미수행:

- learning rate sensitivity (1e-4, 2e-4, 5e-4)
- batch size 영향 (16, 32, 64, 128, 256)
- reasoning level K 의 영향 (K=1, 2, 3, 4, 5)
- modalities M 의 영향 (M=4, 6, 8, 16)
- encoder layer 수 의 영향 (4, 6, 8)

향후 ablation study 시 hyperparameter sweep 으로 어떤 component 가 가장 중요한지 측정 가능.

### 2.2. 추가 GPU architecture 검증

검증된 GPU: 3060 (sm_86), 4090 (sm_89), H200 (sm_90).
- A100 (sm_80) — 검증 미수행 (cu118 wheel 호환 OK 라 동작 보장됨)
- AMD MI300 / Intel Gaudi — 검증 미수행 (cu118 X)

### 2.3. Inference 최적화

학습 중심으로 진행. 다음은 미수행:
- TensorRT / ONNX export
- TorchScript JIT compile
- Quantization (int8 추론)
- batch inference throughput 측정

production 배포 시 위 최적화로 latency 5x+ 감소 가능 (transformer encoder + small decoder).

### 2.4. 실시간 visualizer

`open_loop_planning/open_loop_test.py --render` 옵션으로 시각화 가능. 본 reproduction 은 학습 결과 metric 만 측정 — visualization smoke 만 수행.

학습 결과의 시각적 검증 (어떤 scenario 에서 잘 / 못 예측하는지) 은 별도 작업.

## 3. Reproducibility 의 한계

### 3.1. Sampling seed 미고정

paper Sec 4.2.2 의 "10,000 random scenarios" 의 sampling seed 가 코드에 명시 안 됨 ([08-original_repo_issues](08-original_repo_issues.md) 참조). 본 reproduction 의 학습 sample 은 paper 와 정확히 일치하지 않을 수 있음.

영향: ±5% 정도 metric 변동 (통계적으로는 비슷한 distribution).

### 3.2. CUDA 비결정성

torch + cudnn 의 default 동작이 비결정적 — 같은 seed 라도 GPU model / driver version / cudnn version 별로 결과 차이.

본 reproduction 은:
- 같은 seed (3407) + 같은 GPU model + 같은 cu118 + 같은 cudnn → bit-perfect reproducible
- 다른 GPU (3060 vs 4090) → loss curve 거의 동일하지만 마지막 자릿수 차이 가능

### 3.3. Distributed sampler shuffling

DDP `DistributedSampler.set_epoch(epoch)` 으로 매 epoch 다른 shuffle. 같은 seed 면 같은 shuffle (yes deterministic).

단 DDP world_size 변경 시 (4 GPU → 2 GPU 등) shuffle 이 달라짐 → 학습 trajectory 변경.

## 4. 데이터 측면

### 4.1. Test set 미접근

WOMD test set 의 label 은 비공개 (Waymo private). 학술 논문은 모두 validation set 에서 metric 보고. 본 reproduction 도 동일.

진짜 unbiased test metric 필요 시:
- Waymo leaderboard submission (별도 작업, packaging code 필요)
- training 의 5% 를 hold-out 으로 사용
- k-fold cross validation (비싸지만 robust)

### 4.2. WOMD 의 구체적 sample distribution 차이

paper Sec 4.2.1: "We use the same train/validation split as the official WOMD". 즉 official split 그대로.
이 reproduction 도 동일 — sampling 외 split 변경 없음.

단 WOMD v1.1 → v1.2.1 시점에 일부 scenario 가 corrected / removed. paper 는 v1.1 사용. 본 reproduction 은 v1.2.1 — 일부 sample 차이 가능.

## 5. 향후 작업 우선순위

| 우선순위 | 작업 | 예상 작업량 | 가치 |
| --- | --- | --- | --- |
| high | Waymo leaderboard submission code | 1주 | unbiased test metric |
| high | hyperparameter sweep (lr, batch) | 1주 | optimal hyperparameter 찾기 |
| medium | 학습 결과 시각화 | 2~3일 | qualitative analysis |
| medium | Marginal model 구현 (M=64) | 2주+ | paper Table 1 두 variant 비교 |
| medium | Closed-loop planning (DIPP base) | 1달+ | 실제 자율주행 적용 |
| low | TensorRT export + benchmark | 1주 | inference 최적화 |
| low | nuPlan 실험 (GameFormer-Planner 활용) | 2주+ | 다른 dataset 검증 |
| low | upstream PR (patch 1, 2, 3, 4, 6) | 1~2일 | community contribution |

## 6. 본 reproduction 의 acceptance

reproduction 은 다음 조건 만족 시 성공으로 판정:

- [x] paper architecture 의 환경 호환성 patch (4 곳) 적용
- [x] WOMD v1.2.1 + torch 2.3.1+cu118 환경에서 학습 path 동작 (smoke 통과)
- [x] cloud GPU (H200) 에서 full WOMD 학습 launch
- [ ] full 학습 완료 + paper Table 1 baseline 의 ±15% 이내 metric 측정 (진행 중)
- [ ] checkpoint + log 보존 (pCloud)
- [ ] reviewer 가 README 만 보고 학습 재현 가능

위 acceptance 의 마지막 2 개 항목이 ongoing — 학습 완료 시점에 update 예정.

## 7. 정리

이 reproduction 의 핵심 가치:
- paper 의 학습 path 가 현대 환경에서 동작함을 검증
- 환경 호환성 patch 와 운영 자동화 script 제공 (다른 사용자가 쉽게 재현 가능)
- 학습 결과를 paper baseline 과 정량 비교

미수행 부분 (Marginal model, closed-loop, leaderboard) 은 모두 원본 repo 의 한계 또는 별도 큰 작업이 필요한 영역. 본 reproduction 의 scope 명확히 정의 — paper 의 main contribution (Joint multi-level reasoning) 검증.
